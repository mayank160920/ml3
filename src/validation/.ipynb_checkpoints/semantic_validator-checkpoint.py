"""
--------------------------------------
Section-wise Chain-of-Thought semantic validation engine (Layer 7).

Architecture
------------
For each section:
  1. Rule-based pre-normalisation (ValueNormalizer)
     → Entity pairs with identical normalised values → MATCH (no LLM call)
  2. CoT MLLM validation for remaining non-exact pairs
     → Single section-level prompt containing all unresolved pairs
     → Five-step reasoning: normalise → align → discrepancy → status → confidence
  3. Per-entity EntityValidationResult objects assembled
  4. SectionValidationResult appended to ValidationReport

Cost efficiency:
  - Fast-path matches skip the MLLM entirely
  - Section batching: 1 call per section (not 1 per entity) → ~72% reduction
  - Primary provider: NVIDIA NIM (free tier)
  - Fallback: per-entity retry using NvidiaLLMClient
"""
from __future__ import annotations

from config.config_parser import CMSVSConfig, SectionConfig
from models.nvidia_client import NvidiaLLMClient
from prompts.validation_prompt_builder import ValidationPromptBuilder
from shared_types import (
    DiscrepancyType,
    EntityValidationResult,
    FinalEntityValue,
    SectionValidationResult,
    ValidationReport,
    ValidationStatus,
)
from validation.utils.value_normalizer import ValueNormalizer


class SemanticValidator:
    """
    Performs section-wise CoT semantic validation of entity pairs.

    Parameters
    ----------
    llm_client         : NvidiaLLMClient for CoT validation calls
    config             : CMSVSConfig providing section and entity definitions
    prompt_builder     : ValidationPromptBuilder (created if not provided)
    normalizer         : ValueNormalizer (created if not provided)
    confidence_threshold : threshold below which entities are flagged for review
    """

    def __init__(
        self,
        llm_client: NvidiaLLMClient,
        config: CMSVSConfig,
        prompt_builder: ValidationPromptBuilder | None = None,
        normalizer: ValueNormalizer | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.prompt_builder = prompt_builder or ValidationPromptBuilder()
        self.normalizer = normalizer or ValueNormalizer()
        self.confidence_threshold = confidence_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        doc_a_entities: dict[str, FinalEntityValue],
        doc_b_entities: dict[str, FinalEntityValue],
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
    ) -> ValidationReport:
        """
        Validate all sections and return a complete ValidationReport.

        Parameters
        ----------
        doc_a_entities : entity extraction results for Document A
        doc_b_entities : entity extraction results for Document B
        doc_a_name     : human-readable label for Document A
        doc_b_name     : human-readable label for Document B

        Returns
        -------
        ValidationReport with per-section and per-entity results
        """
        report = ValidationReport(
            doc_a_path=doc_a_name,
            doc_b_path=doc_b_name,
            config_name=self.config.config_name,
        )

        for section in self.config.sections:
            section_result = self.validate_section(
                section=section,
                doc_a_entities=doc_a_entities,
                doc_b_entities=doc_b_entities,
                doc_a_name=doc_a_name,
                doc_b_name=doc_b_name,
            )
            report.section_results.append(section_result)

        return report

    def validate_section(
        self,
        section: SectionConfig,
        doc_a_entities: dict[str, FinalEntityValue],
        doc_b_entities: dict[str, FinalEntityValue],
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
    ) -> SectionValidationResult:
        """
        Validate all entity pairs in one section.

        Parameters
        ----------
        section        : SectionConfig with entity definitions
        doc_a_entities : all extracted entities for Document A
        doc_b_entities : all extracted entities for Document B
        doc_a_name     : label for Document A
        doc_b_name     : label for Document B

        Returns
        -------
        SectionValidationResult
        """
        entity_results: list[EntityValidationResult] = []
        deferred_pairs: list[dict] = []   # pairs not resolved by fast path

        # ── Step 1: Rule-based pre-normalisation fast path ────────────────────
        for entity_cfg in section.entities:
            name = entity_cfg.entity_name
            fev_a = doc_a_entities.get(name)
            fev_b = doc_b_entities.get(name)

            val_a = fev_a.extracted_value if fev_a else None
            val_b = fev_b.extracted_value if fev_b else None
            data_type = entity_cfg.data_type

            norm_a = self.normalizer.normalize(val_a, data_type)
            norm_b = self.normalizer.normalize(val_b, data_type)

            # Handle INELIGIBLE: one or both values missing
            if val_a is None or val_b is None:
                entity_results.append(EntityValidationResult(
                    entity_name=name,
                    section_name=section.section_name,
                    doc_a_value=val_a,
                    doc_b_value=val_b,
                    doc_a_normalized=norm_a,
                    doc_b_normalized=norm_b,
                    validation_status=ValidationStatus.INELIGIBLE,
                    discrepancy_type=DiscrepancyType.NOT_APPLICABLE,
                    reasoning="One or both values are missing (null).",
                    confidence=1.0,
                    review_required=True,
                    fast_path_match=False,
                ))
                continue

            # Fast-path MATCH: identical after normalisation
            if self.normalizer.is_fast_path_match(val_a, val_b, data_type):
                entity_results.append(EntityValidationResult(
                    entity_name=name,
                    section_name=section.section_name,
                    doc_a_value=val_a,
                    doc_b_value=val_b,
                    doc_a_normalized=norm_a,
                    doc_b_normalized=norm_b,
                    validation_status=ValidationStatus.MATCH,
                    discrepancy_type=DiscrepancyType.NOT_APPLICABLE,
                    reasoning=(
                        "Values are identical after rule-based normalisation "
                        "(fast-path match — no LLM call required)."
                    ),
                    confidence=1.0,
                    review_required=False,
                    fast_path_match=True,
                ))
                continue

            # Defer to MLLM CoT validation
            deferred_pairs.append({
                "entity_name":     name,
                "description":     entity_cfg.entity_description,
                "data_type":       data_type,
                "doc_a_value":     val_a,
                "doc_b_value":     val_b,
                "doc_a_normalized": norm_a,
                "doc_b_normalized": norm_b,
            })

        # ── Step 2: Section-level CoT MLLM validation for deferred pairs ──────
        if deferred_pairs:
            mllm_results = self._validate_with_llm(
                section=section,
                entity_pairs=deferred_pairs,
                doc_a_name=doc_a_name,
                doc_b_name=doc_b_name,
            )
            entity_results.extend(mllm_results)

        return SectionValidationResult(
            section_name=section.section_name,
            entity_results=entity_results,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _validate_with_llm(
        self,
        section: SectionConfig,
        entity_pairs: list[dict],
        doc_a_name: str,
        doc_b_name: str,
    ) -> list[EntityValidationResult]:
        """
        Run section-level CoT validation for all deferred entity pairs.

        Attempts the full section batch first. Falls back to per-entity
        validation if the batch response cannot be parsed.
        """
        prompt = self.prompt_builder.build_section_validation_prompt(
            section=section,
            entity_pairs=entity_pairs,
            doc_a_name=doc_a_name,
            doc_b_name=doc_b_name,
        )

        try:
            response = self.llm_client.complete_json(prompt=prompt)
            return self._parse_validation_response(
                response=response,
                section=section,
                entity_pairs=entity_pairs,
            )
        except Exception:
            # Fallback: validate each entity individually
            return self._validate_pairs_individually(
                section=section,
                entity_pairs=entity_pairs,
                doc_a_name=doc_a_name,
                doc_b_name=doc_b_name,
            )

    def _parse_validation_response(
        self,
        response: dict,
        section: SectionConfig,
        entity_pairs: list[dict],
    ) -> list[EntityValidationResult]:
        """Parse the MLLM JSON validation response into EntityValidationResult list."""
        validations = response.get("validations", [])
        val_map: dict[str, dict] = {
            v.get("entity_name", ""): v
            for v in validations
            if isinstance(v, dict)
        }

        results: list[EntityValidationResult] = []
        for pair in entity_pairs:
            name = pair["entity_name"]
            v = val_map.get(name, {})

            status = self._parse_status(v.get("validation_status", "INELIGIBLE"))
            disc = self._parse_discrepancy(v.get("discrepancy_type", "NOT_APPLICABLE"))
            confidence = float(v.get("confidence", 0.5))

            results.append(EntityValidationResult(
                entity_name=name,
                section_name=section.section_name,
                doc_a_value=pair.get("doc_a_value"),
                doc_b_value=pair.get("doc_b_value"),
                doc_a_normalized=v.get("doc_a_normalized", pair.get("doc_a_normalized")),
                doc_b_normalized=v.get("doc_b_normalized", pair.get("doc_b_normalized")),
                validation_status=status,
                discrepancy_type=disc,
                reasoning=v.get("reasoning", "No reasoning provided."),
                confidence=confidence,
                review_required=confidence < self.confidence_threshold,
                fast_path_match=False,
            ))

        return results

    def _validate_pairs_individually(
        self,
        section: SectionConfig,
        entity_pairs: list[dict],
        doc_a_name: str,
        doc_b_name: str,
    ) -> list[EntityValidationResult]:
        """Fallback: validate each entity pair with a single-entity prompt."""
        results: list[EntityValidationResult] = []
        for pair in entity_pairs:
            name = pair["entity_name"]
            prompt = self.prompt_builder.build_single_entity_prompt(
                entity_name=name,
                entity_description=pair.get("description", ""),
                data_type=pair.get("data_type", "text"),
                doc_a_value=pair.get("doc_a_value"),
                doc_b_value=pair.get("doc_b_value"),
                doc_a_name=doc_a_name,
                doc_b_name=doc_b_name,
            )
            try:
                response = self.llm_client.complete_json(prompt=prompt)
                status = self._parse_status(
                    response.get("validation_status", "INELIGIBLE")
                )
                disc = self._parse_discrepancy(
                    response.get("discrepancy_type", "NOT_APPLICABLE")
                )
                confidence = float(response.get("confidence", 0.5))
                reasoning = response.get("reasoning", "No reasoning provided.")
                norm_a = response.get("doc_a_normalized", pair.get("doc_a_normalized"))
                norm_b = response.get("doc_b_normalized", pair.get("doc_b_normalized"))
            except Exception as exc:
                status = ValidationStatus.INELIGIBLE
                disc = DiscrepancyType.NOT_APPLICABLE
                confidence = 0.0
                reasoning = f"Validation failed: {str(exc)[:200]}"
                norm_a = pair.get("doc_a_normalized")
                norm_b = pair.get("doc_b_normalized")

            results.append(EntityValidationResult(
                entity_name=name,
                section_name=section.section_name,
                doc_a_value=pair.get("doc_a_value"),
                doc_b_value=pair.get("doc_b_value"),
                doc_a_normalized=norm_a,
                doc_b_normalized=norm_b,
                validation_status=status,
                discrepancy_type=disc,
                reasoning=reasoning,
                confidence=confidence,
                review_required=confidence < self.confidence_threshold,
                fast_path_match=False,
            ))

        return results

    @staticmethod
    def _parse_status(raw: str) -> ValidationStatus:
        """Parse validation status string safely."""
        try:
            return ValidationStatus(raw.upper())
        except ValueError:
            return ValidationStatus.INELIGIBLE

    @staticmethod
    def _parse_discrepancy(raw: str) -> DiscrepancyType:
        """Parse discrepancy type string safely."""
        try:
            return DiscrepancyType(raw.upper())
        except ValueError:
            return DiscrepancyType.NOT_APPLICABLE
