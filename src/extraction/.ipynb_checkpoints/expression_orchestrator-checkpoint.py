"""
-------------------------------------------
Orchestrates the full EXPRESSION entity pipeline:
  MLLM variable extraction → SimpleEval computation → FinalEntityValue

Wires together MLLMExtractor (for variable extraction) and
ExpressionEvaluator (for safe arithmetic computation), producing
FinalEntityValue objects with complete audit trails.

EXPRESSION entity flow
-----------------------
1. Config declares: expression_template + expression_variables
2. MLLM extracts each variable as a pseudo-entity from page images
3. ExpressionEvaluator computes the final value from extracted variables
4. FinalEntityValue is produced with expression_audit containing:
     - template used
     - raw variable values and their parsed numeric equivalents
     - per-variable extraction confidences
     - computed result
"""
from __future__ import annotations

from config.config_parser import EntityConfig, SectionConfig
from extraction.expression_evaluator import ExpressionEvaluator
from extraction.mllm_extractor import MLLMExtractor
from shared_types import (
    EntityResult,
    EntityType,
    ExtractionStatus,
    FinalEntityValue,
    RawExtraction,
)


class ExpressionOrchestrator:
    """
    Orchestrates EXPRESSION entity extraction and computation.

    Parameters
    ----------
    mllm_extractor    : MLLMExtractor for visual variable extraction
    evaluator         : ExpressionEvaluator for sandboxed arithmetic
    confidence_threshold : minimum confidence before flagging for review
    """

    def __init__(
        self,
        mllm_extractor: MLLMExtractor,
        evaluator: ExpressionEvaluator | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.mllm_extractor = mllm_extractor
        self.evaluator = evaluator or ExpressionEvaluator()
        self.confidence_threshold = confidence_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def process_expression_entity(
        self,
        entity: EntityConfig,
        page_images_b64: list[str],
        section: SectionConfig,
    ) -> FinalEntityValue:
        """
        Execute the full EXPRESSION entity pipeline for one entity.

        Steps
        -----
        1. Build a synthetic SectionConfig containing only the expression
           variable pseudo-entities
        2. Run MLLM extraction to get raw variable values
        3. Evaluate the expression template with the extracted variables
        4. Return a FinalEntityValue with full expression_audit

        Parameters
        ----------
        entity          : EntityConfig with entity_extraction_logic == EXPRESSION
        page_images_b64 : list of base64-encoded page images
        section         : parent SectionConfig (used for section_name in results)

        Returns
        -------
        FinalEntityValue for the expression entity
        """
        # ── Step 1: Extract expression variables via MLLM ─────────────────────
        var_section = self._build_variable_section(entity, section)
        raw_extractions = self.mllm_extractor.extract_section(
            section=var_section,
            page_images_b64=page_images_b64,
        )

        # ── Step 2: Collect variable values and confidences ───────────────────
        variable_values: dict[str, str | None] = {}
        variable_confidences: dict[str, float] = {}

        for var in entity.expression_variables:
            extraction = raw_extractions.get(var.name)
            if extraction is not None:
                variable_values[var.name] = extraction.extracted_value
                variable_confidences[var.name] = extraction.confidence
            else:
                variable_values[var.name] = None
                variable_confidences[var.name] = 0.0

        # ── Step 3: Evaluate the expression ──────────────────────────────────
        expr_result = self.evaluator.evaluate_with_confidences(
            template=entity.expression_template or "",
            variable_values=variable_values,
            variable_confidences=variable_confidences,
            data_type=entity.data_type,
        )

        # ── Step 4: Build FinalEntityValue ────────────────────────────────────
        return self._build_final_value(entity, expr_result)

    def process_all_expression_entities(
        self,
        section: SectionConfig,
        page_images_b64: list[str],
    ) -> dict[str, FinalEntityValue]:
        """
        Process all EXPRESSION entities in a section.

        Parameters
        ----------
        section         : SectionConfig
        page_images_b64 : list of base64-encoded page images

        Returns
        -------
        dict mapping entity_name → FinalEntityValue for each EXPRESSION entity
        """
        results: dict[str, FinalEntityValue] = {}
        for entity in section.entities:
            if entity.entity_extraction_logic == "EXPRESSION":
                results[entity.entity_name] = self.process_expression_entity(
                    entity=entity,
                    page_images_b64=page_images_b64,
                    section=section,
                )
        return results

    # ── private ───────────────────────────────────────────────────────────────

    def _build_variable_section(
        self,
        entity: EntityConfig,
        parent_section: SectionConfig,
    ) -> SectionConfig:
        """
        Build a synthetic SectionConfig where each expression variable
        is treated as a DIRECT entity for MLLM extraction.
        """
        from config.config_parser import EntityConfig as EC, SectionConfig as SC

        variable_entities = [
            EC(
                entity_name=var.name,
                entity_description=var.description,
                entity_extraction_logic="DIRECT",
                entity_example_value=var.example_value,
                data_type=entity.data_type,
            )
            for var in entity.expression_variables
        ]

        return SC(
            section_name=f"{parent_section.section_name} — {entity.entity_name} variables",
            section_description=(
                f"Variables for computing: {entity.entity_description}. "
                f"Template: {entity.expression_template}"
            ),
            section_keywords=parent_section.section_keywords,
            entities=variable_entities,
        )

    def _build_final_value(
        self,
        entity: EntityConfig,
        expr_result: dict,
    ) -> FinalEntityValue:
        """Convert an expression evaluation result into a FinalEntityValue."""
        success = expr_result.get("status") == "SUCCESS"
        confidence = float(expr_result.get("confidence", 0.0))

        return FinalEntityValue(
            entity_name=entity.entity_name,
            extracted_value=expr_result.get("computed_value"),
            extraction_status=(
                ExtractionStatus.FOUND if success
                else ExtractionStatus.ERROR
            ),
            entity_type=EntityType.EXPRESSION,
            confidence=confidence,
            source_page=None,
            source_region="Computed from expression variables",
            raw_context=str(expr_result.get("variable_values", {}))[:200],
            review_required=confidence < self.confidence_threshold or not success,
            fallback_triggered=False,
            expression_audit=expr_result,
        )
