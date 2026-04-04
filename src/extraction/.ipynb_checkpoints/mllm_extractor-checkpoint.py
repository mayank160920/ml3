"""
---------------------------------
MLLM visual extraction layer (Layer 5).

Extracts entity values from document page images using the NVIDIA NIM
multimodal LLM. The MLLM reads raw page images as a human would — using
spatial relationships, table structures, font hierarchies, and column
positions — overcoming the column-interleaving and spatial-context-loss
failures of text-only OCR pipelines.

Key design decisions
---------------------
- Section-batched: all entities in a section are extracted in ONE call,
  reducing API calls from N_entities to N_sections (~72% reduction).
- Visual-first: page images are sent directly; no OCR text is passed to
  the MLLM for extraction (only for indexing in Layer 3).
- Schema-enforced: strict JSON schema is embedded in the prompt.
- Retry-with-re-instruction: JSON parse failures trigger re-prompting
  with an explicit schema reminder (up to max_retries attempts).
- Fallback null: persistent failures produce a null EntityResult with
  review_required=True rather than crashing the pipeline.
"""
from __future__ import annotations

from shared_types import EntityResult, ExtractionStatus, RawExtraction
from config.config_parser import SectionConfig
from models.nvidia_client import NvidiaLLMClient
from prompts.ner_prompt_builder import NERPromptBuilder


class MLLMExtractor:
    """
    Extracts entities from page images using a multimodal LLM.

    Parameters
    ----------
    llm_client         : NvidiaLLMClient for visual inference
    prompt_builder     : NERPromptBuilder for prompt construction
    confidence_threshold: minimum confidence before flagging for review
    """

    def __init__(
        self,
        llm_client: NvidiaLLMClient,
        prompt_builder: NERPromptBuilder | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or NERPromptBuilder()
        self.confidence_threshold = confidence_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def extract_section(
        self,
        section: SectionConfig,
        page_images_b64: list[str],
        fallback: bool = False,
    ) -> dict[str, RawExtraction]:
        """
        Extract all entities in a section from the provided page images.

        Parameters
        ----------
        section          : SectionConfig with all entity definitions
        page_images_b64  : list of base64-encoded PNG page images
        fallback         : if True, uses the fallback prompt variant

        Returns
        -------
        dict mapping entity_name → RawExtraction

        Notes
        -----
        Returns empty-value RawExtractions (NOT_FOUND) for all entities if
        the LLM call or JSON parsing fails after all retries.
        """
        if fallback:
            prompt = self.prompt_builder.build_fallback_prompt(section)
        else:
            prompt = self.prompt_builder.build_section_prompt(section)

        try:
            response = self.llm_client.complete_json(
                prompt=prompt,
                images_b64=page_images_b64,
            )
            return self._parse_extractions(response, section)
        except Exception as exc:
            # Return null extractions for all entities — pipeline continues
            return self._null_extractions(section, error=str(exc))

    def to_entity_results(
        self,
        raw_extractions: dict[str, RawExtraction],
    ) -> dict[str, EntityResult]:
        """
        Convert RawExtraction objects to EntityResult objects, applying
        confidence thresholding and review flagging.

        Parameters
        ----------
        raw_extractions : mapping of entity_name → RawExtraction

        Returns
        -------
        dict mapping entity_name → EntityResult
        """
        results: dict[str, EntityResult] = {}
        for name, raw in raw_extractions.items():
            review = (
                raw.confidence < self.confidence_threshold
                or raw.extraction_status == ExtractionStatus.NOT_FOUND
                or raw.extraction_status == ExtractionStatus.ERROR
            )
            results[name] = EntityResult(
                entity_name=raw.entity_name,
                extracted_value=raw.extracted_value,
                extraction_status=raw.extraction_status,
                source_page=raw.source_page,
                confidence=raw.confidence,
                review_required=review,
                raw_context=raw.raw_context,
                fallback_triggered=False,
            )
        return results

    # ── private ───────────────────────────────────────────────────────────────

    def _parse_extractions(
        self,
        response: dict,
        section: SectionConfig,
    ) -> dict[str, RawExtraction]:
        """
        Parse the MLLM JSON response into RawExtraction objects.

        Handles missing fields gracefully — any extraction that cannot
        be fully parsed defaults to NOT_FOUND with confidence 0.0.
        """
        results: dict[str, RawExtraction] = {}
        extractions = response.get("extractions", [])

        # Index by entity_name for O(1) lookup
        extraction_map: dict[str, dict] = {
            e.get("entity_name", ""): e
            for e in extractions
            if isinstance(e, dict)
        }

        for entity in section.entities:
            name = entity.entity_name
            raw = extraction_map.get(name, {})

            status_raw = raw.get("extraction_status", "NOT_FOUND").upper()
            try:
                status = ExtractionStatus(status_raw)
            except ValueError:
                status = ExtractionStatus.NOT_FOUND

            results[name] = RawExtraction(
                entity_name=name,
                extracted_value=raw.get("extracted_value"),
                extraction_status=status,
                source_page=raw.get("source_page"),
                source_region=raw.get("source_region", ""),
                confidence=float(raw.get("confidence", 0.0)),
                raw_context=raw.get("raw_context", ""),
            )

        return results

    def _null_extractions(
        self,
        section: SectionConfig,
        error: str = "",
    ) -> dict[str, RawExtraction]:
        """
        Build a dict of NOT_FOUND RawExtractions for all section entities.

        Used when the LLM call fails entirely, so the pipeline can continue
        to process remaining sections.
        """
        return {
            entity.entity_name: RawExtraction(
                entity_name=entity.entity_name,
                extracted_value=None,
                extraction_status=ExtractionStatus.ERROR,
                source_page=None,
                source_region="",
                confidence=0.0,
                raw_context=f"Extraction failed: {error[:200]}",
            )
            for entity in section.entities
        }
