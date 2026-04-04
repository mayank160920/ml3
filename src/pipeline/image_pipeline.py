"""
--------------------------------
Direct MLLM pipeline for image document inputs.

Bypasses OCR and dense retrieval entirely. The image is passed directly
to the MLLM, which extracts all section entities in one call per section.

This path is optimal for:
  - Single-page scanned form images (FUNSD dataset)
  - Documents where layout is simple enough for direct visual extraction
  - Cases where OCR + retrieval overhead is not justified

Returns dict[entity_name → FinalEntityValue] — identical structure to
PDFPipeline output, ensuring the validation layer is agnostic to input type.
"""
from __future__ import annotations

from config.config_parser import CMSVSConfig
from extraction.expression_orchestrator import ExpressionOrchestrator
from extraction.mllm_extractor import MLLMExtractor
from input.image_loader import ImageLoader
from models.nvidia_client import NvidiaLLMClient
from shared_types import (
    EntityType,
    FinalEntityValue,
    LoadedInput,
)


class ImagePipeline:
    """
    Executes the direct MLLM pipeline for image documents.

    Parameters
    ----------
    config               : CMSVSConfig with section and entity definitions
    llm_client           : NvidiaLLMClient for visual MLLM extraction
    image_loader         : ImageLoader for loading image files
    confidence_threshold : minimum confidence before flagging for review
    """

    def __init__(
        self,
        config: CMSVSConfig,
        llm_client: NvidiaLLMClient,
        image_loader: ImageLoader | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.image_loader = image_loader or ImageLoader()
        self.confidence_threshold = confidence_threshold

        self._extractor = MLLMExtractor(
            llm_client=llm_client,
            confidence_threshold=confidence_threshold,
        )
        self._expression_orchestrator = ExpressionOrchestrator(
            mllm_extractor=self._extractor,
            confidence_threshold=confidence_threshold,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, image_path: str) -> dict[str, FinalEntityValue]:
        """
        Execute the direct MLLM pipeline for an image document.

        Parameters
        ----------
        image_path : path to the image file

        Returns
        -------
        dict mapping entity_name → FinalEntityValue
        """
        # ── L1: Image Loading ─────────────────────────────────────────────────
        loaded = self.image_loader.load_image(image_path)

        # ── L5 + L6: Direct MLLM Extraction per section ───────────────────────
        return self._extract_all_sections(loaded)

    # ── private ───────────────────────────────────────────────────────────────

    def _extract_all_sections(
        self,
        loaded: LoadedInput,
    ) -> dict[str, FinalEntityValue]:
        """
        Iterate over all config sections and extract entities from the image.

        All sections use the same single image (page_number=1).
        """
        # Single image for all sections
        images_b64 = [
            pi.image_base64 for pi in loaded.page_images.values()
        ]
        all_values: dict[str, FinalEntityValue] = {}

        for section in self.config.sections:
            # ── L5: MLLM Extraction (DIRECT entities) ─────────────────────────
            raw_extractions = self._extractor.extract_section(
                section=section,
                page_images_b64=images_b64,
            )
            entity_results = self._extractor.to_entity_results(raw_extractions)

            # ── L6: Expression Engine ─────────────────────────────────────────
            expression_values = self._expression_orchestrator.process_all_expression_entities(
                section=section,
                page_images_b64=images_b64,
            )

            # ── Merge DIRECT results → FinalEntityValue ───────────────────────
            for name, er in entity_results.items():
                if name not in expression_values:
                    # print(name)
                    # print("-"*50)
                    # print(er)
                    # print("-"*50)
                    # print(er.raw_context)
                    # print("*"*50)
                    all_values[name] = FinalEntityValue(
                        entity_name=er.entity_name,
                        extracted_value=er.extracted_value,
                        extraction_status=er.extraction_status,
                        entity_type=EntityType.DIRECT,
                        confidence=er.confidence,
                        source_page=er.source_page,
                        source_region=er.raw_context[:80] if er.raw_context is not None else None,
                        raw_context=er.raw_context,
                        review_required=er.review_required,
                        fallback_triggered=er.fallback_triggered,
                        expression_audit=None,
                    )

            # print("*"*50+"\n"+"="*50+"\n"+"*"*50)

            # ── Merge EXPRESSION results ──────────────────────────────────────
            all_values.update(expression_values)

        # print(all_values)
        return all_values
        

