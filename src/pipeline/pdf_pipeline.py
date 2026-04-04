"""
------------------------------
Full RAG pipeline for PDF document inputs.

Pipeline stages
---------------
L1  Document Loading    — PyMuPDF renders each page at 150 DPI
L2  OCR Processing      — PaddleOCR Mobile produces composite index text
L3  Dense Vector Index  — NemoRetriever embeds pages into ChromaDB
L4  RAG Page Routing    — Cosine similarity retrieves top-K pages per section
L5  MLLM Extraction     — Visual entity extraction from page images
L6  Expression Engine   — SimpleEval computes EXPRESSION entity values
    Fallback Mechanism  — Expanded top-K retrieval for low-confidence entities

Returns dict[entity_name → FinalEntityValue] identical to the image pipeline,
ensuring the validation layer is agnostic to input type.
"""
from __future__ import annotations

import fitz

from config.config_parser import CMSVSConfig
from extraction.expression_orchestrator import ExpressionOrchestrator
from extraction.mllm_extractor import MLLMExtractor
from ingestion.document_processor import DocumentProcessor
from models.nvidia_client import NvidiaEmbeddingClient, NvidiaLLMClient
from ocr.ocr_engine import OCREngine
from retrieval.dense_retriever import DenseRetriever
from retrieval.index_builder import IndexBuilder
from shared_types import (
    EntityType,
    ExtractionStatus,
    FinalEntityValue,
    LoadedInput,
    StructuredPage,
)


class PDFPipeline:
    """
    Executes the full RAG pipeline for a PDF document.

    Parameters
    ----------
    config               : CMSVSConfig with section and entity definitions
    embedding_client     : NvidiaEmbeddingClient for NemoRetriever embeddings
    llm_client           : NvidiaLLMClient for visual MLLM extraction
    ocr_engine           : OCREngine for page text extraction (indexing only)
    document_processor   : DocumentProcessor for PDF loading
    confidence_threshold : minimum confidence before fallback / review flag
    default_top_k        : pages retrieved per section (default: 2)
    fallback_top_k       : expanded pages used in fallback pass (default: 4)
    """

    def __init__(
        self,
        config: CMSVSConfig,
        embedding_client: NvidiaEmbeddingClient,
        llm_client: NvidiaLLMClient,
        ocr_engine: OCREngine | None = None,
        document_processor: DocumentProcessor | None = None,
        confidence_threshold: float = 0.75,
        default_top_k: int = 2,
        fallback_top_k: int = 4,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.ocr_engine = ocr_engine or OCREngine()
        self.document_processor = document_processor or DocumentProcessor()
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k
        self.fallback_top_k = fallback_top_k

        self._extractor = MLLMExtractor(
            llm_client=llm_client,
            confidence_threshold=confidence_threshold,
        )
        self._expression_orchestrator = ExpressionOrchestrator(
            mllm_extractor=self._extractor,
            confidence_threshold=confidence_threshold,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, pdf_path: str) -> dict[str, FinalEntityValue]:
        """
        Execute all pipeline stages for a PDF document.

        Parameters
        ----------
        pdf_path : path to the PDF file

        Returns
        -------
        dict mapping entity_name → FinalEntityValue
        """
        # ── L1: Document Loading ──────────────────────────────────────────────
        loaded = self.document_processor.process(pdf_path)

        # ── L2: OCR Processing ────────────────────────────────────────────────
        structured_pages = self._run_ocr(pdf_path, loaded)

        # ── L3: Dense Vector Index Construction ──────────────────────────────
        index_builder = IndexBuilder(embedding_client=self.embedding_client)
        collection = index_builder.build(structured_pages)

        # ── L4 + L5 + L6: Section-wise RAG Routing, Extraction, Expression ────
        retriever = DenseRetriever(
            embedding_client=self.embedding_client,
            collection=collection,
            default_top_k=self.default_top_k,
            fallback_top_k=self.fallback_top_k,
        )

        all_entity_values = self._extract_all_sections(loaded, retriever)

        # ── Cleanup: destroy ChromaDB collection ──────────────────────────────
        index_builder.destroy()

        return all_entity_values

    # ── private ───────────────────────────────────────────────────────────────

    def _run_ocr(
        self,
        pdf_path: str,
        loaded: LoadedInput,
    ) -> dict[int, StructuredPage]:
        """
        Run OCR on each page, falling back to PyMuPDF text extraction
        if PaddleOCR is unavailable.
        """
        structured_pages: dict[int, StructuredPage] = {}

        try:
            ocr_pages = self.ocr_engine.process_pdf(pdf_path)
            for sp in ocr_pages:
                structured_pages[sp.page_number] = sp
        except Exception:
            # PyMuPDF fallback: extract text directly from PDF
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc, start=1):
                    raw_text = page.get_text("text")
                    sp = self.ocr_engine.process_text(
                        raw_text, page_number=page_idx
                    )
                    structured_pages[page_idx] = sp

        return structured_pages

    def _extract_all_sections(
        self,
        loaded: LoadedInput,
        retriever: DenseRetriever,
    ) -> dict[str, FinalEntityValue]:
        """
        Iterate over all config sections, retrieve relevant pages, and
        extract entities via MLLM and Expression engine.
        """
        all_values: dict[str, FinalEntityValue] = {}

        for section in self.config.sections:
            # ── L4: RAG Page Routing ──────────────────────────────────────────
            page_numbers = retriever.retrieve_for_section(section)
            images_b64 = self._get_images(loaded, page_numbers)

            # ── L5: MLLM Extraction (DIRECT entities) ─────────────────────────
            raw_extractions = self._extractor.extract_section(
                section=section,
                page_images_b64=images_b64,
            )
            entity_results = self._extractor.to_entity_results(raw_extractions)

            # ── Fallback: low-confidence entities ─────────────────────────────
            low_conf = [
                name for name, er in entity_results.items()
                if er.review_required
                and section_has_direct_entity(section, name)
            ]
            if low_conf:
                fallback_pages = retriever.retrieve_fallback(section)
                fallback_images = self._get_images(loaded, fallback_pages)
                fallback_raw = self._extractor.extract_section(
                    section=section,
                    page_images_b64=fallback_images,
                    fallback=True,
                )
                for name, fb_raw in fallback_raw.items():
                    if name in low_conf:
                        old_conf = entity_results[name].confidence
                        if fb_raw.confidence > old_conf:
                            entity_results[name].extracted_value = fb_raw.extracted_value
                            entity_results[name].confidence = fb_raw.confidence
                            entity_results[name].extraction_status = fb_raw.extraction_status
                            entity_results[name].fallback_triggered = True
                            if fb_raw.confidence >= self.confidence_threshold:
                                entity_results[name].review_required = False

            # ── L6: Expression Engine ─────────────────────────────────────────
            expression_values = self._expression_orchestrator.process_all_expression_entities(
                section=section,
                page_images_b64=images_b64,
            )

            # ── Merge: DIRECT entity results → FinalEntityValue ───────────────
            for name, er in entity_results.items():
                if name not in expression_values:
                    all_values[name] = FinalEntityValue(
                        entity_name=er.entity_name,
                        extracted_value=er.extracted_value,
                        extraction_status=er.extraction_status,
                        entity_type=EntityType.DIRECT,
                        confidence=er.confidence,
                        source_page=er.source_page,
                        source_region=er.raw_context[:80],
                        raw_context=er.raw_context,
                        review_required=er.review_required,
                        fallback_triggered=er.fallback_triggered,
                        expression_audit=None,
                    )

            # ── Merge: EXPRESSION entity results ─────────────────────────────
            all_values.update(expression_values)

        return all_values

    @staticmethod
    def _get_images(
        loaded: LoadedInput,
        page_numbers: list[int],
    ) -> list[str]:
        """Retrieve base64 images for the given page numbers."""
        images = []
        for pn in page_numbers:
            page_image = loaded.page_images.get(pn)
            if page_image:
                images.append(page_image.image_base64)
        return images


def section_has_direct_entity(section, entity_name: str) -> bool:
    """Return True if the entity is a DIRECT (non-expression) entity."""
    for e in section.entities:
        if e.entity_name == entity_name:
            return e.entity_extraction_logic == "DIRECT"
    return False
