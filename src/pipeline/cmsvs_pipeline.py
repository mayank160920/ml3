"""
--------------------------------
Master pipeline orchestrator for the CMSVS system.

Routes each document to the correct sub-pipeline (PDF or Image) based on
file extension, then runs the Section-Wise Semantic Validation layer and
generates the final structured report.

This is the single entry point for all external callers (CLI, API, notebooks).

Usage
-----
>>> pipeline = CMSVSPipeline.from_env(config_path="configs/funsd_ner_config.yaml")
>>> report = pipeline.run(doc_a_path="doc_a.png", doc_b_path="doc_b.png")
>>> pipeline.save_report(report, output_path="output/result.json")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config.config_parser import CMSVSConfig, CMSVSConfigParser
from ingestion.document_processor import DocumentProcessor
from input.image_loader import ImageLoader
from models.nvidia_client import NvidiaEmbeddingClient, NvidiaLLMClient
from ocr.ocr_engine import OCREngine
from output.report_generator import ReportGenerator
from pipeline.image_pipeline import ImagePipeline
from pipeline.pdf_pipeline import PDFPipeline
from shared_types import FinalEntityValue, InputType, ValidationReport
from validation.semantic_validator import SemanticValidator


# ── Extension sets ─────────────────────────────────────────────────────────────

_PDF_EXTENSIONS = frozenset({".pdf"})
_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"
})


class CMSVSPipeline:
    """
    Master orchestrator that runs the full CMSVS pipeline end-to-end.

    Accepts both PDF and image inputs for Doc A and Doc B independently —
    e.g., Doc A can be a PDF (RAG path) while Doc B is an image (direct path).

    Parameters
    ----------
    config               : CMSVSConfig parsed from YAML
    embedding_client     : NvidiaEmbeddingClient for NemoRetriever
    llm_client           : NvidiaLLMClient for extraction + validation
    confidence_threshold : minimum confidence before review flagging
    default_top_k        : RAG pages retrieved per section
    fallback_top_k       : expanded pages used in fallback pass
    """

    def __init__(
        self,
        config: CMSVSConfig,
        embedding_client: NvidiaEmbeddingClient,
        llm_client: NvidiaLLMClient,
        confidence_threshold: float = 0.75,
        default_top_k: int = 2,
        fallback_top_k: int = 4,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k
        self.fallback_top_k = fallback_top_k

        self._report_generator = ReportGenerator()
        self._validator = SemanticValidator(
            llm_client=llm_client,
            config=config,
            confidence_threshold=confidence_threshold,
        )

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        config_path: str | Path,
        confidence_threshold: float = 0.75,
        default_top_k: int = 2,
        fallback_top_k: int = 4,
    ) -> "CMSVSPipeline":
        """
        Construct a CMSVSPipeline from environment variables.

        Required environment variables
        --------------------------------
        NVIDIA_API_KEY : NVIDIA NIM API key (used for both embeddings and LLM)

        Parameters
        ----------
        config_path          : path to the YAML configuration file
        confidence_threshold : minimum confidence for review flagging
        default_top_k        : default RAG retrieval depth
        fallback_top_k       : fallback RAG retrieval depth

        Returns
        -------
        CMSVSPipeline
        """
        nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
        if not nvidia_key:
            raise EnvironmentError(
                "NVIDIA_API_KEY environment variable is not set."
            )

        config = CMSVSConfigParser().load(config_path)

        embedding_client = NvidiaEmbeddingClient(api_key=nvidia_key)
        llm_client = NvidiaLLMClient(api_key=nvidia_key)

        return cls(
            config=config,
            embedding_client=embedding_client,
            llm_client=llm_client,
            confidence_threshold=confidence_threshold,
            default_top_k=default_top_k,
            fallback_top_k=fallback_top_k,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        doc_a_path: str,
        doc_b_path: str,
        doc_a_name: str | None = None,
        doc_b_name: str | None = None,
    ) -> dict:
        """
        Run the full pipeline for a document pair and return the report dict.

        Parameters
        ----------
        doc_a_path : path to Document A (PDF or image)
        doc_b_path : path to Document B (PDF or image)
        doc_a_name : human-readable label for Doc A (defaults to filename)
        doc_b_name : human-readable label for Doc B (defaults to filename)

        Returns
        -------
        dict : complete validation report (serialisable to JSON)
        """
        a_label = doc_a_name or Path(doc_a_path).name
        b_label = doc_b_name or Path(doc_b_path).name

        # ── Extract entities from both documents ──────────────────────────────
        doc_a_entities = self._extract_document(doc_a_path)
        doc_b_entities = self._extract_document(doc_b_path)

        # ── Layer 7: Section-wise semantic validation ─────────────────────────
        validation_report = self._validator.validate(
            doc_a_entities=doc_a_entities,
            doc_b_entities=doc_b_entities,
            doc_a_name=a_label,
            doc_b_name=b_label,
        )

        # ── Generate structured output report ─────────────────────────────────
        return self._report_generator.generate(
            validation_report=validation_report,
            doc_a_entities=doc_a_entities,
            doc_b_entities=doc_b_entities,
            metadata={
                "doc_a_path": doc_a_path,
                "doc_b_path": doc_b_path,
                "config_name": self.config.config_name,
                "confidence_threshold": self.confidence_threshold,
            },
        )

    def run_groundtruth_format(
        self,
        doc_a_path: str,
        doc_b_path: str,
        doc_a_name: str | None = None,
        doc_b_name: str | None = None,
    ) -> dict:
        """
        Run the pipeline and return output in M2 ground truth JSON format.

        Used for direct M5 evaluation against dataset ground truth files.

        Returns
        -------
        dict matching M2 ground truth schema:
          { "entities": [ { entity_name, doc_a_value, doc_b_value,
                            normalized_value, validation_type,
                            validation_result } ] }
        """
        a_label = doc_a_name or Path(doc_a_path).name
        b_label = doc_b_name or Path(doc_b_path).name

        doc_a_entities = self._extract_document(doc_a_path)
        doc_b_entities = self._extract_document(doc_b_path)

        validation_report = self._validator.validate(
            doc_a_entities=doc_a_entities,
            doc_b_entities=doc_b_entities,
            doc_a_name=a_label,
            doc_b_name=b_label,
        )

        return self._report_generator.generate_groundtruth_format(
            validation_report=validation_report
        )

    def save_report(
        self,
        report: dict,
        output_path: str | Path,
    ) -> Path:
        """
        Save a report dict to a JSON file.

        Parameters
        ----------
        report      : dict from run() or run_groundtruth_format()
        output_path : destination file path

        Returns
        -------
        Path : absolute path to the saved file
        """
        return self._report_generator.save(report, output_path)

    # ── private ───────────────────────────────────────────────────────────────

    def _extract_document(
        self,
        doc_path: str,
    ) -> dict[str, FinalEntityValue]:
        """
        Route a document to the correct sub-pipeline and extract entities.

        PDF   → PDFPipeline  (full RAG: OCR → index → retrieve → extract)
        Image → ImagePipeline (direct MLLM: load → extract)
        """
        suffix = Path(doc_path).suffix.lower()

        if suffix in _PDF_EXTENSIONS:
            pipeline = PDFPipeline(
                config=self.config,
                embedding_client=self.embedding_client,
                llm_client=self.llm_client,
                ocr_engine=OCREngine(),
                document_processor=DocumentProcessor(),
                confidence_threshold=self.confidence_threshold,
                default_top_k=self.default_top_k,
                fallback_top_k=self.fallback_top_k,
            )
            return pipeline.run(doc_path)

        if suffix in _IMAGE_EXTENSIONS:
            pipeline = ImagePipeline(
                config=self.config,
                llm_client=self.llm_client,
                image_loader=ImageLoader(),
                confidence_threshold=self.confidence_threshold,
            )
            return pipeline.run(doc_path)

        supported = sorted(_PDF_EXTENSIONS | _IMAGE_EXTENSIONS)
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: {', '.join(supported)}"
        )
