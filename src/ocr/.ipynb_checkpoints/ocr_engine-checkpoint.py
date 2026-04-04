"""
----------------------
OCR processing layer using PaddleOCR Mobile.

Role in the pipeline
--------------------
The OCR engine's ONLY job is to produce text for the dense vector index
(Layer 3). It is NEVER used as the source of extracted entity values —
that is always done visually by the MLLM (Layer 5).

This means OCR imperfections only affect retrieval quality, not
extraction accuracy, since the MLLM reads raw page images directly.

Composite index text format
----------------------------
    SECTIONS: <header1> | <header2>
    KEY_VALUES: <key1>: <val1> | <key2>: <val2>
    RAW_TEXT:
    <full page text>

The structured format prioritises high-signal content (headers and
key-value pairs) at the beginning of the index text for better
embedding quality.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from shared_types import StructuredPage


class OCREngine:
    """
    Wraps PaddleOCR Mobile to produce StructuredPage objects per page.

    Supported engines: 'paddle' (default), 'surya' (not yet implemented).

    Parameters
    ----------
    engine                       : OCR backend ('paddle' only)
    device                       : inference device ('cpu' | 'gpu')
    use_doc_orientation_classify : auto-rotate pages (disabled by default)
    use_doc_unwarping            : dewarp curved pages (disabled by default)
    """

    SUPPORTED_ENGINES: frozenset[str] = frozenset({"surya", "paddle"})

    # Matches "Label: Value" patterns for key-value extraction
    _KEY_VALUE_PATTERN = re.compile(
        r"^\s*(?:[-*]\s+)?(?:#+\s+)?([^:\n]{1,80}):\s*(.+?)\s*$",
        re.MULTILINE,
    )

    def __init__(
        self,
        engine: str = "paddle",
        *,
        device: str = "cpu",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
    ) -> None:
        if engine not in self.SUPPORTED_ENGINES:
            supported = ", ".join(sorted(self.SUPPORTED_ENGINES))
            raise ValueError(
                f"Unsupported OCR engine '{engine}'. "
                f"Supported engines: {supported}"
            )
        if engine == "surya":
            raise NotImplementedError(
                "The 'surya' OCR backend is not yet implemented."
            )

        self.engine = engine
        self.device = device
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self._pipeline: Any | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str | Path) -> list[StructuredPage]:
        """
        Run OCR on all pages of a PDF and return one StructuredPage per page.

        Parameters
        ----------
        pdf_path : path to the PDF file

        Returns
        -------
        list of StructuredPage, one per page (1-based page numbers)

        Raises
        ------
        FileNotFoundError : if pdf_path does not exist
        ImportError       : if PaddleOCR is not installed
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        pipeline = self._get_pipeline()
        pages: list[StructuredPage] = []
        predictions = pipeline.predict(input=str(path))

        for page_number, result in enumerate(predictions, start=1):
            raw_text = self._extract_raw_text(result)
            section_headers = self._extract_section_headers(raw_text)
            key_value_pairs = self._extract_key_value_pairs(raw_text)
            index_text = self._build_index_text(
                section_headers=section_headers,
                key_value_pairs=key_value_pairs,
                raw_text=raw_text,
            )
            pages.append(StructuredPage(
                page_number=page_number,
                raw_text=raw_text,
                section_headers=section_headers,
                key_value_pairs=key_value_pairs,
                index_text=index_text,
            ))

        return pages

    def process_text(self, raw_text: str, page_number: int = 1) -> StructuredPage:
        """
        Build a StructuredPage from pre-extracted text (e.g. PyMuPDF fallback).

        Used when PaddleOCR is unavailable or for plain-text PDFs.

        Parameters
        ----------
        raw_text    : plain text content of the page
        page_number : 1-based page number (default 1)

        Returns
        -------
        StructuredPage
        """
        section_headers = self._extract_section_headers(raw_text)
        key_value_pairs = self._extract_key_value_pairs(raw_text)
        index_text = self._build_index_text(
            section_headers=section_headers,
            key_value_pairs=key_value_pairs,
            raw_text=raw_text,
        )
        return StructuredPage(
            page_number=page_number,
            raw_text=raw_text,
            section_headers=section_headers,
            key_value_pairs=key_value_pairs,
            index_text=index_text,
        )

    # ── private: pipeline management ─────────────────────────────────────────

    def _get_pipeline(self) -> Any:
        """Return the cached pipeline, initialising it on first call."""
        if self._pipeline is None:
            self._pipeline = self._create_paddle_pipeline()
        return self._pipeline

    def _create_paddle_pipeline(self) -> Any:
        """
        Initialise PaddleOCR Mobile pipeline.

        Uses PP-OCRv3 mobile recognition model for fast CPU inference.
        MKL-DNN is disabled for compatibility across environments.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is required for the 'paddle' backend. "
                "Install it with: pip install paddleocr"
            ) from exc

        return PaddleOCR(
            device=self.device,
            text_recognition_model_name="PP-OCRv3_mobile_rec",
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            enable_mkldnn=False,
        )

    # ── private: text processing ──────────────────────────────────────────────

    def _extract_raw_text(self, result: Any) -> str:
        """Extract raw text from a PaddleOCR prediction result."""
        texts = result.get("rec_texts", []) if isinstance(result, dict) else []
        return " ".join(texts)

    def _extract_section_headers(self, raw_text: str) -> list[str]:
        """
        Heuristically identify section headers in page text.

        A line is treated as a header if it is:
          - Shorter than 60 characters
          - At most 6 words
          - Entirely uppercase OR title case
          - Does not contain a colon (colon lines are key-value pairs)
        """
        headers: list[str] = []
        for line in raw_text.splitlines():
            candidate = self._normalize_line(line)
            if not candidate:
                continue
            if len(candidate) > 60:
                continue
            if ":" in candidate:
                continue
            if len(candidate.split()) > 6:
                continue
            if candidate.isupper() or self._is_title_case(candidate):
                headers.append(candidate)
        return headers

    def _extract_key_value_pairs(self, raw_text: str) -> dict[str, str]:
        """Extract 'Label: Value' pairs from page text."""
        pairs: dict[str, str] = {}
        for key, value in self._KEY_VALUE_PATTERN.findall(raw_text):
            cleaned_key = self._normalize_line(key)
            cleaned_value = " ".join(value.split())
            if cleaned_key:
                pairs[cleaned_key] = cleaned_value
        return pairs

    def _build_index_text(
        self,
        section_headers: list[str],
        key_value_pairs: dict[str, str],
        raw_text: str,
    ) -> str:
        """
        Assemble the composite index text for NemoRetriever embedding.

        Format:
            SECTIONS: <h1> | <h2>
            KEY_VALUES: <k1>: <v1> | <k2>: <v2>
            RAW_TEXT:
            <full raw text>
        """
        section_text = " | ".join(section_headers) if section_headers else "NONE"
        kv_text = (
            " | ".join(f"{k}: {v}" for k, v in key_value_pairs.items())
            if key_value_pairs
            else "NONE"
        )
        return "\n".join([
            f"SECTIONS: {section_text}",
            f"KEY_VALUES: {kv_text}",
            "RAW_TEXT:",
            raw_text or "",
        ])

    def _is_title_case(self, text: str) -> bool:
        """Return True if every alphabetic word starts with an uppercase letter."""
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            return False
        return all(
            word[0].isupper() and word[1:].islower()
            for word in words
            if word[0].isalpha()
        )

    def _normalize_line(self, line: str) -> str:
        """Strip leading/trailing whitespace and markdown formatting characters."""
        cleaned = line.strip()
        cleaned = re.sub(r"^[#>\-*]+\s*", "", cleaned)
        return " ".join(cleaned.split())
