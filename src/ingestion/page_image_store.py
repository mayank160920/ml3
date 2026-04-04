"""
----------------------------------
In-memory store of rendered PDF page images.

PyMuPDF renders each PDF page at the configured DPI into an RGB PIL Image,
which is then base64-encoded as PNG and stored keyed by 1-based page number.
The store persists only for the duration of one document's processing.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from shared_types import PageImage


class PageImageStore:
    """
    Renders all pages of a PDF at a given DPI and stores them in memory.

    Provides random access by page number for the RAG retrieval layer.

    Parameters
    ----------
    dpi : dots-per-inch for PDF rendering (default 150)
    """

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi
        self._pages: dict[int, PageImage] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def load_pdf(self, pdf_path: str | Path) -> int:
        """
        Render all pages of a PDF and populate the internal store.

        Parameters
        ----------
        pdf_path : path to the PDF file

        Returns
        -------
        int : total number of pages rendered

        Raises
        ------
        FileNotFoundError : if pdf_path does not exist
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        self._pages = {}
        scale = self.dpi / 72  # 72 DPI is the default PDF unit

        with fitz.open(str(path)) as document:
            for page_index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                self._pages[page_index] = self._to_page_image(
                    image=image,
                    page_number=page_index,
                )

        return len(self._pages)

    def get_pages(self, page_numbers: list[int]) -> dict[int, PageImage]:
        """
        Retrieve specific pages by number.

        Parameters
        ----------
        page_numbers : list of 1-based page numbers

        Returns
        -------
        dict mapping page_number → PageImage

        Raises
        ------
        KeyError : if any requested page number is not in the store
        """
        missing = [n for n in page_numbers if n not in self._pages]
        if missing:
            raise KeyError(f"Requested pages not loaded: {missing}")
        return {n: self._pages[n] for n in page_numbers}

    def get_all_pages(self) -> dict[int, PageImage]:
        """Return all stored pages as a dict keyed by page number."""
        return dict(self._pages)

    @property
    def page_count(self) -> int:
        """Number of pages currently in the store."""
        return len(self._pages)

    def clear(self) -> None:
        """Release all stored page images to free memory."""
        self._pages.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _to_page_image(
        self,
        image: Image.Image,
        page_number: int,
    ) -> PageImage:
        """Encode a PIL Image to base64 PNG and wrap in a PageImage."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return PageImage(
            page_number=page_number,
            image=image,
            image_base64=encoded,
            mime_type="image/png",
            width=image.width,
            height=image.height,
        )
