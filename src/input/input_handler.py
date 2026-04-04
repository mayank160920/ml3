"""
---------------------------
Routes an input file to the correct loading path based on file extension.

PDF  → PageImageStore (full RAG pipeline path)
Image → ImageLoader   (direct MLLM path)

Both paths produce an identical LoadedInput output structure, ensuring
all downstream layers are agnostic to the input type.
"""
from __future__ import annotations

from pathlib import Path

from ingestion.page_image_store import PageImageStore
from input.image_loader import ImageLoader
from shared_types import InputType, LoadedInput


class InputHandler:
    """
    Detects input type and delegates loading to the appropriate loader.

    Supported PDF extensions   : .pdf
    Supported image extensions : .jpg .jpeg .png .tiff .tif .bmp .webp

    Parameters
    ----------
    image_loader      : ImageLoader instance (created if not provided)
    page_image_store  : PageImageStore instance (created if not provided)
    """

    PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
    IMAGE_EXTENSIONS: frozenset[str] = frozenset({
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"
    })

    def __init__(
        self,
        image_loader: ImageLoader | None = None,
        page_image_store: PageImageStore | None = None,
    ) -> None:
        self.image_loader = image_loader or ImageLoader()
        self.page_image_store = page_image_store or PageImageStore()

    # ── public API ────────────────────────────────────────────────────────────

    def detect_input_type(self, path: str | Path) -> InputType:
        """
        Determine whether a path points to a PDF or an image file.

        Parameters
        ----------
        path : file path to inspect

        Returns
        -------
        InputType.PDF or InputType.IMAGE

        Raises
        ------
        ValueError : if the extension is not supported
        """
        suffix = Path(path).suffix.lower()
        if suffix in self.PDF_EXTENSIONS:
            return InputType.PDF
        if suffix in self.IMAGE_EXTENSIONS:
            return InputType.IMAGE

        supported = sorted(self.PDF_EXTENSIONS | self.IMAGE_EXTENSIONS)
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Supported formats: {', '.join(supported)}"
        )

    def load(self, input_path: str | Path) -> LoadedInput:
        """
        Load a document and return a unified LoadedInput.

        Routing:
          PDF   → _load_pdf  (renders pages via PyMuPDF, stores in PageImageStore)
          Image → _load_image (opens + base64-encodes via ImageLoader)

        Parameters
        ----------
        input_path : path to a PDF or image file

        Returns
        -------
        LoadedInput

        Raises
        ------
        FileNotFoundError : if the file does not exist
        ValueError        : if the extension is not supported
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        input_type = self.detect_input_type(path)
        if input_type is InputType.PDF:
            return self._load_pdf(path)
        return self._load_image(path)

    # ── private ───────────────────────────────────────────────────────────────

    def _load_pdf(self, pdf_path: Path) -> LoadedInput:
        """Render all PDF pages and return a multi-page LoadedInput."""
        total_pages = self.page_image_store.load_pdf(pdf_path)
        return LoadedInput(
            input_type=InputType.PDF,
            source_path=pdf_path,
            total_pages=total_pages,
            page_images=self.page_image_store.get_all_pages(),
        )

    def _load_image(self, image_path: Path) -> LoadedInput:
        """Load a single image and return a one-page LoadedInput."""
        return self.image_loader.load_image(image_path)
