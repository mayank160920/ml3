"""
--------------------------
Loads image files (JPEG, PNG, TIFF, BMP, WebP) into a LoadedInput object
with a single PageImage entry. Bypasses OCR and retrieval entirely —
the image is passed directly to the MLLM extraction layer.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from shared_types import InputType, LoadedInput, PageImage


class ImageLoader:
    """
    Loads a single image file and wraps it in a LoadedInput container.

    The image is:
      1. Opened and converted to RGB
      2. Resized if it exceeds MAX_IMAGE_SIZE (preserving aspect ratio)
      3. Base64-encoded as PNG for MLLM consumption

    Supported formats: JPEG, PNG, TIFF, BMP, WebP
    """

    MAX_IMAGE_SIZE: tuple[int, int] = (4096, 4096)

    def load_image(self, image_path: str | Path) -> LoadedInput:
        """
        Load an image file and return a single-page LoadedInput.

        Parameters
        ----------
        image_path : path to the image file

        Returns
        -------
        LoadedInput with input_type=IMAGE and one entry in page_images

        Raises
        ------
        FileNotFoundError : if the image file does not exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as img:
            image = img.convert("RGB")
            # Resize only if the image exceeds the maximum allowed dimensions
            if (image.width > self.MAX_IMAGE_SIZE[0]
                    or image.height > self.MAX_IMAGE_SIZE[1]):
                image.thumbnail(self.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        page_image = self._to_page_image(image=image, page_number=1)
        return LoadedInput(
            input_type=InputType.IMAGE,
            source_path=path,
            total_pages=1,
            page_images={1: page_image},
        )

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
