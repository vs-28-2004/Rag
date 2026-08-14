"""OCR utilities for scanned PDFs and standalone image uploads.

Uses Tesseract via pytesseract. Requires the `tesseract-ocr` system binary
(e.g. `apt-get install tesseract-ocr` / `brew install tesseract`). All
functions degrade gracefully (raise a clear OCRUnavailableError) if the
binary isn't installed, so the rest of the app keeps working.
"""

from __future__ import annotations

import io
from functools import lru_cache

from backend.utils import clean_text


class OCRUnavailableError(Exception):
    """Raised when Tesseract is not installed or OCR fails entirely."""


@lru_cache(maxsize=1)
def _ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes (jpg/png/etc.) and return extracted text."""
    if not _ocr_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed on this system. "
            "Install it with `apt-get install tesseract-ocr` (Linux) or "
            "`brew install tesseract` (macOS)."
        )
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img)
    return clean_text(text)


def ocr_pdf_page(page, zoom: float = 2.0) -> str:
    """Render a single PyMuPDF page to an image and OCR it.

    `page` is a pymupdf.Page object. Rendering at a higher zoom improves
    OCR accuracy on small or low-resolution scans.
    """
    if not _ocr_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed on this system. "
            "Install it with `apt-get install tesseract-ocr` (Linux) or "
            "`brew install tesseract` (macOS)."
        )
    import pymupdf

    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    image_bytes = pix.tobytes("png")
    return ocr_image_bytes(image_bytes)
