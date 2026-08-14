"""Document loading and text extraction for PDF, CSV, and TXT files.

Each loader returns a list of LangChain-style Document objects (dict form:
{"page_content": str, "metadata": dict}) with metadata preserved per the
project spec (filename + page/row/section references).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pymupdf as fitz  # PyMuPDF (modern import path)
import pandas as pd

from backend.ocr import OCRUnavailableError, ocr_image_bytes, ocr_pdf_page
from backend.utils import clean_text


@dataclass
class Doc:
    """Lightweight document container (avoids a hard LangChain Document dependency)."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pdf(file_bytes: bytes, filename: str, ocr_fallback: bool = True) -> list[Doc]:
    """Extract text from a PDF, one Doc per page, preserving page numbers.

    If a page has no extractable text (common for scanned/image-only PDFs)
    and `ocr_fallback` is True, the page is rendered to an image and run
    through Tesseract OCR instead of being skipped.
    """
    docs: list[Doc] = []
    ocr_used = False
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text = clean_text(page.get_text())
            source = "native"
            if not text and ocr_fallback:
                try:
                    text = ocr_pdf_page(page)
                    source = "ocr"
                    if text:
                        ocr_used = True
                except OCRUnavailableError:
                    text = ""
            if not text:
                continue
            docs.append(
                Doc(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "type": "pdf",
                        "page": page_index + 1,
                        "total_pages": len(pdf),
                        "extraction": source,
                    },
                )
            )
    if not docs:
        raise ValueError(
            f"No extractable text found in '{filename}', even after OCR. "
            "It may be a blank, corrupted, or unsupported PDF."
        )
    if ocr_used:
        # Mark on every doc so the UI can surface "OCR used" for this file.
        for d in docs:
            d.metadata.setdefault("ocr_used_in_file", True)
    return docs


def load_csv(file_bytes: bytes, filename: str, rows_per_chunk: int = 25) -> list[Doc]:
    """Load a CSV, grouping rows into blocks and recording row ranges + column names."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse CSV '{filename}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV '{filename}' has no rows.")

    columns = list(df.columns)
    docs: list[Doc] = []
    for start in range(0, len(df), rows_per_chunk):
        chunk_df = df.iloc[start : start + rows_per_chunk]
        end = start + len(chunk_df) - 1
        text_lines = [", ".join(columns)]
        for _, row in chunk_df.iterrows():
            text_lines.append(", ".join(str(v) for v in row.values))
        text = clean_text("\n".join(text_lines))
        docs.append(
            Doc(
                page_content=text,
                metadata={
                    "source": filename,
                    "type": "csv",
                    "row_start": start + 1,
                    "row_end": end + 1,
                    "columns": columns,
                },
            )
        )
    return docs


def load_txt(file_bytes: bytes, filename: str, section_size_chars: int = 2000) -> list[Doc]:
    """Load a TXT file, splitting into numbered sections for citation purposes."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    text = clean_text(text)
    if not text:
        raise ValueError(f"TXT file '{filename}' is empty.")

    docs: list[Doc] = []
    for i, start in enumerate(range(0, len(text), section_size_chars), start=1):
        section_text = text[start : start + section_size_chars]
        docs.append(
            Doc(
                page_content=section_text,
                metadata={"source": filename, "type": "txt", "section": i},
            )
        )
    return docs


def load_image(file_bytes: bytes, filename: str) -> list[Doc]:
    """OCR a standalone image upload (jpg/png/etc.) into a single Doc."""
    text = ocr_image_bytes(file_bytes)
    if not text:
        raise ValueError(f"No readable text found in image '{filename}'.")
    return [
        Doc(
            page_content=text,
            metadata={"source": filename, "type": "image", "extraction": "ocr"},
        )
    ]


def load_file(file_bytes: bytes, filename: str) -> list[Doc]:
    """Dispatch to the correct loader based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return load_pdf(file_bytes, filename)
    if ext == "csv":
        return load_csv(file_bytes, filename)
    if ext == "txt":
        return load_txt(file_bytes, filename)
    if ext in ("jpg", "jpeg", "png", "webp", "bmp"):
        return load_image(file_bytes, filename)
    raise ValueError(f"Unsupported file type: .{ext}")
