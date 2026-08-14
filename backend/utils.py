"""General-purpose helper functions used across the RAGenius AI backend."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Iterable

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALL_UPLOAD_EXTENSIONS = ALLOWED_EXTENSIONS | IMAGE_EXTENSIONS
MAX_FILE_SIZE_MB = 50


def is_supported_file(filename: str) -> bool:
    """Return True if the filename has a supported extension (docs or images)."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALL_UPLOAD_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Return True if the filename is an OCR-able image type."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def get_extension(filename: str) -> str:
    """Return the lowercase file extension including the dot."""
    return os.path.splitext(filename)[1].lower()


def file_hash(data: bytes) -> str:
    """Return a stable sha256 hash for a file's bytes, used for duplicate detection."""
    return hashlib.sha256(data).hexdigest()


def validate_file_size(size_bytes: int) -> tuple[bool, str]:
    """Validate a file does not exceed the maximum allowed size."""
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is {size_mb:.1f}MB, which exceeds the {MAX_FILE_SIZE_MB}MB limit."
    if size_bytes == 0:
        return False, "File is empty."
    return True, ""


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from extracted text."""
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    """Return the number of words in a text blob."""
    return len(re.findall(r"\b\w+\b", text))


def estimated_reading_time_minutes(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes assuming an average adult reading speed."""
    words = word_count(text)
    return max(1, round(words / wpm))


def human_timestamp() -> str:
    """Return a human-readable current timestamp, e.g. '14:05:02'."""
    return time.strftime("%H:%M:%S")


def chunk_list(items: list, size: int) -> Iterable[list]:
    """Yield successive chunks of `size` from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def truncate(text: str, max_chars: int = 400) -> str:
    """Truncate text to a maximum number of characters, adding an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def search_snippet(message: str, query: str, radius: int = 60) -> str:
    """Return a short snippet of `message` centered on the first match of `query`."""
    idx = message.lower().find(query.lower())
    if idx == -1:
        return message[: radius * 2]
    start = max(0, idx - radius)
    end = min(len(message), idx + len(query) + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(message) else ""
    return f"{prefix}{message[start:end]}{suffix}"
