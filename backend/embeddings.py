"""Embedding generation using the Gemini Embeddings API, with an automatic
fallback to a local Sentence Transformers model if Gemini embeddings are
unavailable (missing key, quota, or network failure).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when neither Gemini nor the local fallback can produce embeddings."""


@lru_cache(maxsize=1)
def _get_fallback_model():
    """Lazily load the local Sentence Transformers model (cached across calls)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(FALLBACK_MODEL_NAME)


class EmbeddingEngine:
    """Wraps embedding generation, tracking which backend is actively in use."""

    def __init__(self, client=None):
        self.client = client
        self.backend = "gemini" if client is not None else "fallback"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an (N, dim) float32 numpy array."""
        if not texts:
            return np.zeros((0, 384), dtype="float32")

        if self.client is not None:
            try:
                return self._embed_with_gemini(texts)
            except Exception:
                # Fall back silently to the local model so the app keeps working.
                self.backend = "fallback"

        return self._embed_with_fallback(texts)

    def _embed_with_gemini(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        # Gemini embeds one string (or small batch) per call; batch conservatively.
        for text in texts:
            result = self.client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text,
            )
            vectors.append(result.embeddings[0].values)
        self.backend = "gemini"
        return np.array(vectors, dtype="float32")

    def _embed_with_fallback(self, texts: list[str]) -> np.ndarray:
        model = _get_fallback_model()
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.backend = "fallback"
        return np.array(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([text])[0]
