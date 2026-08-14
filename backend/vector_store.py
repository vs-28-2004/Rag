"""A thin FAISS wrapper providing incremental indexing, persistence, and
metadata-aware search on top of raw vectors.
"""

from __future__ import annotations

import json
import os
import pickle

import faiss
import numpy as np

from backend.loader import Doc


class VectorStore:
    """Manages a FAISS index plus the parallel list of Doc chunks it indexes."""

    def __init__(self, dim: int | None = None):
        self.dim = dim
        self.index: faiss.Index | None = None
        self.docs: list[Doc] = []
        self.file_hashes: set[str] = set()

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dim = dim
            # Inner product on normalized vectors == cosine similarity.
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, docs: list[Doc]) -> None:
        """Incrementally add new vectors and their corresponding docs to the index."""
        if len(vectors) == 0:
            return
        norm_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        self._ensure_index(norm_vectors.shape[1])
        self.index.add(norm_vectors)
        self.docs.extend(docs)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Doc, float]]:
        """Return the top_k (Doc, similarity_score) pairs for a query vector."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_vector.reshape(1, -1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.docs[idx], float(score)))
        return results

    def remove_source(self, filename: str) -> None:
        """Rebuild the index excluding all chunks belonging to `filename`.

        FAISS's flat index doesn't support in-place deletion by id easily
        alongside our parallel doc list, so we rebuild from the retained docs.
        """
        remaining = [d for d in self.docs if d.metadata.get("source") != filename]
        self.docs = []
        self.index = None
        if remaining:
            # Vectors were not cached, so callers must re-embed on full rebuild.
            # This method is used together with re-embedding logic in app.py.
            self.docs = remaining

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def save(self, directory: str) -> None:
        """Persist the FAISS index and doc metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"dim": self.dim, "file_hashes": list(self.file_hashes)}, f)

    def load(self, directory: str) -> bool:
        """Load a previously persisted index. Returns True if successful."""
        index_path = os.path.join(directory, "index.faiss")
        docs_path = os.path.join(directory, "docs.pkl")
        meta_path = os.path.join(directory, "meta.json")
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            return False
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                self.dim = meta.get("dim")
                self.file_hashes = set(meta.get("file_hashes", []))
        return True
