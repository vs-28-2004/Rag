"""Retrieval logic: semantic search (FAISS + embeddings) and a simple keyword
search fallback/alternative mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.embeddings import EmbeddingEngine
from backend.loader import Doc
from backend.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    doc: Doc
    score: float


def semantic_search(
    query: str,
    store: VectorStore,
    engine: EmbeddingEngine,
    top_k: int = 4,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks."""
    if store.is_empty():
        return []
    query_vector = engine.embed_query(query)
    results = store.search(query_vector, top_k=top_k)
    return [RetrievedChunk(doc=d, score=s) for d, s in results]


def keyword_search(query: str, store: VectorStore, top_k: int = 4) -> list[RetrievedChunk]:
    """Rank chunks by simple keyword overlap (case-insensitive token matching)."""
    if store.is_empty():
        return []
    tokens = set(re.findall(r"\b\w+\b", query.lower()))
    if not tokens:
        return []

    scored: list[RetrievedChunk] = []
    for doc in store.docs:
        content_tokens = re.findall(r"\b\w+\b", doc.page_content.lower())
        if not content_tokens:
            continue
        overlap = sum(1 for t in content_tokens if t in tokens)
        if overlap > 0:
            score = overlap / (len(content_tokens) ** 0.5)  # mild length normalization
            scored.append(RetrievedChunk(doc=doc, score=score))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def retrieve(
    query: str,
    store: VectorStore,
    engine: EmbeddingEngine,
    top_k: int = 4,
    mode: str = "semantic",
) -> list[RetrievedChunk]:
    """Dispatch to the requested retrieval mode."""
    if mode == "keyword":
        return keyword_search(query, store, top_k=top_k)
    return semantic_search(query, store, engine, top_k=top_k)


def format_citation(doc: Doc) -> str:
    """Format a human-readable citation string for a retrieved chunk."""
    meta = doc.metadata
    source = meta.get("source", "unknown")
    if meta.get("type") == "pdf":
        page_str = f"{source} (Page {meta.get('page')})"
        if meta.get("extraction") == "ocr":
            page_str += " · OCR"
        return page_str
    if meta.get("type") == "csv":
        return f"{source} (Rows {meta.get('row_start')}–{meta.get('row_end')})"
    if meta.get("type") == "txt":
        return f"{source} (Section {meta.get('section')})"
    if meta.get("type") == "image":
        return f"{source} (Image · OCR)"
    return source


def highlight_terms(text: str, query_or_answer: str, max_terms: int = 12) -> str:
    """Wrap words from `query_or_answer` that appear in `text` with <mark> tags.

    Used to visually highlight the parts of a retrieved chunk that likely
    informed the generated answer, without doing exact-span alignment
    (which the LLM output doesn't guarantee character-for-character).
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "and", "or", "for", "with", "this", "that", "it", "as", "by", "at",
        "be", "has", "have", "had", "not", "but", "from", "what", "how",
        "why", "does", "do", "did", "can", "could", "would", "should",
    }
    terms = [t for t in re.findall(r"\b\w{4,}\b", query_or_answer.lower()) if t not in stopwords]
    # Preserve order, dedupe, cap count to avoid over-highlighting.
    seen: set[str] = set()
    ordered_terms = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            ordered_terms.append(t)
    ordered_terms = ordered_terms[:max_terms]
    if not ordered_terms:
        return text

    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in ordered_terms) + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark class='rg-highlight'>\1</mark>", text)
