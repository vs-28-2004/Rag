"""Orchestrates retrieval + Gemini generation, including streaming responses
and graceful handling of API failures.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator

from google import genai
from google.genai import errors as genai_errors

from backend.embeddings import EmbeddingEngine
from backend.prompts import build_rag_prompt
from backend.retriever import RetrievedChunk, format_citation, retrieve
from backend.vector_store import VectorStore

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


class GeminiUnavailableError(Exception):
    """Raised when the Gemini API cannot be reached or the key is invalid."""


def get_client() -> genai.Client | None:
    """Create a Gemini client from the GEMINI_API_KEY env var, or None if missing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def answer_question(
    client: genai.Client,
    question: str,
    store: VectorStore,
    engine: EmbeddingEngine,
    history: str = "",
    top_k: int = 4,
    mode: str = "semantic",
    temperature: float = 0.4,
    max_output_tokens: int = 1024,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> Generator[str, None, list[RetrievedChunk]]:
    """Stream a grounded answer to `question`, yielding text chunks as they arrive.

    Returns (via StopIteration.value semantics handled by the caller) the list
    of retrieved chunks used, so the UI can render citations after streaming.
    """
    chunks = retrieve(question, store, engine, top_k=top_k, mode=mode)
    context_blocks = [
        f"[{format_citation(c.doc)}]\n{c.doc.page_content}" for c in chunks
    ]
    prompt = build_rag_prompt(question, context_blocks, history=history)

    if client is None:
        yield (
            "⚠️ Gemini API key is not configured. Please set `GEMINI_API_KEY` in your "
            "`.env` file to get real answers. Showing retrieved context only.\n\n"
            + "\n\n".join(context_blocks[:2])
        )
        return chunks

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.interactions.create(
                model="gemini-3.1-flash-lite",
                input=prompt,
            )

            text = response.output_text or ""
            if text:
                yield text

            return chunks
            for event in stream:
                text = getattr(event, "text", None)
                if text:
                    yield text
            return chunks
        except genai_errors.APIError as exc:
            last_error = exc
            status = getattr(exc, "code", None)
            if status == 429 and attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break

    yield f"\n\n⚠️ Gemini API error: {last_error}. Please check your API key, quota, or network connection."
    return chunks


def generate_text(
    client: genai.Client,
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 512,
) -> str:
    """Non-streaming single-shot generation, used for summaries/insights/FAQs."""
    if client is None:
        return "⚠️ Gemini API key is not configured."

    try:
        response = client.interactions.create(
            model="gemini-3.1-flash-lite",
            input=prompt,
        )
        return response.output_text or ""

    except Exception as exc:
        return f"⚠️ Gemini API error: {exc}"