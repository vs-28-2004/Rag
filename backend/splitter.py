"""Chunking logic built on LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.loader import Doc

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def split_documents(
    docs: list[Doc],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Doc]:
    """Split a list of Docs into smaller overlapping chunks, preserving metadata.

    A running `chunk_index` is added to metadata so each chunk can be uniquely
    identified inside the vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Doc] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            chunks.append(Doc(page_content=piece, metadata=meta))
    return chunks
