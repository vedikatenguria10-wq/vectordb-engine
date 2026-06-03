"""Word-based text chunking with overlap for document ingestion."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[tuple[str, int]]:
    """Split text into overlapping word chunks as (chunk_text, chunk_index) pairs."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [(text, 0)]

    step = chunk_size - overlap
    chunks: list[tuple[str, int]] = []
    chunk_index = 0

    for i in range(0, len(words), step):
        end = min(i + chunk_size, len(words))
        chunk_words = words[i:end]
        chunk_str = " ".join(chunk_words)
        chunks.append((chunk_str, chunk_index))
        chunk_index += 1
        if end == len(words):
            break

    return chunks
