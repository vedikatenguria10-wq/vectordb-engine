"""Chunk and encode AG News records for vector storage."""

from __future__ import annotations

from typing import Any

import numpy as np

from embeddings.chunker import chunk_text
from embeddings.encoder import SentenceEncoder


def process_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk each record's text and encode every chunk with SentenceEncoder."""
    encoder = SentenceEncoder()
    processed: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        record_id = int(record["id"])
        category = str(record["category"])
        chunks = chunk_text(str(record["text"]))

        if not chunks:
            continue

        chunk_texts = [c[0] for c in chunks]
        vectors = encoder.encode_batch(chunk_texts)

        for (chunk_str, chunk_index), vector in zip(chunks, vectors):
            processed.append(
                {
                    "id": f"{record_id}_{chunk_index}",
                    "vector": np.asarray(vector, dtype=np.float32),
                    "text": chunk_str,
                    "category": category,
                }
            )

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(records)} records")

    return processed
