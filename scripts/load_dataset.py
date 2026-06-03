#!/usr/bin/env python3
"""Load AG News into the LanceDB vector store (idempotent)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.loader import load_agnews
from datasets.preprocessor import process_records
from db.vector_store import VectorStore

_CHUNK_ID_MULTIPLIER = 10_000


def _chunk_id_to_int(chunk_id: str) -> int:
    """Map a chunk id like ``123_2`` to a stable integer for VectorStore."""
    record_id, chunk_index = chunk_id.rsplit("_", 1)
    return int(record_id) * _CHUNK_ID_MULTIPLIER + int(chunk_index)


def main() -> None:
    """Load, preprocess, and insert AG News vectors if the store is empty."""
    load_dotenv()
    db_path = os.getenv("DB_PATH", "./data/vectordb")
    store = VectorStore(db_path)

    stats = store.get_stats()
    if stats["count"] > 0:
        print(
            f"Data already exists ({stats['count']} vectors). "
            "Skipping load (idempotent)."
        )
        return

    print("Loading AG News from HuggingFace (train split)...")
    records = load_agnews(max_records=2000)
    print(f"Loaded {len(records)} records. Preprocessing...")
    processed = process_records(records)
    print(f"Inserting {len(processed)} vectors into LanceDB...")

    for item in tqdm(processed, desc="Inserting vectors"):
        store.insert(
            _chunk_id_to_int(str(item["id"])),
            item["vector"],
            str(item["text"]),
            str(item["category"]),
        )

    final = store.get_stats()
    print(f"Done. Final count: {final['count']} vectors stored ({final['dims']} dims).")


if __name__ == "__main__":
    main()
