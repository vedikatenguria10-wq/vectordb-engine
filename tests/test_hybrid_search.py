"""Tests for BM25 and hybrid search."""

from __future__ import annotations

import numpy as np

from core.hybrid_search import BM25Search, HybridSearch
from db.vector_store import VectorStore


def _seed_vector_store(store: VectorStore, n: int = 10) -> list[int]:
    """Insert n labeled vectors and return their ids."""
    rng = np.random.default_rng(99)
    ids: list[int] = []
    topics = ["world news report", "sports game score", "business market stock"] * 4
    for i in range(n):
        vec = rng.random(384, dtype=np.float32)
        text = f"{topics[i]} document number {i}"
        store.insert(i, vec, text, "news")
        ids.append(i)
    return ids


def test_bm25_returns_results(vector_store: VectorStore) -> None:
    """BM25 search returns k scored document ids."""
    _seed_vector_store(vector_store, 10)
    items = vector_store.list_all()
    documents = [str(item["text"]) for item in items]
    doc_ids = [int(item["id"]) for item in items]

    bm25 = BM25Search()
    bm25.build_index(documents, doc_ids)

    k = 5
    results = bm25.search("business market", k)
    assert len(results) == k
    assert all(isinstance(doc_id, int) and isinstance(score, float) for doc_id, score in results)


def test_hybrid_search_returns_results(vector_store: VectorStore) -> None:
    """Hybrid search returns k results including combined_score."""
    _seed_vector_store(vector_store, 10)
    items = vector_store.list_all()
    documents = [str(item["text"]) for item in items]
    doc_ids = [int(item["id"]) for item in items]

    bm25 = BM25Search()
    bm25.build_index(documents, doc_ids)
    hybrid = HybridSearch(vector_store, bm25)

    k = 4
    results = hybrid.search("sports game", k, alpha=0.5)
    assert len(results) == k
    assert all("combined_score" in hit for hit in results)


def test_alpha_one_matches_semantic(vector_store: VectorStore) -> None:
    """Alpha=1.0 hybrid fusion should match the top pure semantic hit."""
    _seed_vector_store(vector_store, 10)
    items = vector_store.list_all()
    documents = [str(item["text"]) for item in items]
    doc_ids = [int(item["id"]) for item in items]

    bm25 = BM25Search()
    bm25.build_index(documents, doc_ids)
    hybrid = HybridSearch(vector_store, bm25)

    query_text = "world news report document"
    query_vector = hybrid._encoder.encode(query_text)

    semantic_hits = vector_store.search(query_vector, 1)
    assert semantic_hits
    semantic_top_id = int(semantic_hits[0]["id"])

    hybrid_hits = hybrid.search(query_text, 1, alpha=1.0)
    assert hybrid_hits
    hybrid_top_id = int(hybrid_hits[0]["id"])

    assert hybrid_top_id == semantic_top_id
