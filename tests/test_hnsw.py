"""Tests for HNSWCustom and HNSWFaiss indexes."""

from __future__ import annotations

import numpy as np
import pytest

from core.hnsw_custom import HNSWCustom
from core.hnsw_faiss import HNSWFaiss

faiss = pytest.importorskip("faiss")


def _random_vectors(n: int, dims: int = 384, seed: int = 0) -> dict[int, np.ndarray]:
    """Create n random unit-scale vectors keyed by id."""
    rng = np.random.default_rng(seed)
    return {i: rng.random(dims, dtype=np.float32) for i in range(n)}


def test_insert_and_search() -> None:
    """Inserting 20 vectors and searching returns exactly k results."""
    hnsw = HNSWCustom(m=8, ef_build=100, seed=7)
    vectors = _random_vectors(20, seed=1)
    for vid, vec in vectors.items():
        hnsw.insert(vid, vec)

    query = vectors[0]
    k = 5
    results = hnsw.search(query, k, metric="cosine")
    assert len(results) == k


def test_search_ids_are_valid() -> None:
    """All ids returned by search must have been inserted."""
    hnsw = HNSWCustom(m=8, ef_build=100, seed=11)
    vectors = _random_vectors(15, seed=2)
    inserted_ids = set(vectors.keys())
    for vid, vec in vectors.items():
        hnsw.insert(vid, vec)

    results = hnsw.search(vectors[3], 7, metric="cosine")
    returned_ids = {doc_id for _, doc_id in results}
    assert returned_ids.issubset(inserted_ids)


def test_custom_vs_faiss_similarity() -> None:
    """HNSWCustom and HNSWFaiss share at least two of the top three hits."""
    vectors = _random_vectors(25, seed=3)
    custom = HNSWCustom(m=8, ef_build=100, seed=13)
    faiss_index = HNSWFaiss(dims=384, M=16)

    for vid, vec in vectors.items():
        custom.insert(vid, vec)
        faiss_index.insert(vid, vec)

    query = vectors[10]
    custom_top = {doc_id for _, doc_id in custom.search(query, 3, metric="cosine")}
    faiss_top = {doc_id for _, doc_id in faiss_index.search(query, 3, metric="cosine")}

    assert len(custom_top & faiss_top) >= 2


def test_delete_removes_vector() -> None:
    """Deleted vectors must not appear in subsequent search results."""
    hnsw = HNSWCustom(m=8, ef_build=100, seed=17)
    vectors = _random_vectors(12, seed=4)
    for vid, vec in vectors.items():
        hnsw.insert(vid, vec)

    deleted_id = 6
    hnsw.delete(deleted_id)

    results = hnsw.search(vectors[0], 12, metric="cosine")
    returned_ids = {doc_id for _, doc_id in results}
    assert deleted_id not in returned_ids
