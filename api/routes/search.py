"""Search, benchmark, and index statistics API routes."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query

from api.auth import get_document_store
from core.brute_force import BruteForce
from core.hnsw_custom import HNSWCustom
from core.hnsw_faiss import HNSWFaiss
from core.hybrid_search import BM25Search, HybridSearch
from core.kdtree import KDTree
from db.schema import SearchResult
from db.vector_store import VectorStore
from embeddings.encoder import SentenceEncoder

load_dotenv()

router = APIRouter(prefix="/api", tags=["search"])

ALGORITHMS = ("hnsw_custom", "hnsw_faiss", "kdtree", "bruteforce", "hybrid")
BENCHMARK_ALGOS = ALGORITHMS


@lru_cache
def get_vector_store() -> VectorStore:
    """Return a cached LanceDB vector store."""
    db_path = os.getenv("DB_PATH", "./data/vectordb")
    return VectorStore(db_path)


class _IndexManager:
    """In-memory algorithm indexes rebuilt from VectorStore when data changes."""

    def __init__(self) -> None:
        self.vector_count: int = -1
        self.metadata: dict[int, dict[str, str]] = {}
        self.dims: int = 384
        self.brute_force = BruteForce()
        self.kdtree = KDTree(self.dims)
        self.hnsw_custom = HNSWCustom(m=16, ef_build=200, seed=42)
        self.hnsw_faiss = HNSWFaiss(dims=384, M=32)
        self.bm25 = BM25Search()
        self.hybrid: HybridSearch | None = None

    def sync(self, store: VectorStore) -> None:
        """Reload indexes if the vector store row count changed."""
        stats = store.get_stats()
        count = int(stats["count"])
        if count == self.vector_count:
            return

        self.vector_count = count
        self.metadata.clear()
        self.brute_force = BruteForce()
        self.hnsw_custom = HNSWCustom(m=16, ef_build=200, seed=42)
        self.hnsw_faiss = HNSWFaiss(dims=384, M=32)

        items = store.list_all()
        if not items:
            self.dims = 384
            self.kdtree = KDTree(self.dims)
            self.bm25.build_index([], [])
            self.hybrid = HybridSearch(store, self.bm25)
            return

        self.dims = len(items[0]["vector"])
        self.kdtree = KDTree(self.dims)
        documents: list[str] = []
        doc_ids: list[int] = []

        for item in items:
            item_id = int(item["id"])
            vector = np.asarray(item["vector"], dtype=np.float32)
            text = str(item["text"])
            category = str(item["category"])
            self.metadata[item_id] = {"text": text, "category": category}
            self.brute_force.insert(item_id, vector)
            self.kdtree.insert(item_id, vector)
            self.hnsw_custom.insert(item_id, vector)
            self.hnsw_faiss.insert(item_id, vector)
            documents.append(text)
            doc_ids.append(item_id)

        self.bm25.build_index(documents, doc_ids)
        self.hybrid = HybridSearch(store, self.bm25)

    def get_hybrid(self, store: VectorStore) -> HybridSearch:
        """Return a hybrid searcher bound to the current vector store."""
        if self.hybrid is None:
            self.hybrid = HybridSearch(store, self.bm25)
        return self.hybrid


_manager = _IndexManager()
_encoder = SentenceEncoder()


def get_index_manager() -> _IndexManager:
    """Return indexes synchronized with the vector store."""
    store = get_vector_store()
    _manager.sync(store)
    return _manager


def invalidate_search_indexes() -> None:
    """Force indexes to rebuild on the next search request."""
    _manager.vector_count = -1


def _to_search_results(
    hits: list[tuple[float, int]],
    algo: str,
    metadata: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert (score, id) hits into SearchResult dictionaries."""
    results: list[dict[str, Any]] = []
    for score, item_id in hits:
        meta = metadata.get(item_id, {"text": "", "category": ""})
        result = SearchResult(
            id=item_id,
            text=meta["text"],
            category=meta["category"],
            score=float(score),
            algorithm=algo,
        )
        results.append(result.model_dump())
    return results


def _hybrid_to_search_results(
    hits: list[dict[str, Any]], algo: str
) -> list[dict[str, Any]]:
    """Convert hybrid fusion dicts into SearchResult dictionaries."""
    results: list[dict[str, Any]] = []
    for hit in hits:
        result = SearchResult(
            id=int(hit["id"]),
            text=str(hit["text"]),
            category=str(hit["category"]),
            score=float(hit["combined_score"]),
            algorithm=algo,
        )
        results.append(result.model_dump())
    return results


def _run_algorithm(
    algo: str,
    query_text: str,
    query_vector: np.ndarray,
    k: int,
    metric: str,
    manager: _IndexManager,
    store: VectorStore,
) -> list[dict[str, Any]]:
    """Dispatch search to the selected algorithm implementation."""
    if algo == "bruteforce":
        hits = manager.brute_force.search(query_vector, k, metric)
        return _to_search_results(hits, algo, manager.metadata)
    if algo == "kdtree":
        hits = manager.kdtree.search(query_vector, k, metric)
        return _to_search_results(hits, algo, manager.metadata)
    if algo == "hnsw_custom":
        hits = manager.hnsw_custom.search(query_vector, k, metric)
        return _to_search_results(hits, algo, manager.metadata)
    if algo == "hnsw_faiss":
        hits = manager.hnsw_faiss.search(query_vector, k, metric)
        return _to_search_results(hits, algo, manager.metadata)
    if algo == "hybrid":
        hybrid = manager.get_hybrid(store)
        hits = hybrid.search(query_text, k)
        return _hybrid_to_search_results(hits, algo)
    raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo}")


@router.get("/search")
def api_search(
    q: str = Query(..., description="Query text to encode and search"),
    k: int = Query(5, ge=1),
    metric: str = Query("cosine"),
    algo: str = Query("hnsw_faiss"),
) -> dict[str, Any]:
    """Encode a query and run the selected vector search algorithm."""
    if algo not in ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"algo must be one of: {', '.join(ALGORITHMS)}",
        )

    manager = get_index_manager()
    store = get_vector_store()
    query_vector = _encoder.encode(q)

    t0 = time.perf_counter()
    results = _run_algorithm(algo, q, query_vector, k, metric, manager, store)
    time_ms = (time.perf_counter() - t0) * 1000.0

    return {"results": results, "time_ms": time_ms}


@router.get("/benchmark")
def api_benchmark(
    q: str = Query(..., description="Query text to encode and search"),
    k: int = Query(5, ge=1),
    metric: str = Query("cosine"),
) -> dict[str, Any]:
    """Run every search algorithm on the same query and report timings."""
    manager = get_index_manager()
    store = get_vector_store()
    query_vector = _encoder.encode(q)
    output: dict[str, Any] = {}

    for algo in BENCHMARK_ALGOS:
        t0 = time.perf_counter()
        results = _run_algorithm(algo, q, query_vector, k, metric, manager, store)
        time_ms = (time.perf_counter() - t0) * 1000.0
        output[algo] = {"results": results, "time_ms": time_ms}

    return output


@router.get("/hnsw-info")
def api_hnsw_info() -> dict[str, Any]:
    """Return layer and graph statistics from the custom HNSW index."""
    manager = get_index_manager()
    return manager.hnsw_custom.get_info()


@router.get("/stats")
def api_stats() -> dict[str, Any]:
    """Return combined VectorStore and DocumentStore statistics."""
    vector_stats = get_vector_store().get_stats()
    document_stats = get_document_store().get_stats()
    return {
        "vector_store": vector_stats,
        "document_store": document_stats,
    }
