"""Hybrid semantic + BM25 search with reciprocal rank fusion."""

from __future__ import annotations

from typing import Any, Optional

from rank_bm25 import BM25Okapi

from db.vector_store import VectorStore
from embeddings.encoder import SentenceEncoder

_RRF_K = 60


class BM25Search:
    """Keyword search index backed by rank-bm25."""

    def __init__(self) -> None:
        """Create an empty BM25 index."""
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: list[int] = []

    def build_index(self, documents: list[str], doc_ids: list[int]) -> None:
        """Build a BM25 index over the given documents and parallel doc ids."""
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have the same length")
        self._doc_ids = list(doc_ids)
        tokenized = [doc.split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (doc_id, score) pairs sorted by BM25 score descending."""
        if self._bm25 is None or not self._doc_ids:
            return []

        scores = self._bm25.get_scores(query.split())
        ranked = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [(int(doc_id), float(score)) for doc_id, score in ranked[:k]]


class HybridSearch:
    """Combines VectorStore semantic search with BM25 using weighted RRF."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Search) -> None:
        """Attach a vector store and a BM25 index for hybrid retrieval."""
        self._vector_store = vector_store
        self._bm25 = bm25
        self._encoder = SentenceEncoder()

    def search(
        self, query_text: str, k: int, alpha: float = 0.5
    ) -> list[dict[str, Any]]:
        """Run hybrid search and return the top k fused results."""
        candidate_k = 2 * k

        # Step 1: Encode query_text to vector using SentenceEncoder
        query_vector = self._encoder.encode(query_text)

        # Step 2: Get top 2k results from VectorStore (semantic search)
        semantic_hits = self._vector_store.search(query_vector, candidate_k)
        semantic_ranks: dict[int, int] = {
            int(hit["id"]): rank for rank, hit in enumerate(semantic_hits, start=1)
        }

        # Step 3: Get top 2k results from BM25Search (keyword search)
        bm25_hits = self._bm25.search(query_text, candidate_k)
        bm25_ranks: dict[int, int] = {
            doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits, start=1)
        }

        # Step 4: Combine using RRF formula
        all_ids = set(semantic_ranks) | set(bm25_ranks)
        fused: list[tuple[int, float, Optional[int], Optional[int]]] = []

        for doc_id in all_ids:
            sem_rank = semantic_ranks.get(doc_id)
            bm25_rank = bm25_ranks.get(doc_id)
            combined_score = 0.0
            if sem_rank is not None:
                combined_score += alpha * (1.0 / (_RRF_K + sem_rank))
            if bm25_rank is not None:
                combined_score += (1.0 - alpha) * (1.0 / (_RRF_K + bm25_rank))
            fused.append((doc_id, combined_score, sem_rank, bm25_rank))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_fused = fused[:k]

        # Step 5: Sort by combined score, return top k as list of dicts
        metadata = {int(hit["id"]): hit for hit in semantic_hits}
        missing_ids = {doc_id for doc_id, _, _, _ in top_fused if doc_id not in metadata}
        if missing_ids:
            for item in self._vector_store.list_all():
                item_id = int(item["id"])
                if item_id in missing_ids:
                    metadata[item_id] = item

        results: list[dict[str, Any]] = []
        for doc_id, combined_score, sem_rank, bm25_rank in top_fused:
            meta = metadata.get(doc_id, {})
            results.append(
                {
                    "id": doc_id,
                    "text": str(meta.get("text", "")),
                    "category": str(meta.get("category", "")),
                    "semantic_rank": sem_rank,
                    "bm25_rank": bm25_rank,
                    "combined_score": float(combined_score),
                }
            )

        return results
