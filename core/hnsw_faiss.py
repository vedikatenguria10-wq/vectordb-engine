"""FAISS-backed HNSW index wrapper."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from core.distance import get_dist_fn

FloatArray = NDArray[np.floating]


def _l2_to_euclidean(dists: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert FAISS squared L2 distances to Euclidean distances."""
    return np.sqrt(np.maximum(dists, 0.0))


class HNSWFaiss:
    """HNSW index using faiss.IndexHNSWFlat with an ID map for external ids."""

    def __init__(self, dims: int = 384, M: int = 32, ef_build: int = 200) -> None:
        """Create an HNSW flat index with the given dimensionality and connectivity M."""
        self.dims = dims
        self.M = M
        self.ef_build = ef_build
        self._ef_search = 50

        base = faiss.IndexHNSWFlat(dims, M)
        base.hnsw.efConstruction = ef_build
        self._index: faiss.IndexIDMap2 = faiss.IndexIDMap2(base)
        self._vectors: dict[int, FloatArray] = {}
        self._metric_store: str = "euclidean"

    def _prepare_vector(self, vector: FloatArray, metric: str) -> FloatArray:
        """Normalize or cast a vector for the active storage metric."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if metric == "cosine":
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                v = v / norm
        return v

    def _rebuild_index(self, metric: str) -> None:
        """Rebuild the FAISS index from stored vectors for the given metric."""
        base = faiss.IndexHNSWFlat(self.dims, self.M)
        base.hnsw.efConstruction = self.ef_build
        if metric == "cosine":
            base.metric_type = faiss.METRIC_INNER_PRODUCT

        self._index = faiss.IndexIDMap2(base)
        if not self._vectors:
            return

        ids = sorted(self._vectors.keys())
        vecs = np.stack(
            [self._prepare_vector(self._vectors[i], metric) for i in ids],
            axis=0,
        ).astype(np.float32)
        id_arr = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(vecs, id_arr)

    def insert(self, id: int, vector: FloatArray) -> None:
        """Add a vector to the index under the given id."""
        v = self._prepare_vector(vector, self._metric_store)
        if v.shape[0] != self.dims:
            raise ValueError(f"expected {self.dims}-dimensional vector, got {v.shape[0]}")

        self._vectors[id] = v.copy()
        self._index.add_with_ids(
            v.reshape(1, -1).astype(np.float32),
            np.array([id], dtype=np.int64),
        )

    def search(
        self, query: FloatArray, k: int, metric: str = "euclidean"
    ) -> list[tuple[float, int]]:
        """Return up to k nearest neighbors; non-L2 metrics re-rank stored vectors."""
        if not self._vectors:
            return []

        if metric != self._metric_store and metric in ("cosine", "euclidean"):
            self._metric_store = metric
            self._rebuild_index(metric)

        dist_fn = get_dist_fn(metric)
        q = self._prepare_vector(query, metric)

        if metric == "manhattan":
            ids = np.array(list(self._vectors.keys()), dtype=np.int64)
            embs = np.stack([self._vectors[i] for i in ids], axis=0)
            dists = dist_fn(q, embs)
            order = np.argsort(dists)[:k]
            return [(float(dists[i]), int(ids[i])) for i in order]

        self._index.hnsw.efSearch = max(self._ef_search, k)

        q_mat = q.reshape(1, -1).astype(np.float32)
        k_search = min(k, self._index.ntotal)
        if k_search <= 0:
            return []

        faiss_dists, faiss_ids = self._index.search(q_mat, k_search)
        results: list[tuple[float, int]] = []

        for d, nid in zip(faiss_dists[0], faiss_ids[0]):
            if nid < 0:
                continue
            nid_int = int(nid)
            if metric == "cosine":
                dist_val = float(dist_fn(q, self._vectors[nid_int]))
            elif metric == "euclidean":
                dist_val = float(_l2_to_euclidean(np.array([d]))[0])
            else:
                dist_val = float(d)
            results.append((dist_val, nid_int))

        if metric == "cosine":
            results.sort(key=lambda x: x[0])
        return results[:k]

    def delete(self, id: int) -> None:
        """Remove a vector from the index by id."""
        if id not in self._vectors:
            return
        del self._vectors[id]
        self._index.remove_ids(np.array([id], dtype=np.int64))

    def get_info(self) -> dict[str, Any]:
        """Return FAISS HNSW index metadata."""
        hnsw = self._index.hnsw
        return {
            "dims": self.dims,
            "M": self.M,
            "efConstruction": self.ef_build,
            "efSearch": getattr(hnsw, "efSearch", self._ef_search),
            "nodeCount": len(self._vectors),
            "ntotal": int(self._index.ntotal),
            "max_level": int(hnsw.max_level) if self._index.ntotal > 0 else 0,
            "metric": self._metric_store,
            "backend": "faiss.IndexHNSWFlat",
        }
