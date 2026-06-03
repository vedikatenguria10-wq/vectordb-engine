"""Brute-force k-nearest neighbor search."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.distance import MetricFn, get_dist_fn

FloatArray = NDArray[np.floating]


class BruteForce:
    """Linear-scan index that compares the query against every stored vector."""

    def __init__(self) -> None:
        """Create an empty brute-force index."""
        self.items: dict[int, FloatArray] = {}

    def insert(self, id: int, vector: FloatArray) -> None:
        """Store a vector under the given id."""
        self.items[id] = np.asarray(vector, dtype=np.float32)

    def search(
        self, query: FloatArray, k: int, metric: str = "euclidean"
    ) -> list[tuple[float, int]]:
        """Return up to k nearest neighbors as (distance, id) pairs sorted by distance."""
        if not self.items:
            return []

        dist: MetricFn = get_dist_fn(metric)
        q = np.asarray(query, dtype=np.float32)
        ids = list(self.items.keys())
        embs = np.stack([self.items[i] for i in ids], axis=0)
        dists = np.asarray(dist(q, embs), dtype=np.float32)

        r = sorted(zip(dists.tolist(), ids), key=lambda x: x[0])
        if len(r) > k:
            r = r[:k]
        return r

    def delete(self, id: int) -> None:
        """Remove the vector with the given id if it exists."""
        self.items.pop(id, None)

    def get_stats(self) -> dict[str, Any]:
        """Return basic index statistics."""
        dims = 0
        if self.items:
            dims = int(next(iter(self.items.values())).shape[0])
        return {
            "count": len(self.items),
            "dims": dims,
            "algorithm": "bruteforce",
        }
