"""KD-tree approximate nearest neighbor search."""

from __future__ import annotations

import heapq
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from core.distance import MetricFn, get_dist_fn

FloatArray = NDArray[np.floating]


class _KDNode:
    """Internal tree node holding one point and child links."""

    def __init__(self, id: int, emb: FloatArray) -> None:
        """Attach an id and embedding at this node."""
        self.id = id
        self.emb = np.asarray(emb, dtype=np.float32)
        self.left: Optional[_KDNode] = None
        self.right: Optional[_KDNode] = None


class KDTree:
    """KD-tree index for k-nearest neighbor queries."""

    def __init__(self, dims: int) -> None:
        """Create an empty tree with the given embedding dimensionality."""
        self.root: Optional[_KDNode] = None
        self.dims = dims
        self._count = 0

    def _destroy(self, n: Optional[_KDNode]) -> None:
        """Recursively free the subtree rooted at n."""
        if n is None:
            return
        self._destroy(n.left)
        self._destroy(n.right)

    def _ins(self, n: Optional[_KDNode], id: int, emb: FloatArray, d: int) -> _KDNode:
        """Insert a point into the subtree, alternating split axes."""
        if n is None:
            return _KDNode(id, emb)
        ax = d % self.dims
        v_emb = np.asarray(emb, dtype=np.float32)
        if v_emb[ax] < n.emb[ax]:
            n.left = self._ins(n.left, id, v_emb, d + 1)
        else:
            n.right = self._ins(n.right, id, v_emb, d + 1)
        return n

    def _knn(
        self,
        n: Optional[_KDNode],
        q: FloatArray,
        k: int,
        d: int,
        dist: MetricFn,
        heap: list[tuple[float, int]],
    ) -> None:
        """Recursively collect up to k best candidates in a max-heap by distance."""
        if n is None:
            return

        dn = float(dist(q, n.emb))
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax = d % self.dims
        diff = float(q[ax] - n.emb[ax])
        if diff < 0:
            closer, farther = n.left, n.right
        else:
            closer, farther = n.right, n.left

        self._knn(closer, q, k, d + 1, dist, heap)
        worst = -heap[0][0] if heap else float("inf")
        if len(heap) < k or abs(diff) < worst:
            self._knn(farther, q, k, d + 1, dist, heap)

    def insert(self, id: int, vector: FloatArray) -> None:
        """Insert a vector into the tree."""
        self.root = self._ins(self.root, id, vector, 0)
        self._count += 1

    def search(
        self, query: FloatArray, k: int, metric: str = "euclidean"
    ) -> list[tuple[float, int]]:
        """Return up to k nearest neighbors as (distance, id) pairs sorted by distance."""
        dist = get_dist_fn(metric)
        q = np.asarray(query, dtype=np.float32)
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)

        r = [(-neg_d, nid) for neg_d, nid in heap]
        r.sort(key=lambda x: x[0])
        return r

    def delete(self, id: int) -> None:
        """Remove a point by rebuilding the tree without that id."""
        if self.root is None:
            return

        items: list[tuple[int, FloatArray]] = []

        def collect(n: Optional[_KDNode]) -> None:
            if n is None:
                return
            if n.id != id:
                items.append((n.id, n.emb.copy()))
            collect(n.left)
            collect(n.right)

        collect(self.root)
        self._destroy(self.root)
        self.root = None
        self._count = 0
        for i, emb in items:
            self.insert(i, emb)

    def rebuild(self, items: list[tuple[int, FloatArray]]) -> None:
        """Rebuild the entire tree from a list of (id, vector) pairs."""
        self._destroy(self.root)
        self.root = None
        self._count = 0
        for id, emb in items:
            self.insert(id, emb)

    def get_stats(self) -> dict[str, Any]:
        """Return basic index statistics."""
        return {
            "count": self._count,
            "dims": self.dims,
            "algorithm": "kdtree",
        }
