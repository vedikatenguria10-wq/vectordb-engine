"""Custom HNSW (Hierarchical Navigable Small World) graph index."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from core.distance import MetricFn, get_dist_fn

FloatArray = NDArray[np.floating]


@dataclass
class _Node:
    """Graph vertex storing an embedding and per-layer neighbor lists."""

    id: int
    emb: FloatArray
    maxLyr: int
    nbrs: list[list[int]] = field(default_factory=list)


class HNSWCustom:
    """HNSW index ported from the C++ HNSW class in main.cpp."""

    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42) -> None:
        """Initialize graph parameters M, M0, ef_build, and level multiplier mL."""
        self.G: dict[int, _Node] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.topLayer = -1
        self.entryPt = -1
        self._rng = np.random.default_rng(seed)

    def randLevel(self) -> int:
        """Sample a random maximum layer for a new node."""
        u = float(self._rng.random())
        return int(math.floor(-math.log(u) * self.mL))

    def searchLayer(
        self,
        q: FloatArray,
        ep: int,
        ef: int,
        lyr: int,
        dist: MetricFn,
    ) -> list[tuple[float, int]]:
        """Greedy beam search on one layer, returning up to ef nearest node ids."""
        vis: dict[int, bool] = {}
        cands: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []

        q_arr = np.asarray(q, dtype=np.float32)
        d0 = float(dist(q_arr, self.G[ep].emb))
        vis[ep] = True
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if vis.get(nid) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = float(dist(q_arr, self.G[nid].emb))
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-neg_d, nid) for neg_d, nid in found]
        res.sort(key=lambda x: x[0])
        return res

    def selectNbrs(
        self, cands: list[tuple[float, int]], maxM: int
    ) -> list[int]:
        """Pick up to maxM closest neighbor ids from sorted candidates."""
        r: list[int] = []
        for i in range(min(len(cands), maxM)):
            r.append(cands[i][1])
        return r

    def insert(self, id: int, vector: FloatArray) -> None:
        """Insert a vector into the HNSW graph (cosine distance for link selection)."""
        dist = get_dist_fn("cosine")
        emb = np.asarray(vector, dtype=np.float32)
        lvl = self.randLevel()
        self.G[id] = _Node(id=id, emb=emb, maxLyr=lvl, nbrs=[[] for _ in range(lvl + 1)])

        if self.entryPt == -1:
            self.entryPt = id
            self.topLayer = lvl
            return

        ep = self.entryPt
        for lc in range(self.topLayer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.searchLayer(emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.topLayer, lvl), -1, -1):
            W = self.searchLayer(emb, ep, self.ef_build, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel = self.selectNbrs(W, maxM)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend(
                        [[] for _ in range(lc + 1 - len(self.G[nid].nbrs))]
                    )
                conn = self.G[nid].nbrs[lc]
                conn.append(id)
                if len(conn) > maxM:
                    embs = np.stack(
                        [self.G[c].emb for c in conn if c in self.G], axis=0
                    )
                    conn_ids = [c for c in conn if c in self.G]
                    ds_arr = dist(self.G[nid].emb, embs)
                    ds = sorted(zip(ds_arr.tolist(), conn_ids), key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [
                        c for _, c in ds[:maxM]
                    ]

            if W:
                ep = W[0][1]

        if lvl > self.topLayer:
            self.topLayer = lvl
            self.entryPt = id

    def search(
        self, query: FloatArray, k: int, metric: str = "cosine"
    ) -> list[tuple[float, int]]:
        """Search the graph for k nearest neighbors using the given metric."""
        ef = 50
        if self.entryPt == -1:
            return []

        dist = get_dist_fn(metric)
        q = np.asarray(query, dtype=np.float32)
        ep = self.entryPt

        for lc in range(self.topLayer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.searchLayer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        W = self.searchLayer(q, ep, max(ef, k), 0, dist)
        if len(W) > k:
            W = W[:k]
        return W

    def delete(self, id: int) -> None:
        """Remove a node and strip it from all neighbor lists."""
        if id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id in layer:
                    layer[:] = [x for x in layer if x != id]
        if self.entryPt == id:
            self.entryPt = -1
            for nid in self.G:
                if nid != id:
                    self.entryPt = nid
                    break
        del self.G[id]

    def get_info(self) -> dict[str, Any]:
        """Return graph structure metadata for visualization, matching C++ GraphInfo."""
        maxL = max(self.topLayer + 1, 1)
        nodesPerLayer = [0] * maxL
        edgesPerLayer = [0] * maxL
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, int]] = []

        for nid, nd in self.G.items():
            nodes.append(
                {
                    "id": nid,
                    "maxLyr": nd.maxLyr,
                }
            )
            for lc in range(min(nd.maxLyr, maxL - 1) + 1):
                nodesPerLayer[lc] += 1
                if lc < len(nd.nbrs):
                    for nb in nd.nbrs[lc]:
                        if nid < nb:
                            edgesPerLayer[lc] += 1
                            edges.append({"src": nid, "dst": nb, "lyr": lc})

        return {
            "topLayer": self.topLayer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodesPerLayer,
            "edgesPerLayer": edgesPerLayer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.G)
