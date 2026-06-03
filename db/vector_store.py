"""Persistent vector storage backed by LanceDB."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import lancedb
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

_TABLE_NAME = "vectors"


class VectorStore:
    """LanceDB-backed store for vectors and metadata that persists across restarts."""

    def __init__(self, db_path: str) -> None:
        """Connect to LanceDB at db_path and open or prepare the vectors table."""
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table: lancedb.table.LanceTable | None = None
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)

    def _ensure_table(self, vector: FloatArray) -> lancedb.table.LanceTable:
        """Open the table or create it using the dimension of the first vector."""
        if self._table is not None:
            return self._table
        record = self._make_record(
            id=0,
            vector=vector,
            text="",
            category="",
            created_at=datetime.now(timezone.utc),
        )
        self._table = self._db.create_table(_TABLE_NAME, data=[record])
        self._table.delete("id = 0")
        return self._table

    @staticmethod
    def _make_record(
        id: int,
        vector: FloatArray,
        text: str,
        category: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Build a LanceDB row dict from scalar fields and a numpy vector."""
        return {
            "id": id,
            "text": text,
            "category": category,
            "vector": np.asarray(vector, dtype=np.float32).tolist(),
            "created_at": created_at.isoformat(),
        }

    def insert(
        self,
        id: int,
        vector: FloatArray,
        text: str,
        category: str,
    ) -> None:
        """Save a vector and its metadata to LanceDB."""
        table = self._ensure_table(vector)
        record = self._make_record(
            id=id,
            vector=vector,
            text=text,
            category=category,
            created_at=datetime.now(timezone.utc),
        )
        table.add([record])

    def search(
        self, query_vector: FloatArray, k: int
    ) -> list[dict[str, Any]]:
        """Return the k nearest stored items ordered by vector distance."""
        if self._table is None:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        rows = (
            self._table.search(q)
            .limit(k)
            .to_list()
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": int(row["id"]),
                    "text": str(row["text"]),
                    "category": str(row["category"]),
                    "score": float(row.get("_distance", 0.0)),
                    "created_at": row.get("created_at"),
                }
            )
        return results

    def delete(self, id: int) -> None:
        """Remove the item with the given id from the table."""
        if self._table is None:
            return
        self._table.delete(f"id = {int(id)}")

    def list_all(self) -> list[dict[str, Any]]:
        """Return every stored item as a list of dictionaries."""
        if self._table is None:
            return []
        rows = self._table.to_arrow().to_pylist()
        return [
            {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "category": str(row["category"]),
                "vector": row["vector"],
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return row count and embedding dimensionality."""
        if self._table is None:
            return {"count": 0, "dims": 0, "db_path": self.db_path}

        count = self._table.count_rows()
        dims = 0
        if count > 0:
            schema = self._table.schema
            if "vector" in schema.names:
                vector_field = schema.field("vector")
                dims = int(vector_field.type.list_size)

        return {
            "count": count,
            "dims": dims,
            "db_path": self.db_path,
        }
