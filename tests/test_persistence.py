"""Persistence tests for VectorStore and DocumentStore."""

from __future__ import annotations

import os

import numpy as np

from db.document_store import DocumentStore
from db.schema import User
from db.vector_store import VectorStore


def test_vector_survives_restart(temp_db_path: str) -> None:
    """Vectors written to disk should be visible after reopening the store."""
    vec = np.random.default_rng(1).random(384, dtype=np.float32)
    vector_id = 42

    store1 = VectorStore(temp_db_path)
    store1.insert(vector_id, vec, "persistent text", "test")

    store2 = VectorStore(temp_db_path)
    results = store2.search(vec, 1)
    assert results
    assert int(results[0]["id"]) == vector_id


def test_delete_survives_restart(temp_db_path: str) -> None:
    """Deleted vectors should remain absent after reopening the store."""
    rng = np.random.default_rng(2)
    vec = rng.random(384, dtype=np.float32)
    keep_id = 1
    delete_id = 2

    store1 = VectorStore(temp_db_path)
    store1.insert(keep_id, vec, "keep me", "test")
    store1.insert(delete_id, rng.random(384, dtype=np.float32), "remove me", "test")
    store1.delete(delete_id)

    store2 = VectorStore(temp_db_path)
    results = store2.search(vec, 10)
    returned_ids = {int(hit["id"]) for hit in results}
    assert delete_id not in returned_ids
    assert keep_id in returned_ids


def test_user_survives_restart(temp_db_path: str) -> None:
    """Users stored in SQLite should be retrievable from a new connection."""
    sqlite_path = os.path.join(temp_db_path, "documents.db")
    username = "persist_user"

    store1 = DocumentStore(sqlite_path)
    user = User(username=username, hashed_password="hashed_value")
    store1.insert_user(user)

    store2 = DocumentStore(sqlite_path)
    found = store2.get_user_by_username(username)
    assert found is not None
    assert found.username == username
    assert found.id == user.id
