"""Shared pytest fixtures for the VectorDB Engine test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.document_store import DocumentStore
from db.vector_store import VectorStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Return a temporary directory path string for isolated databases."""
    db_dir = tmp_path / "vectordb"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)


@pytest.fixture
def vector_store(temp_db_path: str) -> VectorStore:
    """Return a fresh LanceDB vector store in a temporary directory."""
    return VectorStore(temp_db_path)


@pytest.fixture
def document_store(temp_db_path: str) -> DocumentStore:
    """Return a fresh SQLite document store in a temporary directory."""
    sqlite_path = os.path.join(temp_db_path, "documents.db")
    return DocumentStore(sqlite_path)


@pytest.fixture
def sample_vectors() -> list[np.ndarray]:
    """Return ten random 384-dimensional embedding vectors."""
    rng = np.random.default_rng(42)
    return [rng.random(384, dtype=np.float32) for _ in range(10)]


@pytest.fixture
def test_client(temp_db_path: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a FastAPI TestClient wired to temporary database paths."""
    monkeypatch.setenv("DB_PATH", temp_db_path)

    from api.auth import _document_store

    _document_store.cache_clear()

    from api.routes import search as search_module

    search_module.get_vector_store.cache_clear()
    search_module._manager.vector_count = -1

    from main import app

    with TestClient(app) as client:
        yield client
