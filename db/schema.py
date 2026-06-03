"""Pydantic models for vector items, document chunks, users, and search results."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class VectorItem(BaseModel):
    """A categorized text item stored with a vector embedding."""

    id: int
    text: str
    category: str
    created_at: datetime = Field(default_factory=_utc_now)


class DocumentChunk(BaseModel):
    """One chunk of a document linked to an embedding record."""

    id: str
    title: str
    chunk_text: str
    chunk_index: int
    embedding_id: str
    created_at: datetime = Field(default_factory=_utc_now)


class User(BaseModel):
    """An authenticated user account."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    hashed_password: str
    created_at: datetime = Field(default_factory=_utc_now)


class SearchResult(BaseModel):
    """A single hit returned from a vector search."""

    id: int
    text: str
    category: str
    score: float
    algorithm: str
