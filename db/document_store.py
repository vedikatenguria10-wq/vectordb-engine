"""SQLite-backed storage for document chunks and users."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Optional

from db.schema import DocumentChunk, User

_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding_id TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class DocumentStore:
    """SQLite database for document chunks and user accounts."""

    def __init__(self, db_path: str) -> None:
        """Create the SQLite file and tables if they do not already exist."""
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_CHUNKS_TABLE + _USERS_TABLE)

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        """Parse an ISO-8601 timestamp stored in SQLite."""
        return datetime.fromisoformat(value)

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> DocumentChunk:
        """Convert a SQLite row into a DocumentChunk model."""
        return DocumentChunk(
            id=str(row["id"]),
            title=str(row["title"]),
            chunk_text=str(row["chunk_text"]),
            chunk_index=int(row["chunk_index"]),
            embedding_id=str(row["embedding_id"]),
            created_at=DocumentStore._parse_dt(str(row["created_at"])),
        )

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> User:
        """Convert a SQLite row into a User model."""
        return User(
            id=str(row["id"]),
            username=str(row["username"]),
            hashed_password=str(row["hashed_password"]),
            created_at=DocumentStore._parse_dt(str(row["created_at"])),
        )

    def insert_chunk(self, chunk: DocumentChunk) -> None:
        """Insert a document chunk row."""
        created_at = chunk.created_at.isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO document_chunks
                    (id, title, chunk_text, chunk_index, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.title,
                    chunk.chunk_text,
                    chunk.chunk_index,
                    chunk.embedding_id,
                    created_at,
                ),
            )

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all document chunks ordered by title and chunk index."""
        cur = self._conn.execute(
            """
            SELECT id, title, chunk_text, chunk_index, embedding_id, created_at
            FROM document_chunks
            ORDER BY title, chunk_index
            """
        )
        return [self._chunk_from_row(row) for row in cur.fetchall()]

    def delete_chunk(self, id: str) -> None:
        """Delete a document chunk by its id."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM document_chunks WHERE id = ?",
                (id,),
            )

    def insert_user(self, user: User) -> None:
        """Insert a new user row."""
        created_at = user.created_at.isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO users (id, username, hashed_password, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user.id, user.username, user.hashed_password, created_at),
            )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Look up a user by username, or return None if not found."""
        cur = self._conn.execute(
            """
            SELECT id, username, hashed_password, created_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._user_from_row(row)

    def get_stats(self) -> dict[str, Any]:
        """Return counts of stored chunks and users."""
        chunk_cur = self._conn.execute("SELECT COUNT(*) FROM document_chunks")
        user_cur = self._conn.execute("SELECT COUNT(*) FROM users")
        chunk_count = int(chunk_cur.fetchone()[0])
        user_count = int(user_cur.fetchone()[0])
        return {
            "chunk_count": chunk_count,
            "user_count": user_count,
            "db_path": self.db_path,
        }
