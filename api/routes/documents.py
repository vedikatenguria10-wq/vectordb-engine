"""Document ingestion, listing, deletion, and RAG ask routes."""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import get_current_user, get_document_store
from api.routes.search import get_vector_store, invalidate_search_indexes
from db.document_store import DocumentStore
from db.schema import DocumentChunk, User
from embeddings.chunker import chunk_text
from embeddings.encoder import SentenceEncoder
from embeddings.ollama_client import OllamaClient

load_dotenv()

router = APIRouter(prefix="/doc", tags=["documents"])

_encoder = SentenceEncoder()
_ollama = OllamaClient()


def _chunk_to_vector_id(chunk_id: str) -> int:
    """Map a chunk id string to a stable integer id for VectorStore."""
    return abs(hash(chunk_id)) % (2**31 - 1)


class InsertRequest(BaseModel):
    """Document insert payload."""

    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class InsertResponse(BaseModel):
    """Number of chunks stored for a document."""

    chunks_inserted: int


class AskRequest(BaseModel):
    """RAG question payload."""

    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1)


class AskResponse(BaseModel):
    """Generated answer plus retrieved context chunks."""

    answer: str
    context_chunks: list[dict[str, Any]]


@router.post("/insert", response_model=InsertResponse)
def insert_document(
    body: InsertRequest,
    _: User = Depends(get_current_user),
    doc_store: DocumentStore = Depends(get_document_store),
) -> InsertResponse:
    """Chunk a document, embed each chunk, and persist to both stores."""
    vector_store = get_vector_store()
    doc_uuid = str(uuid.uuid4())
    chunks = chunk_text(body.text)
    inserted = 0

    for chunk_str, chunk_index in chunks:
        chunk_id = f"{doc_uuid}_{chunk_index}"
        vector_id = _chunk_to_vector_id(chunk_id)
        vector = _encoder.encode(chunk_str)

        vector_store.insert(
            vector_id,
            vector,
            chunk_str,
            "doc",
        )
        doc_store.insert_chunk(
            DocumentChunk(
                id=chunk_id,
                title=body.title,
                chunk_text=chunk_str,
                chunk_index=chunk_index,
                embedding_id=str(vector_id),
            )
        )
        inserted += 1

    invalidate_search_indexes()
    return InsertResponse(chunks_inserted=inserted)


@router.get("/list")
def list_documents(
    doc_store: DocumentStore = Depends(get_document_store),
) -> list[dict[str, Any]]:
    """Return all document chunks (public)."""
    return [chunk.model_dump(mode="json") for chunk in doc_store.get_all_chunks()]


@router.delete("/delete/{id}")
def delete_document(
    id: str,
    _: User = Depends(get_current_user),
    doc_store: DocumentStore = Depends(get_document_store),
) -> dict[str, Any]:
    """Delete a chunk from DocumentStore and its vector from VectorStore."""
    chunks = doc_store.get_all_chunks()
    target = next((c for c in chunks if c.id == id), None)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )

    vector_store = get_vector_store()
    try:
        vector_id = int(target.embedding_id)
    except ValueError:
        vector_id = _chunk_to_vector_id(id)

    vector_store.delete(vector_id)
    doc_store.delete_chunk(id)
    invalidate_search_indexes()
    return {"ok": True, "id": id}


@router.post("/ask", response_model=AskResponse)
def ask_document(
    body: AskRequest,
    _: User = Depends(get_current_user),
) -> AskResponse:
    """Retrieve top-k chunks and generate an answer with Ollama."""
    vector_store = get_vector_store()
    query_vector = _encoder.encode(body.question)
    hits = vector_store.search(query_vector, body.k)

    context_chunks: list[dict[str, Any]] = []
    context_parts: list[str] = []
    for hit in hits:
        chunk = {
            "id": hit["id"],
            "text": hit["text"],
            "category": hit["category"],
            "score": hit.get("score", 0.0),
        }
        context_chunks.append(chunk)
        context_parts.append(str(hit["text"]))

    context = "\n\n".join(context_parts)
    answer = _ollama.generate(body.question, context)
    return AskResponse(answer=answer, context_chunks=context_chunks)
