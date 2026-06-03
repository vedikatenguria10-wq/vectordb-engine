"""Service health and status routes."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter

from api.auth import get_document_store
from api.routes.search import get_vector_store
from embeddings.ollama_client import OllamaClient

load_dotenv()

router = APIRouter(tags=["health"])

_ollama = OllamaClient()


@router.get("/status")
def status() -> dict[str, Any]:
    """Report Ollama availability, data counts, and configured model names."""
    vector_stats = get_vector_store().get_stats()
    document_stats = get_document_store().get_stats()
    return {
        "ollama_available": _ollama.is_available(),
        "vector_count": int(vector_stats.get("count", 0)),
        "chunk_count": int(document_stats.get("chunk_count", 0)),
        "embed_model": os.getenv("EMBED_MODEL", _ollama.embed_model),
        "gen_model": os.getenv("MODEL_NAME", _ollama.gen_model),
    }
