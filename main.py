"""VectorDB Engine FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from api.auth import get_document_store
from api.routes.auth import router as auth_router
from api.routes.documents import router as documents_router
from api.routes.health import router as health_router
from api.routes.search import get_index_manager, get_vector_store
from api.routes.search import router as search_router
from db.document_store import DocumentStore
from db.vector_store import VectorStore
from embeddings.encoder import SentenceEncoder
from embeddings.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VectorDB Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(health_router)

_FRONTEND_INDEX = Path(__file__).resolve().parent / "frontend" / "index.html"


@app.on_event("startup")
async def startup() -> None:
    """Initialize stores, models, and log readiness."""
    load_dotenv()
    db_path = os.getenv("DB_PATH", "./data/vectordb")

    vector_store: VectorStore = get_vector_store()
    document_store: DocumentStore = get_document_store()

    app.state.vector_store = vector_store
    app.state.document_store = document_store

    encoder = SentenceEncoder()
    encoder.encode("startup warmup")
    app.state.encoder = encoder

    ollama = OllamaClient()
    ollama_available = ollama.is_available()
    app.state.ollama = ollama
    logger.info("Ollama available: %s", ollama_available)

    get_index_manager()
    vector_count = int(vector_store.get_stats().get("count", 0))
    logger.info("Total vectors loaded: %s", vector_count)
    logger.info("Document store path: %s", document_store.db_path)
    logger.info("VectorDB Engine ready at http://localhost:8080")


@app.get("/")
def serve_frontend() -> HTMLResponse | PlainTextResponse:
    """Serve the frontend UI or a plain-text fallback."""
    if _FRONTEND_INDEX.is_file():
        return HTMLResponse(_FRONTEND_INDEX.read_text(encoding="utf-8"))
    return PlainTextResponse("VectorDB Engine Running")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
