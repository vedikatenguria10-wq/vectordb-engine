"""HTTP client for Ollama embeddings and generation with local fallback."""

from __future__ import annotations

import json
import os
from typing import Optional
from urllib.parse import urlparse

import httpx
import numpy as np
from numpy.typing import NDArray

from embeddings.encoder import SentenceEncoder

FloatArray = NDArray[np.floating]

_DEFAULT_BASE_URL = "http://localhost:11434"
_CONNECT_TIMEOUT = 2.0
_EMBED_TIMEOUT = 30.0
_GENERATE_TIMEOUT = 180.0


class OllamaClient:
    """Calls Ollama REST APIs; falls back to SentenceEncoder when Ollama is down."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        """Configure the Ollama base URL from the argument or OLLAMA_URL env."""
        raw = base_url or os.getenv("OLLAMA_URL", _DEFAULT_BASE_URL)
        self.base_url = raw.rstrip("/")
        self.embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.gen_model = os.getenv("MODEL_NAME", "llama3.2")
        self._encoder = SentenceEncoder()
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Return True if Ollama responds on /api/tags, False otherwise."""
        try:
            with httpx.Client(timeout=_CONNECT_TIMEOUT) as client:
                res = client.get(f"{self.base_url}/api/tags")
                ok = res.status_code == 200
        except (httpx.HTTPError, OSError):
            ok = False
        self._available = ok
        return ok

    def embed(self, text: str) -> FloatArray:
        """Return an embedding from Ollama, or from SentenceEncoder if unavailable."""
        if self._available is False:
            return self._encoder.encode(text)
        if self._available is None and not self.is_available():
            return self._encoder.encode(text)

        vec = self._ollama_embed(text)
        if vec is None:
            return self._encoder.encode(text)
        return vec

    def _ollama_embed(self, text: str) -> Optional[FloatArray]:
        """Call Ollama /api/embeddings and parse the embedding vector."""
        try:
            with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
                res = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
            if res.status_code != 200:
                return None
            data = res.json()
            embedding = data.get("embedding")
            if not embedding:
                return None
            return np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (httpx.HTTPError, OSError, json.JSONDecodeError, ValueError):
            return None

    def generate(self, prompt: str, context: str) -> str:
        """Generate text from Ollama using optional context prepended to the prompt."""
        full_prompt = prompt
        if context.strip():
            full_prompt = f"Context:\n{context}\n\n{prompt}"

        try:
            with httpx.Client(timeout=_GENERATE_TIMEOUT) as client:
                res = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.gen_model,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                )
            if res.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            data = res.json()
            return str(data.get("response", ""))
        except (httpx.HTTPError, OSError, json.JSONDecodeError):
            return "ERROR: Ollama unavailable. Run: ollama serve"
