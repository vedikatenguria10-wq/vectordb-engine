"""Local sentence-transformer encoding with a shared model instance."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBED_DIMS = 384

_model: Optional[Any] = None


def _get_model() -> Any:
    """Load and cache the sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class SentenceEncoder:
    """Encodes text into 384-dimensional vectors using a local MiniLM model."""

    def encode(self, text: str) -> FloatArray:
        """Encode a single string into a 1-D embedding vector."""
        emb = _get_model().encode(text, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def encode_batch(self, texts: list[str]) -> FloatArray:
        """Encode multiple strings into a 2-D array of shape (n, 384)."""
        if not texts:
            return np.zeros((0, _EMBED_DIMS), dtype=np.float32)
        embs = _get_model().encode(texts, convert_to_numpy=True)
        return np.asarray(embs, dtype=np.float32)

    def get_model_info(self) -> dict[str, Any]:
        """Return the model name and embedding dimensionality."""
        return {
            "model_name": _MODEL_NAME,
            "dimensions": _EMBED_DIMS,
        }
