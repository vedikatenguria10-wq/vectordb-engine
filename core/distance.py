"""Distance metrics for vector similarity search."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
MetricFn = Callable[[FloatArray, FloatArray], Union[float, FloatArray]]

_EPS = 1e-9


def cosine_similarity(a: FloatArray, b: FloatArray) -> Union[float, FloatArray]:
    """Return cosine distance (1 minus normalized dot product); lower means more similar."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.ndim == 1 and b_arr.ndim == 1:
        dot = float(np.dot(a_arr, b_arr))
        na = float(np.dot(a_arr, a_arr))
        nb = float(np.dot(b_arr, b_arr))
        if na < _EPS or nb < _EPS:
            return 1.0
        return 1.0 - dot / (np.sqrt(na) * np.sqrt(nb))

    if a_arr.ndim == 1:
        dot = b_arr @ a_arr
        na = float(np.dot(a_arr, a_arr))
        nb = np.sum(b_arr * b_arr, axis=1)
        out = np.ones(len(b_arr), dtype=np.float32)
        valid = (na >= _EPS) & (nb >= _EPS)
        out[valid] = 1.0 - dot[valid] / (np.sqrt(na) * np.sqrt(nb[valid]))
        return out

    if b_arr.ndim == 1:
        dot = a_arr @ b_arr
        na = np.sum(a_arr * a_arr, axis=1)
        nb = float(np.dot(b_arr, b_arr))
        out = np.ones(len(a_arr), dtype=np.float32)
        valid = (na >= _EPS) & (nb >= _EPS)
        out[valid] = 1.0 - dot[valid] / (np.sqrt(na[valid]) * np.sqrt(nb))
        return out

    dot = np.sum(a_arr * b_arr, axis=1)
    na = np.sum(a_arr * a_arr, axis=1)
    nb = np.sum(b_arr * b_arr, axis=1)
    out = np.ones(len(a_arr), dtype=np.float32)
    valid = (na >= _EPS) & (nb >= _EPS)
    out[valid] = 1.0 - dot[valid] / (np.sqrt(na[valid]) * np.sqrt(nb[valid]))
    return out


def euclidean(a: FloatArray, b: FloatArray) -> Union[float, FloatArray]:
    """Return Euclidean distance between one or many vector pairs."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.ndim == 1 and b_arr.ndim == 1:
        diff = a_arr - b_arr
        return float(np.sqrt(np.dot(diff, diff)))

    if a_arr.ndim == 1:
        diff = b_arr - a_arr
        return np.sqrt(np.sum(diff * diff, axis=1))

    if b_arr.ndim == 1:
        diff = a_arr - b_arr
        return np.sqrt(np.sum(diff * diff, axis=1))

    diff = a_arr - b_arr
    return np.sqrt(np.sum(diff * diff, axis=1))


def manhattan(a: FloatArray, b: FloatArray) -> Union[float, FloatArray]:
    """Return Manhattan (L1) distance between one or many vector pairs."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return float(np.sum(np.abs(a_arr - b_arr)))

    if a_arr.ndim == 1:
        return np.sum(np.abs(b_arr - a_arr), axis=1)

    if b_arr.ndim == 1:
        return np.sum(np.abs(a_arr - b_arr), axis=1)

    return np.sum(np.abs(a_arr - b_arr), axis=1)


def get_dist_fn(metric: str) -> MetricFn:
    """Return the distance function for a metric name, matching C++ getDistFn."""
    if metric == "cosine":
        return cosine_similarity
    if metric == "manhattan":
        return manhattan
    return euclidean
