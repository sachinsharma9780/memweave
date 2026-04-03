"""
memweave/embedding/vectors.py — Embedding vector utilities.

Provides L2 normalization for embedding vectors.
Used by the embedding layer to normalize raw API output to unit length
before storing vectors in the SQLite index.

Vector similarity search is handled by sqlite-vec (via SQL ``vec_distance_cosine``),
not by this module.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def normalize_embedding(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length.

    After normalization, ``||v|| == 1.0``. Storing unit vectors means
    ``vec_distance_cosine`` in sqlite-vec gives correct cosine similarity
    without needing a separate normalization step at query time.

    If the vector is all zeros (zero norm), it is returned unchanged rather
    than raising a division-by-zero error — the caller must handle this edge
    case (typically by discarding the chunk).

    Args:
        vec: Raw embedding vector as a list of floats.

    Returns:
        New list of floats with the same length, ``||result|| ≈ 1.0``.
        Returns ``vec`` unchanged if the norm is zero.

    Example::

        v = normalize_embedding([3.0, 4.0])
        # → [0.6, 0.8]   (3/5, 4/5)
        assert abs(sum(x**2 for x in v) - 1.0) < 1e-9
    """
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return vec
    return cast(list[float], (arr / norm).tolist())
