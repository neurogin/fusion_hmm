"""Matrix and parcel-summary utilities shared across the public Stage-6 backends.

The Stage-6 reconstruction modules repeatedly need to:
- rebuild parcel-by-parcel matrices from saved upper-triangle vectors
- compute nodal mean connectivity summaries
- derive stable plotting limits from many arrays

This file keeps that logic small, named, and easy to reuse.
"""

from __future__ import annotations

import math

import numpy as np


def infer_ut_matrix_size(length: int) -> int:
    """Infer the square matrix size `n` from an upper-triangle vector length."""
    n = int((1 + math.sqrt(1 + 8 * length)) / 2)
    if n * (n - 1) // 2 != length:
        raise ValueError(f"Upper-triangle length {length} does not match any n x n matrix.")
    return n


def ut_to_square(vec: np.ndarray, fill_diag: float = 1.0) -> np.ndarray:
    """Rebuild a symmetric square matrix from its upper triangle."""
    vec = np.asarray(vec)
    n = infer_ut_matrix_size(vec.size)
    matrix = np.zeros((n, n), dtype=np.float32)
    iu = np.triu_indices(n, 1)
    matrix[iu] = vec
    matrix[(iu[1], iu[0])] = vec
    np.fill_diagonal(matrix, fill_diag)
    return matrix


def square_to_ut(matrix: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle of a square matrix."""
    iu = np.triu_indices(matrix.shape[0], 1)
    return np.asarray(matrix)[iu]


def corr_from_cov(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix."""
    diag = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    return (covariance / diag[:, None]) / diag[None, :]


def compute_nodal_mean_r(matrix: np.ndarray) -> np.ndarray:
    """Average each parcel's correlations after masking the diagonal."""
    tmp = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(tmp, np.nan)
    return np.nanmean(tmp, axis=1)


def compute_symmetric_limits(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    quantile: float = 0.98,
    floor: float = 1e-8,
) -> float:
    """Choose a stable symmetric plotting limit from one or more arrays."""
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(np.abs(arr))
    if not vals:
        return float(max(floor, 1.0))
    vmax = np.quantile(np.concatenate(vals), quantile)
    return float(max(vmax, floor))
