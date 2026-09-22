from __future__ import annotations

from functools import lru_cache

import numpy as np


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = None,
    window: int | None = None,
    normalize: bool = True,
) -> float:
    """Multivariate DTW distance using weighted squared Euclidean local cost."""

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("DTW inputs must have shape (length, features)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("DTW inputs must have the same feature dimension")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("inf")

    w = np.ones(x.shape[1], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (x.shape[1],):
        raise ValueError("weights must have shape (features,)")

    n, m = x.shape[0], y.shape[0]
    band = max(n, m) if window is None else int(window)
    band = max(band, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    plen = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band) + 1
        for j in range(j_start, j_end):
            diff = (x[i - 1] - y[j - 1]) * w
            local = float(np.dot(diff, diff))
            choices = (
                (cost[i - 1, j], plen[i - 1, j]),
                (cost[i, j - 1], plen[i, j - 1]),
                (cost[i - 1, j - 1], plen[i - 1, j - 1]),
            )
            prev_cost, prev_len = min(choices, key=lambda item: item[0])
            cost[i, j] = local + prev_cost
            plen[i, j] = prev_len + 1

    total = cost[n, m]
    if not np.isfinite(total):
        return float("inf")
    if normalize:
        total = total / max(int(plen[n, m]), 1)
    return float(np.sqrt(max(total, 0.0)))


def pairwise_dtw(
    segments: list[np.ndarray],
    weights: np.ndarray | None = None,
    window: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    q = len(segments)
    out = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(i + 1, q):
            d = dtw_distance(segments[i], segments[j], weights, window, normalize)
            out[i, j] = out[j, i] = d
    return out


def cached_dtw_factory(
    segments: list[np.ndarray],
    weights: np.ndarray | None = None,
    window: int | None = None,
    normalize: bool = True,
):
    @lru_cache(maxsize=None)
    def distance(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return dtw_distance(segments[i], segments[j], weights, window, normalize)

    return distance
