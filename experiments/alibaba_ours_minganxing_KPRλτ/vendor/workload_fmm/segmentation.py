from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import default_mask, ensure_3d_resource_tensor


@dataclass
class SegmentationConfig:
    min_length: int = 3
    max_length: int = 24
    split_penalty: float = 1.0
    trend_weight: float = 0.2


def segment_cost(segment: np.ndarray, trend_weight: float = 0.2) -> float:
    seg = np.asarray(segment, dtype=float)
    if seg.ndim != 2 or seg.shape[0] == 0:
        return float("inf")
    centered = seg - seg.mean(axis=0, keepdims=True)
    sse = float(np.sum(centered * centered))
    if seg.shape[0] == 1:
        trend = 0.0
    else:
        trend_vec = (seg[-1] - seg[0]) / max(seg.shape[0] - 1, 1)
        trend = float(np.dot(trend_vec, trend_vec))
    return sse + trend_weight * trend


def _active_runs(seq: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            runs.append(seq[start:i])
            start = None
    if start is not None:
        runs.append(seq[start:])
    return [run for run in runs if len(run) > 0]


def extract_stable_segments(
    sequence: np.ndarray,
    mask: np.ndarray | None = None,
    config: SegmentationConfig | None = None,
) -> list[np.ndarray]:
    cfg = config or SegmentationConfig()
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 2:
        raise ValueError("sequence must have shape (time, features)")
    active = np.ones(seq.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if active.shape != (seq.shape[0],):
        raise ValueError("mask must have shape (time,)")

    segments: list[np.ndarray] = []
    for run in _active_runs(seq, active):
        t_len = run.shape[0]
        if t_len < cfg.min_length:
            segments.append(run.copy())
            continue
        max_len = max(cfg.min_length, min(cfg.max_length, t_len))
        dp = np.full(t_len + 1, np.inf, dtype=float)
        prev = np.full(t_len + 1, -1, dtype=int)
        dp[0] = 0.0
        for end in range(1, t_len + 1):
            for length in range(cfg.min_length, min(max_len, end) + 1):
                start = end - length
                cost = dp[start] + segment_cost(run[start:end], cfg.trend_weight) + cfg.split_penalty
                if cost < dp[end]:
                    dp[end] = cost
                    prev[end] = start
            if prev[end] == -1 and end <= max_len:
                dp[end] = segment_cost(run[:end], cfg.trend_weight) + cfg.split_penalty
                prev[end] = 0
        end = t_len
        pieces: list[np.ndarray] = []
        while end > 0 and prev[end] >= 0:
            start = int(prev[end])
            pieces.append(run[start:end].copy())
            end = start
        if not pieces:
            pieces.append(run.copy())
        segments.extend(reversed(pieces))
    return segments


def extract_candidate_segments(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    config: SegmentationConfig | None = None,
) -> list[np.ndarray]:
    arr = ensure_3d_resource_tensor(values)
    active = default_mask(arr, mask)
    segments: list[np.ndarray] = []
    for n in range(arr.shape[0]):
        segments.extend(extract_stable_segments(arr[n], active[n], config))
    return segments
