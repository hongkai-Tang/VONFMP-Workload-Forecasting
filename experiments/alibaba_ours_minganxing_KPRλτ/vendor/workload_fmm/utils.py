from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def bipolar_sigmoid(x: np.ndarray) -> np.ndarray:
    """Bipolar sigmoid with range [-1, 1]."""

    return np.tanh(np.asarray(x, dtype=float) / 2.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def normalize_simplex(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clipped = np.maximum(values, 0.0)
    total = clipped.sum(axis=axis, keepdims=True)
    size = clipped.shape[axis]
    uniform = np.full_like(clipped, 1.0 / max(size, 1))
    return np.where(total > eps, clipped / np.maximum(total, eps), uniform)


def ensure_3d_resource_tensor(values: np.ndarray, resource_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError("resource tensor must have shape (workloads, time, resources)")
    if resource_dim is not None and arr.shape[-1] != resource_dim:
        raise ValueError(f"last dimension must contain {resource_dim} resource features")
    return arr


def default_mask(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if mask is None:
        return np.isfinite(arr).all(axis=-1)
    out = np.asarray(mask, dtype=bool)
    if out.shape != arr.shape[:2]:
        raise ValueError("mask must have shape (workloads, time)")
    return out


def time_encoding(index: int | np.ndarray, period: float = 24.0) -> np.ndarray:
    idx = np.asarray(index, dtype=float)
    phase = 2.0 * np.pi * idx / max(period, 1e-12)
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1)
