from __future__ import annotations

import numpy as np


def normalize_deployment(deployment: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Column-normalize H_{v,n} so each workload distributes over its nodes."""

    h = np.asarray(deployment, dtype=float)
    if h.ndim != 2:
        raise ValueError("deployment must have shape (nodes, workloads)")
    col_sum = h.sum(axis=0, keepdims=True)
    return np.divide(h, np.maximum(col_sum, eps), out=np.zeros_like(h), where=col_sum > eps)


def hyperedge_intersections(deployment: np.ndarray, normalize: bool = True) -> np.ndarray:
    h = (np.asarray(deployment) > 0).astype(float)
    raw = h.T @ h
    np.fill_diagonal(raw, 0.0)
    if not normalize:
        return raw
    max_val = float(np.max(raw))
    if max_val <= 0.0:
        return raw
    return raw / max_val


def active_neighbor_indices(
    deployment: np.ndarray,
    active_mask: np.ndarray,
    workload_index: int,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    c = hyperedge_intersections(deployment, normalize=True)
    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 2:
        raise ValueError("active_mask must have shape (workloads, time)")
    neighbors = np.flatnonzero((c[workload_index] > 0.0) & active[:, time_index])
    neighbors = neighbors[neighbors != workload_index]
    return neighbors, c[workload_index, neighbors]


def deployment_from_pairs(
    pairs: list[tuple[int, int]],
    num_nodes: int | None = None,
    num_workloads: int | None = None,
) -> np.ndarray:
    if not pairs:
        raise ValueError("pairs must not be empty")
    max_node = max(v for v, _ in pairs)
    max_workload = max(w for _, w in pairs)
    v_count = max_node + 1 if num_nodes is None else num_nodes
    w_count = max_workload + 1 if num_workloads is None else num_workloads
    h = np.zeros((v_count, w_count), dtype=float)
    for node_id, workload_id in pairs:
        h[int(node_id), int(workload_id)] = 1.0
    return h
