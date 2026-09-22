from __future__ import annotations

import numpy as np

from .utils import bipolar_sigmoid, normalize_simplex


def centered_einstein_modulation(
    membership: np.ndarray,
    modulation: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Apply the centered Einstein modulation operator.

    membership: values in [0, 1]
    modulation: bipolar neighborhood effect in [-1, 1]
    strength: lambda, constrained to 0 <= lambda < 1
    """

    mu = np.clip(np.asarray(membership, dtype=float), 0.0, 1.0)
    m = np.clip(np.asarray(modulation, dtype=float), -1.0, 1.0)
    lam = float(np.clip(strength, 0.0, 1.0 - 1e-9))
    r = 2.0 * mu - 1.0
    d = lam * m
    denom = np.maximum(1.0 + r * d, 1e-12)
    out = 0.5 * (1.0 + (r + d) / denom)
    return np.clip(out, 0.0, 1.0)


def state_specific_neighbor_modulation(
    neighbor_memberships: np.ndarray,
    overlap_scores: np.ndarray,
    relation_matrix: np.ndarray,
) -> np.ndarray:
    """Compute M_{m,n,k} for all target states with state-specific Theta."""

    neigh = np.asarray(neighbor_memberships, dtype=float)
    theta = np.asarray(relation_matrix, dtype=float)
    if neigh.ndim != 2:
        raise ValueError("neighbor_memberships must have shape (neighbors, states)")
    num_states = theta.shape[1]
    if theta.shape != (neigh.shape[1] + 1, neigh.shape[1]):
        raise ValueError("relation_matrix must have shape (states + 1, states)")
    if neigh.shape[0] == 0:
        return np.zeros(num_states, dtype=float)
    overlaps = np.asarray(overlap_scores, dtype=float).reshape(-1, 1)
    if overlaps.shape[0] != neigh.shape[0]:
        raise ValueError("overlap_scores length must match number of neighbors")
    features = np.concatenate([neigh, np.clip(overlaps, 0.0, 1.0)], axis=1)
    return bipolar_sigmoid(features @ theta).mean(axis=0)


def apply_neighbor_modulation(
    base_membership: np.ndarray,
    neighbor_memberships: np.ndarray,
    overlap_scores: np.ndarray,
    relation_matrix: np.ndarray,
    strength: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    modulation = state_specific_neighbor_modulation(
        neighbor_memberships,
        overlap_scores,
        relation_matrix,
    )
    adjusted = centered_einstein_modulation(base_membership, modulation, strength)
    return normalize_simplex(adjusted) if normalize else adjusted
