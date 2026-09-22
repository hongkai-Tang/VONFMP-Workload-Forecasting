from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtw import cached_dtw_factory, dtw_distance
from .segmentation import SegmentationConfig, extract_candidate_segments
from .types import ShapePrototype, ShapeSet
from .utils import default_mask, ensure_3d_resource_tensor, normalize_simplex


@dataclass
class ShapeSetConfig:
    min_length: int = 3
    max_length: int = 24
    split_penalty: float = 1.0
    trend_weight: float = 0.2
    coverage_threshold: float = 0.9
    coverage_radius: float | None = None
    max_shapes: int = 8
    kmedoids_iter: int = 10
    dtw_window: int | None = None
    resource_weights: tuple[float, ...] | None = None
    random_state: int = 7
    max_candidates: int = 300

    def segmentation_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            min_length=self.min_length,
            max_length=self.max_length,
            split_penalty=self.split_penalty,
            trend_weight=self.trend_weight,
        )


def _estimate_radius(
    candidates: list[np.ndarray],
    weights: np.ndarray,
    window: int | None,
    random_state: int,
    sample_size: int = 120,
) -> float:
    if len(candidates) <= 1:
        return 1.0
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(candidates))
    if len(idx) > sample_size:
        idx = rng.choice(idx, size=sample_size, replace=False)
    distances = []
    for pos, i in enumerate(idx):
        best = float("inf")
        for j in idx:
            if i == j:
                continue
            best = min(best, dtw_distance(candidates[i], candidates[j], weights, window))
        if np.isfinite(best):
            distances.append(best)
    if not distances:
        return 1.0
    return float(max(np.quantile(distances, 0.75), 1e-3))


def _initial_medoids(
    candidates: list[np.ndarray],
    k: int,
    distance,
    random_state: int,
) -> list[int]:
    rng = np.random.default_rng(random_state)
    first = int(rng.integers(0, len(candidates)))
    medoids = [first]
    while len(medoids) < k:
        best_i = None
        best_d = -1.0
        for i in range(len(candidates)):
            if i in medoids:
                continue
            d = min(distance(i, m) for m in medoids)
            if d > best_d:
                best_d = d
                best_i = i
        if best_i is None:
            break
        medoids.append(int(best_i))
    return medoids


def _assign(candidates: list[np.ndarray], medoids: list[int], distance) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(candidates), dtype=int)
    nearest = np.zeros(len(candidates), dtype=float)
    for i in range(len(candidates)):
        dists = np.asarray([distance(i, m) for m in medoids], dtype=float)
        labels[i] = int(np.argmin(dists))
        nearest[i] = float(dists[labels[i]])
    return labels, nearest


def _kmedoids(
    candidates: list[np.ndarray],
    k: int,
    weights: np.ndarray,
    window: int | None,
    max_iter: int,
    random_state: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    if k >= len(candidates):
        medoids = list(range(len(candidates)))
        labels = np.arange(len(candidates), dtype=int)
        nearest = np.zeros(len(candidates), dtype=float)
        return medoids, labels, nearest

    distance = cached_dtw_factory(candidates, weights, window)
    medoids = _initial_medoids(candidates, k, distance, random_state)
    for _ in range(max_iter):
        labels, nearest = _assign(candidates, medoids, distance)
        new_medoids = medoids.copy()
        for cluster_id in range(len(medoids)):
            members = np.flatnonzero(labels == cluster_id)
            if len(members) == 0:
                farthest = int(np.argmax(nearest))
                new_medoids[cluster_id] = farthest
                continue
            best_member = int(members[0])
            best_sum = float("inf")
            for candidate_idx in members:
                total = sum(distance(int(candidate_idx), int(other)) for other in members)
                if total < best_sum:
                    best_sum = total
                    best_member = int(candidate_idx)
            new_medoids[cluster_id] = best_member
        if new_medoids == medoids:
            break
        medoids = new_medoids
    labels, nearest = _assign(candidates, medoids, distance)
    return medoids, labels, nearest


def build_shape_set(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    config: ShapeSetConfig | None = None,
) -> ShapeSet:
    cfg = config or ShapeSetConfig()
    arr = ensure_3d_resource_tensor(values)
    active = default_mask(arr, mask)
    candidates = extract_candidate_segments(arr, active, cfg.segmentation_config())
    candidates = [seg for seg in candidates if seg.shape[0] > 0]
    if not candidates:
        raise ValueError("no candidate stable segments were extracted")

    rng = np.random.default_rng(cfg.random_state)
    if len(candidates) > cfg.max_candidates:
        keep = rng.choice(np.arange(len(candidates)), size=cfg.max_candidates, replace=False)
        candidates = [candidates[int(i)] for i in keep]

    weights = np.ones(arr.shape[-1], dtype=float) if cfg.resource_weights is None else np.asarray(cfg.resource_weights, dtype=float)
    if weights.shape != (arr.shape[-1],):
        raise ValueError("resource_weights length must match resource dimension")
    epsilon = cfg.coverage_radius
    if epsilon is None:
        epsilon = _estimate_radius(candidates, weights, cfg.dtw_window, cfg.random_state)

    best_result = None
    max_k = max(1, min(cfg.max_shapes, len(candidates)))
    for k in range(1, max_k + 1):
        medoids, labels, nearest = _kmedoids(
            candidates,
            k,
            weights,
            cfg.dtw_window,
            cfg.kmedoids_iter,
            cfg.random_state + k,
        )
        coverage = float(np.mean(nearest <= epsilon))
        best_result = (k, medoids, labels, nearest, coverage)
        if coverage >= cfg.coverage_threshold:
            break

    assert best_result is not None
    _, medoids, labels, nearest, coverage = best_result
    prototypes: list[ShapePrototype] = []
    for state_id, medoid_idx in enumerate(medoids):
        members = np.flatnonzero(labels == state_id)
        cluster_dist = nearest[members] if len(members) else np.asarray([epsilon], dtype=float)
        sigma = float(max(np.median(cluster_dist), np.std(cluster_dist), 1e-3))
        prototypes.append(
            ShapePrototype(
                id=state_id,
                values=np.array(candidates[medoid_idx], dtype=float, copy=True),
                sigma=sigma,
                medoid_source_index=int(medoid_idx),
                coverage_radius=float(epsilon),
            )
        )
    return ShapeSet(
        prototypes=prototypes,
        coverage=float(coverage),
        epsilon=float(epsilon),
        metadata={"candidate_count": len(candidates), "config": cfg.__dict__.copy()},
    )


def membership_for_observation(
    sequence_prefix: np.ndarray,
    shape_set: ShapeSet,
    weights: np.ndarray | None = None,
    dtw_window: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    if shape_set.num_states == 0:
        raise ValueError("shape_set contains no prototypes")
    seq = np.asarray(sequence_prefix, dtype=float)
    weights_arr = np.ones(seq.shape[-1], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    scores = []
    for prototype in shape_set.prototypes:
        length = prototype.length
        obs = seq[-length:] if seq.shape[0] >= length else seq
        dist = dtw_distance(obs, prototype.values, weights_arr, dtw_window)
        sigma = max(float(prototype.sigma), 1e-6)
        scores.append(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))
    raw = np.asarray(scores, dtype=float)
    if normalize:
        return normalize_simplex(raw)
    return raw


def compute_memberships(
    values: np.ndarray,
    shape_set: ShapeSet,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    dtw_window: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    arr = ensure_3d_resource_tensor(values)
    active = default_mask(arr, mask)
    out = np.zeros((arr.shape[0], arr.shape[1], shape_set.num_states), dtype=float)
    for n in range(arr.shape[0]):
        active_indices = np.flatnonzero(active[n])
        for t in active_indices:
            prefix_idx = active_indices[active_indices <= t]
            prefix = arr[n, prefix_idx]
            out[n, t] = membership_for_observation(prefix, shape_set, weights, dtw_window, normalize)
    return out
