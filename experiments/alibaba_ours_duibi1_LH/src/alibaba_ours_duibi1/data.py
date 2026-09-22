from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ExperimentConfig, TaskSpec
from .progress import ProgressBar, print_stage
from .utils import atomic_save_npz, atomic_write_json


class DataError(RuntimeError):
    pass


@dataclass
class RawDataset:
    workload_ids: np.ndarray
    workload_node_ids: np.ndarray
    resource_names: np.ndarray
    deployment: np.ndarray
    observed_values: np.ndarray
    input_values: np.ndarray
    target_mask: np.ndarray
    input_mask: np.ndarray
    time_seconds: np.ndarray

    @property
    def workloads(self) -> int:
        return int(self.input_values.shape[0])

    @property
    def steps(self) -> int:
        return int(self.input_values.shape[1])

    @property
    def resources(self) -> int:
        return int(self.input_values.shape[2])


@dataclass
class PreparedData:
    workload_ids: np.ndarray
    resource_names: np.ndarray
    deployment: np.ndarray
    input_normalized: np.ndarray
    input_memberships: np.ndarray
    target_memberships: np.ndarray
    observed_values: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    time_seconds: np.ndarray
    resource_mean: np.ndarray
    resource_std: np.ndarray
    centroids_normalized: np.ndarray
    membership_scale2: float
    train_end: int
    validation_end: int
    cache_path: Path
    cache_signature: str

    @property
    def workloads(self) -> int:
        return int(self.input_normalized.shape[0])

    @property
    def steps(self) -> int:
        return int(self.input_normalized.shape[1])

    @property
    def resources(self) -> int:
        return int(self.input_normalized.shape[2])

    @property
    def num_states(self) -> int:
        return int(self.target_memberships.shape[-1])

    def split_bounds(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.train_end
        if split == "validation":
            return self.train_end, self.validation_end
        if split == "test":
            return self.validation_end, self.steps
        raise KeyError(split)


def load_raw_dataset(path: Path) -> RawDataset:
    if not path.is_file():
        raise DataError(f"数据集不存在: {path}")
    required = {
        "workload_ids",
        "resource_names",
        "deployment",
        "observed_values",
        "input_values",
        "target_observed_mask",
        "input_valid_mask",
        "time_seconds",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise DataError(f"dataset.npz 缺少字段: {sorted(missing)}")
        workload_ids = archive["workload_ids"].astype(str)
        workload_node_ids = (
            archive["workload_node_ids"].astype(str)
            if "workload_node_ids" in archive.files
            else np.full(workload_ids.shape, "", dtype=str)
        )
        resource_names = archive["resource_names"].astype(str)
        deployment = archive["deployment"].astype(np.float32)
        observed = archive["observed_values"].astype(np.float32)
        inputs = archive["input_values"].astype(np.float32)
        target_mask = archive["target_observed_mask"].astype(bool)
        input_mask = archive["input_valid_mask"].astype(bool)
        times = archive["time_seconds"].astype(np.int64)
    if observed.shape != inputs.shape or observed.ndim != 3:
        raise DataError("observed_values/input_values 必须具有相同的(N,T,R)形状")
    if target_mask.shape != observed.shape[:2] or input_mask.shape != observed.shape[:2]:
        raise DataError("有效值掩码必须具有(N,T)形状")
    if times.shape != (observed.shape[1],):
        raise DataError("time_seconds 长度与时间轴不一致")
    if deployment.ndim != 2 or deployment.shape[1] != observed.shape[0]:
        raise DataError("deployment 必须是(node, workload)矩阵")
    return RawDataset(
        workload_ids=workload_ids,
        workload_node_ids=workload_node_ids,
        resource_names=resource_names,
        deployment=deployment,
        observed_values=observed,
        input_values=inputs,
        target_mask=target_mask,
        input_mask=input_mask,
        time_seconds=times,
    )


def split_indices(total_steps: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    train_end = int(np.floor(total_steps * ratios[0]))
    validation_end = train_end + int(np.floor(total_steps * ratios[1]))
    if not 0 < train_end < validation_end < total_steps:
        raise DataError("时间切分后至少有一个区间为空")
    return train_end, validation_end


def valid_origins(
    *,
    split_start: int,
    split_end: int,
    history_length: int,
    horizon_steps: int,
    stride: int,
    training: bool = False,
) -> list[int]:
    # origin=t is the last observed slot. Targets are t+1 ... t+H.
    minimum = history_length - 1 if training else max(split_start, history_length - 1)
    maximum = split_end - horizon_steps - 1
    if maximum < minimum:
        return []
    step = 1 if training else max(int(stride), 1)
    return list(range(minimum, maximum + 1, step))


def _cache_signature(config: ExperimentConfig, dataset_path: Path) -> str:
    dataset_digest = hashlib.sha256()
    with dataset_path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            dataset_digest.update(block)
    payload = {
        "dataset_config_path": str(config.raw["dataset_path"]),
        "dataset_sha256": dataset_digest.hexdigest(),
        "time_step_seconds": config.time_step_seconds,
        "num_states": config.num_states,
        "split": config.raw["split"],
        "membership": config.membership,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _distance2(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)


def _fit_kmeans(
    points: np.ndarray,
    *,
    clusters: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] < clusters:
        raise DataError("训练数据不足以拟合K个候选模式")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, points.shape[0]))
    selected = [first]
    nearest = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, clusters):
        total = float(nearest.sum())
        if total <= 0 or not np.isfinite(total):
            candidate = int(rng.integers(0, points.shape[0]))
        else:
            candidate = int(rng.choice(points.shape[0], p=nearest / total))
        selected.append(candidate)
        nearest = np.minimum(nearest, np.sum((points - points[candidate]) ** 2, axis=1))
    centroids = points[selected].astype(np.float64, copy=True)
    for _ in range(max(int(iterations), 1)):
        labels = np.argmin(_distance2(points, centroids), axis=1)
        updated = centroids.copy()
        minimum_distance = np.min(_distance2(points, centroids), axis=1)
        for cluster in range(clusters):
            active = labels == cluster
            if active.any():
                updated[cluster] = points[active].mean(axis=0)
            else:
                replacement = int(np.argmax(minimum_distance))
                updated[cluster] = points[replacement]
                minimum_distance[replacement] = -1.0
        shift = float(np.max(np.linalg.norm(updated - centroids, axis=1)))
        centroids = updated
        if shift < 1e-6:
            break
    order = np.lexsort(tuple(centroids[:, index] for index in reversed(range(centroids.shape[1]))))
    return centroids[order].astype(np.float32)


def _memberships(
    values_normalized: np.ndarray,
    valid_mask: np.ndarray,
    centroids: np.ndarray,
    scale2: float,
    temperature: float,
) -> np.ndarray:
    workloads, steps, resources = values_normalized.shape
    output = np.full(
        (workloads, steps, centroids.shape[0]),
        1.0 / centroids.shape[0],
        dtype=np.float32,
    )
    flat = values_normalized.reshape(-1, resources)
    active = valid_mask.reshape(-1) & np.isfinite(flat).all(axis=1)
    indices = np.flatnonzero(active)
    denominator = max(2.0 * float(scale2) * float(temperature), 1e-8)
    chunk = 200_000
    flat_output = output.reshape(-1, centroids.shape[0])
    for offset in range(0, len(indices), chunk):
        selected = indices[offset : offset + chunk]
        distance = _distance2(flat[selected], centroids)
        logits = -distance / denominator
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
        flat_output[selected] = probabilities.astype(np.float32)
    return output


def prepare_data(config: ExperimentConfig, *, force: bool = False) -> PreparedData:
    signature = _cache_signature(config, config.dataset_path)
    cache_dir = config.output_dir / "_cache"
    cache_path = cache_dir / f"preprocessed-{signature[:16]}.npz"
    metadata_path = cache_dir / f"preprocessed-{signature[:16]}.json"
    if cache_path.is_file() and metadata_path.is_file() and not force:
        print_stage(f"复用逐时隙隶属度缓存：{cache_path}")
        return load_prepared_data(cache_path, signature)

    raw = load_raw_dataset(config.dataset_path)
    if raw.workloads != config.expected_workloads:
        raise DataError(
            f"要求{config.expected_workloads}个容器，数据集中实际为{raw.workloads}个"
        )
    cadence = np.diff(raw.time_seconds)
    if cadence.size and not np.all(cadence == config.time_step_seconds):
        raise DataError("time_seconds 不是严格的60秒等间隔时间轴")
    train_end, validation_end = split_indices(raw.steps, config.split_ratios)
    training_values = raw.observed_values[:, :train_end]
    training_mask = raw.target_mask[:, :train_end] & np.isfinite(training_values).all(axis=-1)
    samples = training_values[training_mask]
    if samples.shape[0] < config.num_states:
        raise DataError("训练段有效资源样本不足")
    mean = samples.mean(axis=0, dtype=np.float64)
    std = samples.std(axis=0, dtype=np.float64)
    minimum_scale = float(config.membership.get("minimum_scale", 1e-4))
    std = np.maximum(std, minimum_scale)
    normalized_samples = ((samples - mean) / std).astype(np.float32)
    max_samples = int(config.membership.get("max_fit_samples", 100_000))
    if normalized_samples.shape[0] > max_samples:
        rng = np.random.default_rng(config.seed)
        chosen = np.sort(rng.choice(normalized_samples.shape[0], max_samples, replace=False))
        fit_samples = normalized_samples[chosen]
    else:
        fit_samples = normalized_samples
    print_stage(
        f"使用训练段{fit_samples.shape[0]}个资源向量拟合固定K={config.num_states}候选模式"
    )
    centroids = _fit_kmeans(
        fit_samples,
        clusters=config.num_states,
        iterations=int(config.membership.get("kmeans_iterations", 40)),
        seed=config.seed,
    )
    nearest2 = np.min(_distance2(fit_samples, centroids), axis=1)
    positive = nearest2[nearest2 > 1e-12]
    scale2 = float(np.median(positive)) if positive.size else 1.0
    scale2 = max(scale2, minimum_scale**2)

    input_normalized = ((raw.input_values - mean) / std).astype(np.float32)
    observed_normalized = ((raw.observed_values - mean) / std).astype(np.float32)
    input_memberships = _memberships(
        input_normalized,
        raw.input_mask,
        centroids,
        scale2,
        float(config.membership.get("temperature", 1.0)),
    )
    target_memberships = _memberships(
        observed_normalized,
        raw.target_mask,
        centroids,
        scale2,
        float(config.membership.get("temperature", 1.0)),
    )
    input_normalized = np.nan_to_num(input_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    print_stage(f"写入可复用缓存：{cache_path}")
    atomic_save_npz(
        cache_path,
        compressed=True,
        workload_ids=raw.workload_ids,
        resource_names=raw.resource_names,
        deployment=raw.deployment,
        input_normalized=input_normalized,
        input_memberships=input_memberships,
        target_memberships=target_memberships,
        observed_values=raw.observed_values,
        input_mask=raw.input_mask.astype(np.uint8),
        target_mask=raw.target_mask.astype(np.uint8),
        time_seconds=raw.time_seconds,
        resource_mean=mean.astype(np.float32),
        resource_std=std.astype(np.float32),
        centroids_normalized=centroids,
        membership_scale2=np.asarray(scale2, dtype=np.float64),
        train_end=np.asarray(train_end, dtype=np.int64),
        validation_end=np.asarray(validation_end, dtype=np.int64),
        cache_signature=np.asarray(signature),
    )
    atomic_write_json(
        metadata_path,
        {
            "cache_signature": signature,
            "source_dataset": config.dataset_path,
            "workloads": raw.workloads,
            "steps": raw.steps,
            "resources": raw.resources,
            "num_states": config.num_states,
            "train_end": train_end,
            "validation_end": validation_end,
            "membership_definition": "fixed per-slot resource-vector fuzzy membership fitted on train only",
            "membership_independent_of_L_and_H": True,
            "resource_mean": mean,
            "resource_std": std,
            "centroids_normalized": centroids,
            "membership_scale2": scale2,
        },
    )
    return load_prepared_data(cache_path, signature)


def load_prepared_data(path: Path, expected_signature: str | None = None) -> PreparedData:
    with np.load(path, allow_pickle=False) as archive:
        signature = str(archive["cache_signature"].item())
        if expected_signature is not None and signature != expected_signature:
            raise DataError("缓存签名不匹配")
        return PreparedData(
            workload_ids=archive["workload_ids"].astype(str),
            resource_names=archive["resource_names"].astype(str),
            deployment=archive["deployment"].astype(np.float32),
            input_normalized=archive["input_normalized"].astype(np.float32),
            input_memberships=archive["input_memberships"].astype(np.float32),
            target_memberships=archive["target_memberships"].astype(np.float32),
            observed_values=archive["observed_values"].astype(np.float32),
            input_mask=archive["input_mask"].astype(bool),
            target_mask=archive["target_mask"].astype(bool),
            time_seconds=archive["time_seconds"].astype(np.int64),
            resource_mean=archive["resource_mean"].astype(np.float32),
            resource_std=archive["resource_std"].astype(np.float32),
            centroids_normalized=archive["centroids_normalized"].astype(np.float32),
            membership_scale2=float(archive["membership_scale2"].item()),
            train_end=int(archive["train_end"].item()),
            validation_end=int(archive["validation_end"].item()),
            cache_path=path,
            cache_signature=signature,
        )


def origin_report(config: ExperimentConfig, data: PreparedData) -> list[dict[str, Any]]:
    stride = int(config.evaluation.get("origin_stride_steps", 60))
    rows: list[dict[str, Any]] = []
    for task in config.task_queue():
        train = valid_origins(
            split_start=0,
            split_end=data.train_end,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            stride=1,
            training=True,
        )
        validation = valid_origins(
            split_start=data.train_end,
            split_end=data.validation_end,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            stride=stride,
        )
        test = valid_origins(
            split_start=data.validation_end,
            split_end=data.steps,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            stride=stride,
        )
        rows.append(
            {
                **task.as_dict(),
                "train_origins_available": len(train),
                "validation_origins": len(validation),
                "test_origins": len(test),
            }
        )
    return rows


__all__ = [
    "DataError",
    "PreparedData",
    "RawDataset",
    "load_prepared_data",
    "load_raw_dataset",
    "origin_report",
    "prepare_data",
    "split_indices",
    "valid_origins",
]
