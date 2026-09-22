from __future__ import annotations

"""Resume-safe physical-time resampling of the portable Alibaba cache."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from alibaba_ours_exp.progress import ProgressTracker

from .design import Condition


REQUIRED_ARRAYS = {
    "workload_ids",
    "workload_node_ids",
    "resource_names",
    "observed_values",
    "input_values",
    "target_observed_mask",
    "input_valid_mask",
    "imputed_mask",
    "time_seconds",
    "deployment",
    "node_ids",
    "cohort_eligible_mask",
    "history_coverage",
    "cohort_target_observed_mask",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _masked_bucket_mean(
    values: np.ndarray,
    mask: np.ndarray,
    factor: int,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workloads, total_steps, resources = values.shape
    buckets = total_steps // factor
    selected = values[:, : buckets * factor].reshape(workloads, buckets, factor, resources)
    selected_mask = mask[:, : buckets * factor].reshape(workloads, buckets, factor)
    counts = selected_mask.sum(axis=2, dtype=np.int32)
    sums = np.where(selected_mask[..., None], selected, 0.0).sum(axis=2, dtype=np.float64)
    means = sums / np.maximum(counts[..., None], 1)
    valid = counts >= int(np.ceil(minimum_coverage * factor - 1e-12))
    means = means.astype(np.float32)
    means[~valid] = np.nan
    return means, valid, counts.astype(np.int32)


def _cohort_arrays(
    input_valid: np.ndarray,
    target_valid: np.ndarray,
    condition: Condition,
    split_ratios: tuple[float, float, float],
    minimum_history_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = input_valid.shape[1]
    train_stop = int(total * split_ratios[0])
    validation_stop = train_stop + int(total * split_ratios[1])
    rows: list[int] = []
    split_codes: list[int] = []
    for split_code, (start, stop) in enumerate(
        ((train_stop, validation_stop), (validation_stop, total)), start=1
    ):
        first = max(start - 1, condition.history_length - 1)
        last = stop - condition.forecast_horizon - 1
        if first <= last:
            for origin in range(first, last + 1, condition.origin_stride_steps):
                rows.append(origin)
                split_codes.append(split_code)
    history_coverage = np.empty((len(rows), input_valid.shape[0]), dtype=np.float32)
    eligible = np.empty_like(history_coverage, dtype=bool)
    targets = np.empty((len(rows), input_valid.shape[0], 1), dtype=bool)
    for row, origin in enumerate(rows):
        history = input_valid[:, origin - condition.history_length + 1 : origin + 1]
        coverage = history.mean(axis=1, dtype=np.float64)
        target = target_valid[:, origin + 1]
        history_coverage[row] = coverage.astype(np.float32)
        eligible[row] = (coverage >= minimum_history_coverage) & input_valid[:, origin]
        targets[row, :, 0] = target
    return (
        eligible,
        history_coverage,
        targets,
        np.column_stack(
            [np.asarray(rows, dtype=np.int64), np.asarray(split_codes, dtype=np.int8)]
        ),
    )


def _validate_cache(path: Path, expected_steps: int | None = None) -> dict[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS.difference(archive.files))
        if missing:
            raise ValueError(f"portable dataset is missing arrays: {missing}")
        shapes = {name: tuple(int(value) for value in archive[name].shape) for name in archive.files}
        if expected_steps is not None and shapes["input_values"][1] != expected_steps:
            raise ValueError(
                f"resampled cache has {shapes['input_values'][1]} steps, expected {expected_steps}"
            )
        return shapes


def ensure_resampled_dataset(
    source_path: str | Path,
    output_root: str | Path,
    condition: Condition,
    *,
    split_ratios: tuple[float, float, float] = (0.60, 0.20, 0.20),
    minimum_bucket_coverage: float = 0.80,
    minimum_history_coverage: float = 0.80,
    progress_root: str | Path | None = None,
    resume: bool = True,
) -> Path:
    source = Path(source_path).resolve()
    if condition.granularity_seconds == 60:
        _validate_cache(source)
        return source

    destination = Path(output_root).resolve() / f"tau-{condition.granularity_seconds}s"
    dataset_path = destination / "dataset.npz"
    manifest_path = destination / "dataset_manifest.json"
    source_digest = _sha256(source)
    expected_steps = (777_600 - 86_400) // condition.granularity_seconds
    if resume and dataset_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_sha256") == source_digest
            and int(manifest.get("granularity_seconds", -1)) == condition.granularity_seconds
            and int(manifest.get("history_length", -1)) == condition.history_length
            and float(manifest.get("minimum_bucket_coverage", -1.0))
            == float(minimum_bucket_coverage)
        ):
            _validate_cache(dataset_path, expected_steps)
            return dataset_path

    tracker = ProgressTracker(
        Path(progress_root or destination / "progress") / f"tau-{condition.granularity_seconds}s",
        run_id=f"resample-tau-{condition.granularity_seconds}s",
    )
    tracker.start(
        total=4,
        stage="resample-tau",
        message=f"loading source tau=60s -> tau={condition.granularity_seconds}s",
        reset=True,
    )
    try:
        with np.load(source, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        tracker.update(completed=1, message="source cache loaded")

        factor = condition.granularity_seconds // 60
        if factor <= 0 or condition.granularity_seconds % 60:
            raise ValueError("tau must be a positive multiple of 60 seconds")
        observed, target_valid, observed_counts = _masked_bucket_mean(
            payload["observed_values"].astype(np.float32),
            payload["target_observed_mask"].astype(bool),
            factor,
            minimum_bucket_coverage,
        )
        inputs, input_valid, input_counts = _masked_bucket_mean(
            payload["input_values"].astype(np.float32),
            payload["input_valid_mask"].astype(bool),
            factor,
            minimum_bucket_coverage,
        )
        buckets = inputs.shape[1]
        imputed = payload["imputed_mask"][:, : buckets * factor].reshape(
            inputs.shape[0], buckets, factor
        ).any(axis=2)
        tracker.update(completed=2, message="resource buckets aggregated")

        eligible, history_coverage, cohort_targets, origin_grid = _cohort_arrays(
            input_valid,
            target_valid,
            condition,
            split_ratios,
            minimum_history_coverage,
        )
        times = payload["time_seconds"][: buckets * factor : factor].astype(np.int64)
        output = {
            "workload_ids": payload["workload_ids"],
            "workload_node_ids": payload["workload_node_ids"],
            "resource_names": payload["resource_names"],
            "observed_values": observed,
            "input_values": inputs,
            "target_observed_mask": target_valid,
            "input_valid_mask": input_valid,
            "imputed_mask": imputed,
            "time_seconds": times,
            "deployment": payload["deployment"],
            "node_ids": payload["node_ids"],
            "cohort_eligible_mask": eligible,
            "history_coverage": history_coverage,
            "cohort_target_observed_mask": cohort_targets,
            "resample_observed_counts": observed_counts,
            "resample_input_counts": input_counts,
            "resample_origin_grid": origin_grid,
        }
        tracker.update(completed=3, message="cohort masks rebuilt")
        _write_npz_atomic(dataset_path, output)
        manifest = {
            "schema_version": 1,
            "source_name": "portable_prepared_cache_resampled",
            "source_path": str(source),
            "source_sha256": source_digest,
            "dataset_sha256": _sha256(dataset_path),
            "granularity_seconds": condition.granularity_seconds,
            "history_duration_seconds": condition.history_duration_seconds,
            "history_length": condition.history_length,
            "forecast_horizon_steps": condition.forecast_horizon,
            "origin_stride_steps": condition.origin_stride_steps,
            "minimum_bucket_coverage": minimum_bucket_coverage,
            "minimum_history_coverage": minimum_history_coverage,
            "workload_count": int(inputs.shape[0]),
            "time_steps": int(inputs.shape[1]),
            "resource_names": payload["resource_names"].astype(str).tolist(),
            "observed_bucket_coverage_mean": float(target_valid.mean()),
            "input_bucket_coverage_mean": float(input_valid.mean()),
            "origin_count": int(eligible.shape[0]),
        }
        _write_json_atomic(manifest_path, manifest)
        tracker.update(completed=4, message="resampled portable dataset saved")
        tracker.complete(summary=manifest, message="resampling complete")
        return dataset_path
    except BaseException as exc:
        tracker.fail(exc, context={"tau_seconds": condition.granularity_seconds})
        raise


__all__ = ["REQUIRED_ARRAYS", "ensure_resampled_dataset"]
