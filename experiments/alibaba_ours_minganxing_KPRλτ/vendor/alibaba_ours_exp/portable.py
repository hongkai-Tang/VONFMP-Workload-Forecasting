"""Portable, human-readable export of the prepared 200-workload dataset."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import hash_file
from .metrics import atomic_write_csv, atomic_write_json


def atomic_copy_file(source: str | Path, destination: str | Path) -> Path:
    """Copy a generated artifact through a temporary file and atomic replace."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        with source_path.open("rb") as input_handle, temporary.open("wb") as output_handle:
            while True:
                block = input_handle.read(8 * 1024 * 1024)
                if not block:
                    break
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


def _cell(value: Any) -> str:
    number = float(value)
    return "" if not math.isfinite(number) else format(number, ".9g")


def _split_name(index: int, total: int, split_ratios: Sequence[float]) -> str:
    train_stop = int(total * float(split_ratios[0]))
    validation_stop = train_stop + int(total * float(split_ratios[1]))
    if index < train_stop:
        return "train"
    if index < validation_stop:
        return "validation"
    return "test"


def export_portable_cache(
    cache: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    split_ratios: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    """Stream a prepared cache to one atomic CSV without materializing rows."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "processed_60s.csv"
    workload_ids = np.asarray(cache["workload_ids"]).astype(str)
    node_ids = np.asarray(cache["workload_node_ids"]).astype(str)
    resources = np.asarray(cache["resource_names"]).astype(str)
    times = np.asarray(cache["time_seconds"], dtype=np.int64)
    observed = np.asarray(cache["observed_values"], dtype=np.float32)
    inputs = np.asarray(cache["input_values"], dtype=np.float32)
    target_mask = np.asarray(cache["target_observed_mask"], dtype=bool)
    input_mask = np.asarray(cache["input_valid_mask"], dtype=bool)
    imputed_mask = np.asarray(cache["imputed_mask"], dtype=bool)
    expected = (len(workload_ids), len(times), len(resources))
    if observed.shape != expected or inputs.shape != expected:
        raise ValueError("portable export resource arrays have inconsistent shapes")
    if target_mask.shape != expected[:2] or input_mask.shape != expected[:2]:
        raise ValueError("portable export masks have inconsistent shapes")

    fields = [
        "workload_id",
        "node_id",
        "time_index",
        "time_seconds",
        "time_bucket_60s",
        "split",
        *[f"{name}_observed" for name in resources],
        *[f"{name}_input" for name in resources],
        "target_observed",
        "input_valid",
        "imputed",
    ]
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=destination
    )
    temporary = Path(raw_temp)
    row_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            total_steps = len(times)
            for workload_index, (workload, node) in enumerate(zip(workload_ids, node_ids)):
                for time_index, time_seconds in enumerate(times):
                    writer.writerow(
                        [
                            workload,
                            node,
                            time_index,
                            int(time_seconds),
                            int(time_seconds) // 60,
                            _split_name(time_index, total_steps, split_ratios),
                            *(_cell(value) for value in observed[workload_index, time_index]),
                            *(_cell(value) for value in inputs[workload_index, time_index]),
                            int(target_mask[workload_index, time_index]),
                            int(input_mask[workload_index, time_index]),
                            int(imputed_mask[workload_index, time_index]),
                        ]
                    )
                    row_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    summary_rows = []
    for index, (workload, node) in enumerate(zip(workload_ids, node_ids)):
        summary_rows.append(
            {
                "workload_id": workload,
                "node_id": node,
                "time_steps": len(times),
                "observed_points": int(target_mask[index].sum()),
                "input_valid_points": int(input_mask[index].sum()),
                "imputed_points": int(imputed_mask[index].sum()),
                "observed_coverage": float(target_mask[index].mean()),
            }
        )
    summary_path = atomic_write_csv(destination / "selected_workloads.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "description": "Alibaba Ours experiment portable prepared dataset",
        "seed": int(seed),
        "workload_count": int(len(workload_ids)),
        "time_steps_per_workload": int(len(times)),
        "row_count": int(row_count),
        "resource_names": resources.tolist(),
        "csv_file": csv_path.name,
        "csv_bytes": int(csv_path.stat().st_size),
        "csv_sha256": hash_file(csv_path),
        "summary_file": summary_path.name,
        "summary_sha256": hash_file(summary_path),
        "contains_original_unused_columns": False,
    }
    raw_path = destination / "container_usage_selected_200.csv"
    raw_schema_path = destination / "raw_schema.json"
    if raw_path.is_file():
        raw_by_workload = destination / "raw_by_workload"
        individual_files = sorted(raw_by_workload.glob("*.csv"))
        manifest.update(
            {
                "raw_csv_file": raw_path.name,
                "raw_csv_bytes": int(raw_path.stat().st_size),
                "raw_csv_sha256": hash_file(raw_path),
                "raw_schema_file": raw_schema_path.name if raw_schema_path.is_file() else None,
                "contains_original_unused_columns": True,
                "raw_rows_preserved_byte_for_byte": True,
                "raw_by_workload_dir": raw_by_workload.name,
                "raw_by_workload_file_count": len(individual_files),
                "raw_by_workload_files": [path.name for path in individual_files],
            }
        )
    dataset_path = destination / "dataset.npz"
    if dataset_path.is_file():
        manifest.update(
            {
                "portable_dataset_file": dataset_path.name,
                "portable_dataset_bytes": int(dataset_path.stat().st_size),
                "portable_dataset_sha256": hash_file(dataset_path),
            }
        )
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


__all__ = ["atomic_copy_file", "export_portable_cache"]
