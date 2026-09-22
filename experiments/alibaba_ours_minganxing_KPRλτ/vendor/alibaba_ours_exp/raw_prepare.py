from __future__ import annotations

"""Resumable one-pass and two-pass preparation for Alibaba's large raw CSV."""

import json
import math
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .config import ExperimentConfig, SourceConfig
from .data_index import build_data_index
from .dataset import (
    DatasetError,
    DenseDataset,
    SchemaReport,
    WorkloadStats,
    bounded_causal_fill,
    build_deployment_matrix,
    build_origin_cohort,
    preflight_source,
    select_workloads,
)
from .metrics import atomic_write_json, atomic_write_npz


ProgressCallback = Callable[[str, int, int, int], None]


@dataclass(frozen=True)
class ScanState:
    byte_offset: int
    rows_scanned: int


def _parse_binary_line(
    line: bytes,
    source: SourceConfig,
    config: ExperimentConfig,
    column_position: Mapping[str, int],
    allowed_workloads: Mapping[bytes, int] | None = None,
) -> tuple[str, str | None, int, np.ndarray] | None:
    try:
        values = line.rstrip(b"\r\n").split(source.delimiter.encode("ascii"))
        workload_bytes = values[column_position[source.workload_column]].strip()
        if not workload_bytes:
            return None
        # The extraction pass only needs 200 selected workloads.  Reject every
        # other row before parsing timestamps/resources or allocating NumPy
        # arrays; this makes pass 2 primarily a sequential disk scan.
        if allowed_workloads is not None and workload_bytes not in allowed_workloads:
            return None
        workload = workload_bytes.decode("utf-8")
        node: str | None = None
        if source.node_column is not None:
            node = values[column_position[source.node_column]].decode("utf-8").strip() or None
        raw_time = float(values[column_position[source.time_column]])
        if source.time_unit == "bucket":
            time_seconds = int(math.floor(raw_time)) * config.resample_seconds
        else:
            time_seconds = int(math.floor(raw_time / config.resample_seconds)) * config.resample_seconds
        if not config.start_seconds <= time_seconds < config.end_seconds_exclusive:
            return None
        parsed: dict[str, float] = {}
        output = np.empty(len(source.resources), dtype=np.float64)
        for index, rule in enumerate(source.resources):
            if rule.column not in parsed:
                parsed[rule.column] = float(values[column_position[rule.column]])
            raw = parsed[rule.column]
            if not math.isfinite(raw):
                return None
            if rule.valid_min is not None and raw < rule.valid_min:
                return None
            if rule.valid_max is not None and raw > rule.valid_max:
                return None
            output[index] = raw * rule.scale
        return workload, node, time_seconds, output
    except (IndexError, KeyError, UnicodeError, ValueError, OverflowError):
        return None


def _parse_stats_line(
    line: bytes,
    source: SourceConfig,
    config: ExperimentConfig,
    column_position: Mapping[str, int],
) -> tuple[bytes, int] | None:
    """Validate one raw row for pass 1 without allocating a resource vector."""

    try:
        values = line.rstrip(b"\r\n").split(source.delimiter.encode("ascii"))
        workload = values[column_position[source.workload_column]].strip()
        if not workload:
            return None
        raw_time = float(values[column_position[source.time_column]])
        if source.time_unit == "bucket":
            time_seconds = int(math.floor(raw_time)) * config.resample_seconds
        else:
            time_seconds = int(math.floor(raw_time / config.resample_seconds)) * config.resample_seconds
        if not config.start_seconds <= time_seconds < config.end_seconds_exclusive:
            return None
        parsed: dict[str, float] = {}
        for rule in source.resources:
            if rule.column not in parsed:
                parsed[rule.column] = float(values[column_position[rule.column]])
            raw = parsed[rule.column]
            if not math.isfinite(raw):
                return None
            if rule.valid_min is not None and raw < rule.valid_min:
                return None
            if rule.valid_max is not None and raw > rule.valid_max:
                return None
        return workload, time_seconds
    except (IndexError, KeyError, UnicodeError, ValueError, OverflowError):
        return None


def _column_positions(source: SourceConfig) -> dict[str, int]:
    if source.has_header:
        raise DatasetError("resumable raw scanner currently requires a headerless CSV")
    return {name: index for index, name in enumerate(source.columns)}


def _checkpoint_paths(directory: Path, stage: str) -> tuple[Path, Path]:
    return directory / f"{stage}.npz", directory / f"{stage}.json"


def _load_scan_stats(directory: Path) -> tuple[ScanState, dict[str, WorkloadStats]] | None:
    arrays_path, metadata_path = _checkpoint_paths(directory, "pass1")
    if not arrays_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        identifiers = archive["workload_ids"].astype(str)
        valid_rows = archive["valid_rows"].astype(np.int64)
        valid_time_points = archive["valid_time_points"].astype(np.int64)
        minimum = archive["min_time_seconds"].astype(np.int64)
        maximum = archive["max_time_seconds"].astype(np.int64)
        archive_last = archive["last_time_seconds"].astype(np.int64)
    stats: dict[str, WorkloadStats] = {}
    for workload, count, point_count, start, stop, last in zip(
        identifiers, valid_rows, valid_time_points, minimum, maximum, archive_last
    ):
        stats[str(workload)] = WorkloadStats(
            workload_id=str(workload),
            valid_rows=int(count),
            valid_time_points=int(point_count),
            min_time_seconds=None if int(start) < 0 else int(start),
            max_time_seconds=None if int(stop) < 0 else int(stop),
            last_time_seconds=None if int(last) < 0 else int(last),
        )
    return ScanState(int(metadata["byte_offset"]), int(metadata["rows_scanned"])), stats


def _save_scan_stats(
    directory: Path,
    state: ScanState,
    stats: Mapping[str, WorkloadStats],
) -> None:
    arrays_path, metadata_path = _checkpoint_paths(directory, "pass1")
    ordered = sorted(stats)
    atomic_write_npz(
        arrays_path,
        {
            "workload_ids": np.asarray(ordered, dtype=str),
            "valid_rows": np.asarray([stats[item].valid_rows for item in ordered], dtype=np.int64),
            "valid_time_points": np.asarray(
                [stats[item].valid_time_points for item in ordered], dtype=np.int64
            ),
            "min_time_seconds": np.asarray(
                [stats[item].min_time_seconds if stats[item].min_time_seconds is not None else -1 for item in ordered],
                dtype=np.int64,
            ),
            "max_time_seconds": np.asarray(
                [stats[item].max_time_seconds if stats[item].max_time_seconds is not None else -1 for item in ordered],
                dtype=np.int64,
            ),
            "last_time_seconds": np.asarray(
                [stats[item].last_time_seconds if stats[item].last_time_seconds is not None else -1 for item in ordered],
                dtype=np.int64,
            ),
        },
    )
    atomic_write_json(
        metadata_path,
        {"byte_offset": state.byte_offset, "rows_scanned": state.rows_scanned},
    )


def scan_stats_resumable(
    source: SourceConfig,
    config: ExperimentConfig,
    checkpoint_dir: Path,
    *,
    resume: bool,
    checkpoint_every_rows: int = 10_000_000,
    progress_every_rows: int = 100_000,
    progress_every_seconds: float = 2.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, WorkloadStats]:
    positions = _column_positions(source)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    restored = _load_scan_stats(checkpoint_dir) if resume else None
    state, stats = restored if restored is not None else (ScanState(0, 0), {})
    stats_by_bytes = {key.encode("utf-8"): value for key, value in stats.items()}
    total_bytes = source.path.stat().st_size
    with source.path.open("rb") as handle:
        handle.seek(state.byte_offset)
        rows = state.rows_scanned
        if progress_callback is not None:
            progress_callback("prepare-pass1", rows, state.byte_offset, total_bytes)
        last_progress_at = time.monotonic()
        while True:
            line = handle.readline()
            if not line:
                break
            rows += 1
            parsed = _parse_stats_line(line, source, config, positions)
            if parsed is not None:
                workload_bytes, time_seconds = parsed
                item = stats_by_bytes.get(workload_bytes)
                if item is None:
                    workload = workload_bytes.decode("utf-8")
                    item = WorkloadStats(workload_id=workload)
                    stats[workload] = item
                    stats_by_bytes[workload_bytes] = item
                item.observe(time_seconds)
            if progress_callback is not None and rows % progress_every_rows == 0:
                now = time.monotonic()
                if now - last_progress_at >= progress_every_seconds:
                    progress_callback("prepare-pass1", rows, handle.tell(), total_bytes)
                    last_progress_at = now
            if rows % checkpoint_every_rows == 0:
                _save_scan_stats(checkpoint_dir, ScanState(handle.tell(), rows), stats)
        final_state = ScanState(handle.tell(), rows)
    _save_scan_stats(checkpoint_dir, final_state, stats)
    if progress_callback is not None:
        progress_callback("prepare-pass1", final_state.rows_scanned, total_bytes, total_bytes)
    return stats


def _load_extract_state(
    directory: Path,
    shape: tuple[int, int, int],
    checkpoint_stage: str = "pass2",
) -> tuple[ScanState, np.ndarray, np.ndarray, list[dict[str, int]]] | None:
    arrays_path, metadata_path = _checkpoint_paths(directory, checkpoint_stage)
    nodes_path = directory / f"{checkpoint_stage}-nodes.json"
    if not arrays_path.exists() or not metadata_path.exists() or not nodes_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        sums = archive["sums"].astype(np.float64)
        counts = archive["counts"].astype(np.int32)
    if sums.shape != shape or counts.shape != shape[:2]:
        raise DatasetError(
            f"{checkpoint_stage} checkpoint shape does not match the current cohort"
        )
    node_counts = [
        {str(key): int(value) for key, value in item.items()}
        for item in json.loads(nodes_path.read_text(encoding="utf-8"))
    ]
    return ScanState(int(metadata["byte_offset"]), int(metadata["rows_scanned"])), sums, counts, node_counts


def _save_extract_state(
    directory: Path,
    state: ScanState,
    sums: np.ndarray,
    counts: np.ndarray,
    node_counts: list[dict[str, int]],
    checkpoint_stage: str = "pass2",
    raw_output_offset: int | None = None,
) -> None:
    arrays_path, metadata_path = _checkpoint_paths(directory, checkpoint_stage)
    atomic_write_npz(arrays_path, {"sums": sums, "counts": counts})
    metadata = {"byte_offset": state.byte_offset, "rows_scanned": state.rows_scanned}
    if raw_output_offset is not None:
        metadata["raw_output_offset"] = int(raw_output_offset)
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(directory / f"{checkpoint_stage}-nodes.json", node_counts)


def extract_selected_resumable(
    source: SourceConfig,
    config: ExperimentConfig,
    workload_ids: tuple[str, ...],
    checkpoint_dir: Path,
    *,
    resume: bool,
    checkpoint_every_rows: int = 10_000_000,
    progress_every_rows: int = 100_000,
    progress_every_seconds: float = 2.0,
    progress_callback: ProgressCallback | None = None,
    checkpoint_stage: str = "pass2",
    progress_stage: str = "prepare-pass2",
    require_all_nodes: bool = True,
    raw_capture_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    positions = _column_positions(source)
    workload_position = {item: index for index, item in enumerate(workload_ids)}
    workload_position_bytes = {
        item.encode("utf-8"): index for item, index in workload_position.items()
    }
    shape = (len(workload_ids), config.total_steps, len(source.resources))
    restored = (
        _load_extract_state(checkpoint_dir, shape, checkpoint_stage) if resume else None
    )
    if restored is None:
        state = ScanState(0, 0)
        sums = np.zeros(shape, dtype=np.float64)
        counts = np.zeros(shape[:2], dtype=np.int32)
        node_counts: list[dict[str, int]] = [dict() for _ in workload_ids]
    else:
        state, sums, counts, node_counts = restored
    raw_handle = None
    if raw_capture_path is not None:
        capture = Path(raw_capture_path)
        capture.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = capture.open("r+b" if capture.exists() else "w+b")
        raw_offset = 0
        if restored is not None:
            _, metadata_path = _checkpoint_paths(checkpoint_dir, checkpoint_stage)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_offset = int(metadata.get("raw_output_offset", 0))
        raw_handle.truncate(raw_offset)
        raw_handle.seek(raw_offset)
    total_bytes = source.path.stat().st_size
    delimiter = source.delimiter.encode("ascii")
    try:
        with source.path.open("rb") as handle:
            handle.seek(state.byte_offset)
            rows = state.rows_scanned
            if progress_callback is not None:
                progress_callback(progress_stage, rows, state.byte_offset, total_bytes)
            last_progress_at = time.monotonic()
            while True:
                line = handle.readline()
                if not line:
                    break
                rows += 1
                if raw_handle is not None:
                    raw_id = line.split(delimiter, 1)[0].strip()
                    if raw_id in workload_position_bytes:
                        raw_handle.write(line)
                parsed = _parse_binary_line(
                    line,
                    source,
                    config,
                    positions,
                    allowed_workloads=workload_position_bytes,
                )
                if parsed is not None:
                    workload, node, time_seconds, values = parsed
                    workload_index = workload_position.get(workload)
                    if workload_index is not None:
                        time_index = (time_seconds - config.start_seconds) // config.resample_seconds
                        sums[workload_index, time_index] += values
                        counts[workload_index, time_index] += 1
                        if node:
                            node_counts[workload_index][node] = node_counts[workload_index].get(node, 0) + 1
                if progress_callback is not None and rows % progress_every_rows == 0:
                    now = time.monotonic()
                    if now - last_progress_at >= progress_every_seconds:
                        progress_callback(progress_stage, rows, handle.tell(), total_bytes)
                        last_progress_at = now
                if rows % checkpoint_every_rows == 0:
                    if raw_handle is not None:
                        raw_handle.flush()
                        os.fsync(raw_handle.fileno())
                    _save_extract_state(
                        checkpoint_dir,
                        ScanState(handle.tell(), rows),
                        sums,
                        counts,
                        node_counts,
                        checkpoint_stage,
                        None if raw_handle is None else raw_handle.tell(),
                    )
            final_state = ScanState(handle.tell(), rows)
        if raw_handle is not None:
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        _save_extract_state(
            checkpoint_dir,
            final_state,
            sums,
            counts,
            node_counts,
            checkpoint_stage,
            None if raw_handle is None else raw_handle.tell(),
        )
    finally:
        if raw_handle is not None:
            raw_handle.close()
    observed = counts > 0
    values = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, np.maximum(counts[..., None], 1), out=values, where=observed[..., None])
    nodes: list[str] = []
    for workload, counts_by_node in zip(workload_ids, node_counts):
        if not counts_by_node:
            if require_all_nodes:
                raise DatasetError(f"selected workload {workload} has no deployment node")
            nodes.append("")
        else:
            nodes.append(min(counts_by_node, key=lambda item: (-counts_by_node[item], item)))
    if progress_callback is not None:
        progress_callback(progress_stage, final_state.rows_scanned, total_bytes, total_bytes)
    return values, observed, tuple(nodes)


def export_selected_raw_rows(
    candidate_rows_path: str | Path,
    output_path: str | Path,
    selected_workloads: tuple[str, ...],
    *,
    delimiter: str = ",",
    per_workload_dir: str | Path | None = None,
) -> dict[str, int]:
    """Copy selected raw rows byte-for-byte into an atomic headerless CSV."""

    source = Path(candidate_rows_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = {item.encode("utf-8") for item in selected_workloads}
    counts = {item: 0 for item in selected_workloads}
    encoded_delimiter = delimiter.encode("ascii")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_temp)
    workload_handles: dict[bytes, Any] = {}
    workload_temporary: dict[bytes, Path] = {}
    workload_destinations: dict[bytes, Path] = {}
    try:
        if per_workload_dir is not None:
            individual_root = Path(per_workload_dir)
            individual_root.mkdir(parents=True, exist_ok=True)
            for workload in selected_workloads:
                workload_bytes = workload.encode("utf-8")
                item_descriptor, item_temp = tempfile.mkstemp(
                    prefix=f".{workload}.", suffix=".tmp", dir=individual_root
                )
                workload_handles[workload_bytes] = os.fdopen(item_descriptor, "wb")
                workload_temporary[workload_bytes] = Path(item_temp)
                workload_destinations[workload_bytes] = individual_root / f"{workload}.csv"
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            for line in input_handle:
                workload_bytes = line.split(encoded_delimiter, 1)[0].strip()
                if workload_bytes in selected:
                    output_handle.write(line)
                    if workload_bytes in workload_handles:
                        workload_handles[workload_bytes].write(line)
                    counts[workload_bytes.decode("utf-8")] += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        for workload_bytes, handle in workload_handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(
                workload_temporary[workload_bytes],
                workload_destinations[workload_bytes],
            )
        os.replace(temporary, destination)
    except Exception:
        for handle in workload_handles.values():
            if not handle.closed:
                handle.close()
        for path in workload_temporary.values():
            path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    return counts


def load_seeded_candidate_workloads(config: ExperimentConfig) -> tuple[str, ...]:
    """Deterministically sample candidate IDs from the small container metadata."""

    metadata_path = config.candidate_metadata_path
    if metadata_path is None or not metadata_path.is_file():
        raise DatasetError(
            "seeded_metadata_candidates requires selection.candidate_metadata_path"
        )
    identifiers: set[bytes] = set()
    if tarfile.is_tarfile(metadata_path):
        with tarfile.open(metadata_path, mode="r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == "container_meta.csv"
            ]
            if not members:
                raise DatasetError("container_meta.csv is missing from candidate metadata archive")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise DatasetError("cannot read container_meta.csv from metadata archive")
            for line in extracted:
                identifier = line.split(b",", 1)[0].strip()
                if identifier and identifier != b"container_id":
                    identifiers.add(identifier)
    else:
        with metadata_path.open("rb") as handle:
            for line in handle:
                identifier = line.split(b",", 1)[0].strip()
                if identifier and identifier != b"container_id":
                    identifiers.add(identifier)
    ordered = sorted(identifiers)
    if len(ordered) < config.candidate_pool_size:
        raise DatasetError(
            f"candidate metadata has {len(ordered)} workloads, fewer than "
            f"candidate_pool_size={config.candidate_pool_size}"
        )
    rng = np.random.default_rng(int(config.protocol["seed"]))
    positions = np.sort(
        rng.choice(len(ordered), size=config.candidate_pool_size, replace=False)
    )
    return tuple(ordered[int(position)].decode("utf-8") for position in positions)


def build_seeded_candidate_dataset_resumable(
    config: ExperimentConfig,
    source: SourceConfig,
    checkpoint_dir: str | Path,
    *,
    resume: bool,
    progress_callback: ProgressCallback | None = None,
    raw_export_path: str | Path | None = None,
) -> DenseDataset:
    """Prepare a reproducible 200-workload cohort with one raw CSV pass."""

    report = preflight_source(source)
    if not report.ok:
        raise DatasetError(f"raw source failed preflight: {report}")
    candidates = load_seeded_candidate_workloads(config)
    checkpoint_root = Path(checkpoint_dir)
    candidate_values, candidate_mask, candidate_nodes = extract_selected_resumable(
        source,
        config,
        candidates,
        checkpoint_root,
        resume=resume,
        checkpoint_every_rows=200_000_000,
        progress_callback=progress_callback,
        checkpoint_stage="candidate-pass",
        progress_stage="prepare-one-pass",
        require_all_nodes=False,
        raw_capture_path=checkpoint_root / "candidate-pass-raw.csv",
    )
    candidate_stats: dict[str, WorkloadStats] = {}
    for index, workload in enumerate(candidates):
        valid_positions = np.flatnonzero(candidate_mask[index])
        if valid_positions.size == 0 or not candidate_nodes[index]:
            continue
        candidate_stats[workload] = WorkloadStats(
            workload_id=workload,
            valid_rows=int(valid_positions.size),
            valid_time_points=int(valid_positions.size),
            min_time_seconds=(
                config.start_seconds + int(valid_positions[0]) * config.resample_seconds
            ),
            max_time_seconds=(
                config.start_seconds + int(valid_positions[-1]) * config.resample_seconds
            ),
            last_time_seconds=(
                config.start_seconds + int(valid_positions[-1]) * config.resample_seconds
            ),
        )
    workload_ids = select_workloads(candidate_stats, config)
    if raw_export_path is not None:
        raw_counts = export_selected_raw_rows(
            checkpoint_root / "candidate-pass-raw.csv",
            raw_export_path,
            workload_ids,
            delimiter=source.delimiter,
            per_workload_dir=Path(raw_export_path).parent / "raw_by_workload",
        )
        atomic_write_json(
            Path(raw_export_path).parent / "raw_schema.json",
            {
                "format": "headerless_csv",
                "columns": list(source.columns),
                "delimiter": source.delimiter,
                "workload_count": len(workload_ids),
                "row_count": int(sum(raw_counts.values())),
                "rows_by_workload": raw_counts,
                "preserves_original_rows": True,
            },
        )
    candidate_position = {workload: index for index, workload in enumerate(candidates)}
    selected_positions = np.asarray(
        [candidate_position[workload] for workload in workload_ids], dtype=np.int64
    )
    observed_values = candidate_values[selected_positions]
    target_mask = candidate_mask[selected_positions]
    workload_nodes = tuple(candidate_nodes[index] for index in selected_positions)
    if any(not node for node in workload_nodes):
        raise DatasetError("selected candidate workload has no deployment node")
    node_ids, deployment = build_deployment_matrix(workload_nodes)
    input_values, input_valid_mask, imputed_mask = bounded_causal_fill(
        observed_values, config.max_causal_fill_steps
    )
    data_index = build_data_index(config)
    cohort = build_origin_cohort(
        config, data_index, workload_ids, input_valid_mask, target_mask
    )
    if not np.any(cohort.eligible_mask):
        raise DatasetError(
            "seeded candidate cohort is empty; increase selection.candidate_pool_size"
        )
    return DenseDataset(
        source_name=f"{source.name}:seeded_metadata_candidates",
        source_report=report,
        workload_ids=workload_ids,
        workload_node_ids=workload_nodes,
        node_ids=node_ids,
        deployment=deployment,
        node_assignment_source=f"{source.name}:{source.node_column}",
        resource_names=source.resource_names,
        observed_values=observed_values,
        input_values=input_values,
        target_observed_mask=target_mask,
        input_valid_mask=input_valid_mask,
        imputed_mask=imputed_mask,
        data_index=data_index,
        cohort=cohort,
        workload_stats={workload: candidate_stats[workload] for workload in workload_ids},
    )


def build_raw_dataset_resumable(
    config: ExperimentConfig,
    source: SourceConfig,
    checkpoint_dir: str | Path,
    *,
    resume: bool,
    progress_callback: ProgressCallback | None = None,
) -> DenseDataset:
    report = preflight_source(source)
    if not report.ok:
        raise DatasetError(f"raw source failed preflight: {report}")
    checkpoint_root = Path(checkpoint_dir)
    stats = scan_stats_resumable(
        source,
        config,
        checkpoint_root,
        resume=resume,
        progress_callback=progress_callback,
    )
    workload_ids = select_workloads(stats, config)
    observed_values, target_mask, workload_nodes = extract_selected_resumable(
        source,
        config,
        workload_ids,
        checkpoint_root,
        resume=resume,
        progress_callback=progress_callback,
    )
    node_ids, deployment = build_deployment_matrix(workload_nodes)
    input_values, input_valid_mask, imputed_mask = bounded_causal_fill(
        observed_values, config.max_causal_fill_steps
    )
    data_index = build_data_index(config)
    cohort = build_origin_cohort(
        config, data_index, workload_ids, input_valid_mask, target_mask
    )
    if not np.any(cohort.eligible_mask):
        raise DatasetError("common origin cohort is empty after resumable raw preparation")
    return DenseDataset(
        source_name=source.name,
        source_report=report,
        workload_ids=workload_ids,
        workload_node_ids=workload_nodes,
        node_ids=node_ids,
        deployment=deployment,
        node_assignment_source=f"{source.name}:{source.node_column}",
        resource_names=source.resource_names,
        observed_values=observed_values,
        input_values=input_values,
        target_observed_mask=target_mask,
        input_valid_mask=input_valid_mask,
        imputed_mask=imputed_mask,
        data_index=data_index,
        cohort=cohort,
        workload_stats={item: stats[item] for item in workload_ids},
    )


__all__ = [
    "ScanState",
    "build_raw_dataset_resumable",
    "extract_selected_resumable",
    "scan_stats_resumable",
]
