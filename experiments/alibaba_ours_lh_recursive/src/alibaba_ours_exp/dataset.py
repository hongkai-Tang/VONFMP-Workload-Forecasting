from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .config import (
    DeploymentSourceConfig,
    ExperimentConfig,
    LinkFeatureSourceConfig,
    SourceConfig,
)
from .data_index import DataIndex, OriginRecord, build_data_index


class DatasetError(RuntimeError):
    """Raised when an input source cannot satisfy the experiment contract."""


@dataclass(frozen=True)
class SchemaReport:
    name: str
    format: str
    path: Path
    exists: bool
    readable: bool
    columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exists and self.readable and not self.missing_columns and self.error is None


@dataclass
class WorkloadStats:
    workload_id: str
    valid_rows: int = 0
    valid_time_points: int = 0
    min_time_seconds: int | None = None
    max_time_seconds: int | None = None
    last_time_seconds: int | None = None

    def observe(self, time_seconds: int) -> None:
        self.valid_rows += 1
        # Alibaba container_usage is ordered by physical time.  Counting a
        # changed minute avoids rewarding workloads merely because they have
        # more 10-second samples inside the same 60-second bucket.
        if self.last_time_seconds != time_seconds:
            self.valid_time_points += 1
            self.last_time_seconds = time_seconds
        if self.min_time_seconds is None or time_seconds < self.min_time_seconds:
            self.min_time_seconds = time_seconds
        if self.max_time_seconds is None or time_seconds > self.max_time_seconds:
            self.max_time_seconds = time_seconds

    @property
    def observed_span_seconds(self) -> int:
        if self.min_time_seconds is None or self.max_time_seconds is None:
            return 0
        return self.max_time_seconds - self.min_time_seconds


@dataclass(frozen=True)
class OriginCohort:
    origin_ids: tuple[str, ...]
    workload_ids: tuple[str, ...]
    h_values: tuple[int, ...]
    eligible_mask: np.ndarray
    history_coverage: np.ndarray
    target_observed_mask: np.ndarray

    def eligible_workloads(self, origin_id: str) -> tuple[str, ...]:
        try:
            origin_position = self.origin_ids.index(origin_id)
        except ValueError as exc:
            raise KeyError(origin_id) from exc
        return tuple(
            workload_id
            for workload_id, keep in zip(
                self.workload_ids,
                self.eligible_mask[origin_position],
            )
            if bool(keep)
        )

    def target_mask(self, origin_id: str, h: int) -> np.ndarray:
        try:
            origin_position = self.origin_ids.index(origin_id)
            h_position = self.h_values.index(int(h))
        except ValueError as exc:
            raise KeyError((origin_id, h)) from exc
        return self.target_observed_mask[origin_position, :, h_position].copy()


@dataclass(frozen=True)
class DenseDataset:
    source_name: str
    source_report: SchemaReport
    workload_ids: tuple[str, ...]
    workload_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    deployment: np.ndarray
    node_assignment_source: str
    resource_names: tuple[str, ...]
    observed_values: np.ndarray
    input_values: np.ndarray
    target_observed_mask: np.ndarray
    input_valid_mask: np.ndarray
    imputed_mask: np.ndarray
    data_index: DataIndex
    cohort: OriginCohort
    workload_stats: Mapping[str, WorkloadStats]

    def input_history(self, workload_position: int, origin: OriginRecord, history_length: int) -> np.ndarray:
        history_length = int(history_length)
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        start = origin.origin_index - history_length + 1
        if start < 0:
            raise IndexError("history begins before the continuous time axis")
        return self.input_values[workload_position, start : origin.origin_index + 1].copy()

    def target(self, workload_position: int, origin: OriginRecord, h: int) -> tuple[np.ndarray, bool]:
        target_index = origin.target_index(h)
        return (
            self.observed_values[workload_position, target_index].copy(),
            bool(self.target_observed_mask[workload_position, target_index]),
        )


def preflight_source(source: SourceConfig) -> SchemaReport:
    if not source.path.exists():
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=False,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error="file does not exist",
        )
    try:
        columns = _source_columns(source)
    except Exception as exc:
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=True,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error=str(exc),
        )
    missing = tuple(sorted(set(source.required_columns).difference(columns)))
    return SchemaReport(
        name=source.name,
        format=source.format,
        path=source.path,
        exists=True,
        readable=True,
        columns=columns,
        required_columns=source.required_columns,
        missing_columns=missing,
        error=None,
    )


def preflight_deployment_source(source: DeploymentSourceConfig) -> SchemaReport:
    if not source.path.exists():
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=False,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error="file does not exist",
        )
    try:
        if source.format == "csv":
            with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=source.delimiter)
                columns = tuple(next(reader))
        else:
            columns = _parquet_columns(source.path)
    except Exception as exc:
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=True,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error=str(exc),
        )
    missing = tuple(sorted(set(source.required_columns).difference(columns)))
    return SchemaReport(
        name=source.name,
        format=source.format,
        path=source.path,
        exists=True,
        readable=True,
        columns=columns,
        required_columns=source.required_columns,
        missing_columns=missing,
        error=None,
    )


def preflight_link_feature_source(source: LinkFeatureSourceConfig) -> SchemaReport:
    if not source.path.exists():
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=False,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error="file does not exist",
        )
    try:
        if source.format == "csv":
            with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=source.delimiter)
                columns = tuple(next(reader))
        else:
            columns = _parquet_columns(source.path)
    except Exception as exc:
        return SchemaReport(
            name=source.name,
            format=source.format,
            path=source.path,
            exists=True,
            readable=False,
            columns=(),
            required_columns=source.required_columns,
            missing_columns=source.required_columns,
            error=str(exc),
        )
    missing = tuple(sorted(set(source.required_columns).difference(columns)))
    return SchemaReport(
        name=source.name,
        format=source.format,
        path=source.path,
        exists=True,
        readable=True,
        columns=columns,
        required_columns=source.required_columns,
        missing_columns=missing,
        error=None,
    )


def preflight_configured_sources(config: ExperimentConfig) -> dict[str, SchemaReport]:
    reports = {name: preflight_source(source) for name, source in config.sources.items()}
    reports.update(
        {
            f"deployment_{name}": preflight_deployment_source(source)
            for name, source in config.deployment_sources.items()
        }
    )
    reports.update(
        {
            f"link_features_{name}": preflight_link_feature_source(source)
            for name, source in config.link_feature_sources.items()
        }
    )
    return reports


def resolve_source(config: ExperimentConfig) -> tuple[SourceConfig, SchemaReport]:
    failures: list[str] = []
    for name in config.source_priority:
        source = config.sources[name]
        report = preflight_source(source)
        if report.ok:
            return source, report
        failures.append(f"{name}: {report.error or report.missing_columns}")
    raise DatasetError("no configured data source passed schema preflight; " + "; ".join(failures))


def resolve_deployment_source(
    config: ExperimentConfig,
) -> tuple[DeploymentSourceConfig, SchemaReport]:
    failures: list[str] = []
    for name in config.deployment_priority:
        source = config.deployment_sources[name]
        report = preflight_deployment_source(source)
        if report.ok:
            return source, report
        failures.append(f"{name}: {report.error or report.missing_columns}")
    raise DatasetError(
        "no deployment source passed schema preflight; " + "; ".join(failures)
    )


def resolve_link_feature_source(
    config: ExperimentConfig,
) -> tuple[LinkFeatureSourceConfig, SchemaReport] | None:
    for name in config.link_feature_priority:
        source = config.link_feature_sources[name]
        report = preflight_link_feature_source(source)
        if report.ok:
            return source, report
    return None


def scan_workload_stats(
    source: SourceConfig,
    config: ExperimentConfig,
) -> dict[str, WorkloadStats]:
    """Stage one: stream a source and retain only per-workload selection statistics."""

    report = preflight_source(source)
    if not report.ok:
        raise DatasetError(f"source {source.name!r} failed preflight: {report}")
    stats: dict[str, WorkloadStats] = {}
    for row in iter_source_rows(source):
        parsed = _parse_row(row, source, config)
        if parsed is None:
            continue
        workload_id, time_seconds, _ = parsed
        item = stats.get(workload_id)
        if item is None:
            item = WorkloadStats(workload_id=workload_id)
            stats[workload_id] = item
        item.observe(time_seconds)
    return stats


def select_workloads(
    stats: Mapping[str, WorkloadStats],
    config: ExperimentConfig,
) -> tuple[str, ...]:
    candidates = list(stats.values())
    if config.selection_strategy == "lexical":
        candidates.sort(key=lambda item: item.workload_id)
    else:
        candidates.sort(
            key=lambda item: (
                -item.valid_time_points,
                -item.valid_rows,
                -item.observed_span_seconds,
                item.workload_id,
            )
        )
    selected = tuple(item.workload_id for item in candidates[: config.max_workloads])
    if config.require_exact_workload_count and len(selected) != config.max_workloads:
        raise DatasetError(
            f"requested {config.max_workloads} workloads but only {len(selected)} are valid"
        )
    return selected


def load_selected_source(
    source: SourceConfig,
    config: ExperimentConfig,
    workload_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str | None, ...]]:
    """Stage two: rescan only selected workloads and aggregate them onto the minute grid."""

    selected = tuple(str(item) for item in workload_ids)
    workload_position = {item: position for position, item in enumerate(selected)}
    sums = np.zeros(
        (len(selected), config.total_steps, len(source.resources)),
        dtype=np.float64,
    )
    counts = np.zeros((len(selected), config.total_steps), dtype=np.int32)
    node_counts: list[dict[str, int]] = [dict() for _ in selected]
    for row in iter_source_rows(source):
        raw_workload = row.get(source.workload_column)
        if raw_workload is None:
            continue
        workload_id = str(raw_workload).strip()
        position = workload_position.get(workload_id)
        if position is None:
            continue
        parsed = _parse_row(row, source, config)
        if parsed is None:
            continue
        _, time_seconds, values = parsed
        time_position = (time_seconds - config.start_seconds) // config.resample_seconds
        sums[position, time_position] += values
        counts[position, time_position] += 1
        if source.node_column is not None:
            raw_node = row.get(source.node_column)
            if raw_node not in (None, ""):
                node_id = str(raw_node).strip()
                if node_id:
                    node_counts[position][node_id] = node_counts[position].get(node_id, 0) + 1

    target_observed_mask = counts > 0
    values = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(
        sums,
        np.maximum(counts[..., None], 1),
        out=values,
        where=target_observed_mask[..., None],
    )
    source_node_ids: list[str | None] = []
    for counts_by_node in node_counts:
        if not counts_by_node:
            source_node_ids.append(None)
            continue
        node_id = min(
            counts_by_node,
            key=lambda item: (-counts_by_node[item], item),
        )
        source_node_ids.append(node_id)
    return values, target_observed_mask, tuple(source_node_ids)


def load_deployment_nodes(
    source: DeploymentSourceConfig,
    workload_ids: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(str(item) for item in workload_ids)
    selected_set = set(selected)
    assignments: dict[str, str] = {}
    for row in _iter_named_rows(
        source.path,
        source.format,
        source.required_columns,
        source.delimiter,
    ):
        raw_workload = row.get(source.workload_column)
        raw_node = row.get(source.node_column)
        if raw_workload in (None, "") or raw_node in (None, ""):
            continue
        workload_id = str(raw_workload).strip()
        node_id = str(raw_node).strip()
        if workload_id in selected_set and node_id:
            assignments.setdefault(workload_id, node_id)
    missing = [item for item in selected if item not in assignments]
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetError(
            f"deployment source has no node for {len(missing)} selected workloads: {preview}"
        )
    return tuple(assignments[item] for item in selected)


def build_deployment_matrix(
    workload_node_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    node_ids = tuple(sorted(set(str(item) for item in workload_node_ids)))
    if not node_ids:
        raise DatasetError("no node assignments are available")
    node_position = {item: position for position, item in enumerate(node_ids)}
    deployment = np.zeros((len(node_ids), len(workload_node_ids)), dtype=np.float32)
    for workload_position, node_id in enumerate(workload_node_ids):
        deployment[node_position[str(node_id)], workload_position] = 1.0
    return node_ids, deployment


def bounded_causal_fill(
    values: np.ndarray,
    max_causal_fill_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-fill at most the next configured minutes without consulting future values."""

    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("values must have shape (workloads, time, resources)")
    filled = data.copy()
    imputed_cells = np.zeros(data.shape, dtype=bool)
    if max_causal_fill_steps > 0:
        for workload_position in range(data.shape[0]):
            for resource_position in range(data.shape[2]):
                last_value = float("nan")
                missing_age = max_causal_fill_steps + 1
                for time_position in range(data.shape[1]):
                    current = float(data[workload_position, time_position, resource_position])
                    if math.isfinite(current):
                        last_value = current
                        missing_age = 0
                        continue
                    missing_age += 1
                    if math.isfinite(last_value) and missing_age <= max_causal_fill_steps:
                        filled[workload_position, time_position, resource_position] = last_value
                        imputed_cells[workload_position, time_position, resource_position] = True
    input_valid_mask = np.isfinite(filled).all(axis=-1)
    imputed_mask = imputed_cells.any(axis=-1)
    return filled, input_valid_mask, imputed_mask


def build_origin_cohort(
    config: ExperimentConfig,
    data_index: DataIndex,
    workload_ids: Sequence[str],
    input_valid_mask: np.ndarray,
    target_observed_mask: np.ndarray,
) -> OriginCohort:
    origins = data_index.origins
    workload_count = len(workload_ids)
    history_coverage = np.zeros((len(origins), workload_count), dtype=np.float32)
    target_mask = np.zeros(
        (len(origins), workload_count, len(config.forecast_horizons)),
        dtype=bool,
    )
    eligible = np.zeros((len(origins), workload_count), dtype=bool)
    for origin_position, origin in enumerate(origins):
        history = input_valid_mask[
            :,
            origin.history_start_index : origin.origin_index + 1,
        ]
        coverage = history.mean(axis=1, dtype=np.float64)
        history_coverage[origin_position] = coverage.astype(np.float32)
        target_indices = np.asarray(origin.target_indices, dtype=int)
        target_mask[origin_position] = target_observed_mask[:, target_indices]
        keep = coverage >= config.min_history_coverage
        keep &= input_valid_mask[:, origin.origin_index]
        if config.require_all_forecast_targets_observed:
            keep &= target_mask[origin_position].all(axis=1)
        eligible[origin_position] = keep
    return OriginCohort(
        origin_ids=tuple(item.origin_id for item in origins),
        workload_ids=tuple(str(item) for item in workload_ids),
        h_values=config.forecast_horizons,
        eligible_mask=eligible,
        history_coverage=history_coverage,
        target_observed_mask=target_mask,
    )


def build_dataset(
    config: ExperimentConfig,
    source_name: str | None = None,
) -> DenseDataset:
    if source_name is None:
        source, report = resolve_source(config)
    else:
        try:
            source = config.sources[source_name]
        except KeyError as exc:
            raise DatasetError(f"unknown source {source_name!r}") from exc
        report = preflight_source(source)
        if not report.ok:
            raise DatasetError(f"source {source_name!r} failed schema preflight: {report}")

    stats = scan_workload_stats(source, config)
    workload_ids = select_workloads(stats, config)
    observed_values, target_observed_mask, source_node_ids = load_selected_source(
        source,
        config,
        workload_ids,
    )
    if all(item is not None for item in source_node_ids):
        workload_node_ids = tuple(str(item) for item in source_node_ids)
        node_assignment_source = f"{source.name}:{source.node_column}"
    else:
        deployment_source, _ = resolve_deployment_source(config)
        workload_node_ids = load_deployment_nodes(deployment_source, workload_ids)
        node_assignment_source = f"deployment:{deployment_source.name}"
    node_ids, deployment = build_deployment_matrix(workload_node_ids)
    input_values, input_valid_mask, imputed_mask = bounded_causal_fill(
        observed_values,
        config.max_causal_fill_steps,
    )
    data_index = build_data_index(config)
    cohort = build_origin_cohort(
        config,
        data_index,
        workload_ids,
        input_valid_mask,
        target_observed_mask,
    )
    if not np.any(cohort.eligible_mask):
        raise DatasetError(
            "common origin cohort is empty; use the complete raw source or revise only "
            "the documented coverage policy before running the experiment"
        )
    selected_stats = {item: stats[item] for item in workload_ids}
    return DenseDataset(
        source_name=source.name,
        source_report=report,
        workload_ids=workload_ids,
        workload_node_ids=workload_node_ids,
        node_ids=node_ids,
        deployment=deployment,
        node_assignment_source=node_assignment_source,
        resource_names=source.resource_names,
        observed_values=observed_values,
        input_values=input_values,
        target_observed_mask=target_observed_mask,
        input_valid_mask=input_valid_mask,
        imputed_mask=imputed_mask,
        data_index=data_index,
        cohort=cohort,
        workload_stats=selected_stats,
    )


def iter_source_rows(source: SourceConfig) -> Iterator[Mapping[str, Any]]:
    if source.format == "csv":
        yield from _iter_csv_rows(source)
    elif source.format == "parquet":
        yield from _iter_parquet_rows(source)
    else:
        raise DatasetError(f"unsupported source format {source.format!r}")


def _iter_csv_rows(source: SourceConfig) -> Iterator[Mapping[str, str]]:
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        if source.has_header:
            reader = csv.DictReader(handle, delimiter=source.delimiter)
            for row in reader:
                yield row
        else:
            reader = csv.reader(handle, delimiter=source.delimiter)
            for values in reader:
                if not values:
                    continue
                if len(values) < len(source.columns):
                    continue
                yield dict(zip(source.columns, values))


def _iter_parquet_rows(source: SourceConfig) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetError("Parquet input requires optional dependency pyarrow") from exc
    parquet_file = parquet.ParquetFile(source.path)
    columns = list(source.required_columns)
    for batch in parquet_file.iter_batches(batch_size=65536, columns=columns):
        data = batch.to_pydict()
        for position in range(batch.num_rows):
            yield {name: data[name][position] for name in columns}


def _iter_named_rows(
    path: Path,
    format_name: str,
    columns: Sequence[str],
    delimiter: str,
) -> Iterator[Mapping[str, Any]]:
    if format_name == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                yield row
        return
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetError("Parquet input requires optional dependency pyarrow") from exc
    parquet_file = parquet.ParquetFile(path)
    selected_columns = list(columns)
    for batch in parquet_file.iter_batches(batch_size=65536, columns=selected_columns):
        data = batch.to_pydict()
        for position in range(batch.num_rows):
            yield {name: data[name][position] for name in selected_columns}


def _source_columns(source: SourceConfig) -> tuple[str, ...]:
    if source.format == "parquet":
        return _parquet_columns(source.path)
    if not source.has_header:
        return source.columns
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=source.delimiter)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise DatasetError("CSV source is empty") from exc


def _parquet_columns(path: Path) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetError("Parquet schema preflight requires optional dependency pyarrow") from exc
    return tuple(parquet.ParquetFile(path).schema_arrow.names)


def _parse_row(
    row: Mapping[str, Any],
    source: SourceConfig,
    config: ExperimentConfig,
) -> tuple[str, int, np.ndarray] | None:
    raw_workload = row.get(source.workload_column)
    raw_time = row.get(source.time_column)
    if raw_workload in (None, "") or raw_time in (None, ""):
        return None
    workload_id = str(raw_workload).strip()
    if not workload_id:
        return None
    try:
        source_time = float(raw_time)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(source_time):
        return None
    if source.time_unit == "bucket":
        time_seconds = int(math.floor(source_time)) * config.resample_seconds
    else:
        time_seconds = int(math.floor(source_time / config.resample_seconds)) * config.resample_seconds
    if not config.start_seconds <= time_seconds < config.end_seconds_exclusive:
        return None

    parsed_columns: dict[str, float] = {}
    output = np.empty(len(source.resources), dtype=np.float64)
    for position, rule in enumerate(source.resources):
        if rule.column not in parsed_columns:
            try:
                raw_value = float(row.get(rule.column))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(raw_value):
                return None
            parsed_columns[rule.column] = raw_value
        raw_value = parsed_columns[rule.column]
        if rule.valid_min is not None and raw_value < rule.valid_min:
            return None
        if rule.valid_max is not None and raw_value > rule.valid_max:
            return None
        output[position] = raw_value * rule.scale
    return workload_id, time_seconds, output
