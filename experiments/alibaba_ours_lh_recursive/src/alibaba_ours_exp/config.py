from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


LINK_MODES = frozenset({"topology_only", "real_link_quality"})


@dataclass(frozen=True)
class ResourceRule:
    name: str
    column: str
    scale: float = 1.0
    valid_min: float | None = None
    valid_max: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceRule":
        return cls(
            name=str(value["name"]),
            column=str(value["column"]),
            scale=float(value.get("scale", 1.0)),
            valid_min=_optional_float(value.get("valid_min")),
            valid_max=_optional_float(value.get("valid_max")),
        )

    def validate(self) -> None:
        if not self.name or not self.column:
            raise ConfigError("resource rule names and columns must not be empty")
        if not math.isfinite(self.scale):
            raise ConfigError(f"resource {self.name!r} has a non-finite scale")
        if self.valid_min is not None and self.valid_max is not None:
            if self.valid_min > self.valid_max:
                raise ConfigError(f"resource {self.name!r} has valid_min > valid_max")


@dataclass(frozen=True)
class SourceConfig:
    name: str
    format: str
    path: Path
    has_header: bool
    columns: tuple[str, ...]
    workload_column: str
    node_column: str | None
    time_column: str
    time_unit: str
    delimiter: str
    resources: tuple[ResourceRule, ...]

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        project_root: Path,
    ) -> "SourceConfig":
        raw_path = _relative_path(value["path"], f"sources.{name}.path")
        return cls(
            name=name,
            format=str(value.get("format", "csv")).lower(),
            path=(project_root / raw_path).resolve(),
            has_header=bool(value.get("has_header", True)),
            columns=tuple(str(item) for item in value.get("columns", ())),
            workload_column=str(value["workload_column"]),
            node_column=(
                None if value.get("node_column") is None else str(value["node_column"])
            ),
            time_column=str(value["time_column"]),
            time_unit=str(value.get("time_unit", "seconds")).lower(),
            delimiter=str(value.get("delimiter", ",")),
            resources=tuple(ResourceRule.from_mapping(item) for item in value["resources"]),
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        ordered = [self.workload_column, self.time_column]
        if self.node_column is not None:
            ordered.append(self.node_column)
        ordered.extend(rule.column for rule in self.resources)
        return tuple(dict.fromkeys(ordered))

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.resources)

    def validate(self) -> None:
        if self.format not in {"csv", "parquet"}:
            raise ConfigError(f"source {self.name!r} format must be csv or parquet")
        if self.time_unit not in {"seconds", "bucket"}:
            raise ConfigError(f"source {self.name!r} time_unit must be seconds or bucket")
        if self.format == "csv" and not self.has_header and not self.columns:
            raise ConfigError(f"headerless CSV source {self.name!r} requires columns")
        if len(self.delimiter) != 1:
            raise ConfigError(f"source {self.name!r} delimiter must be one character")
        if not self.resources:
            raise ConfigError(f"source {self.name!r} requires at least one resource")
        if len(set(self.resource_names)) != len(self.resource_names):
            raise ConfigError(f"source {self.name!r} has duplicate output resource names")
        for rule in self.resources:
            rule.validate()
        if self.columns:
            missing = set(self.required_columns).difference(self.columns)
            if missing:
                raise ConfigError(
                    f"source {self.name!r} configured columns omit {sorted(missing)}"
                )


@dataclass(frozen=True)
class DeploymentSourceConfig:
    name: str
    format: str
    path: Path
    required_columns: tuple[str, ...]
    workload_column: str
    node_column: str
    has_header: bool = True
    delimiter: str = ","

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        project_root: Path,
    ) -> "DeploymentSourceConfig":
        raw_path = _relative_path(value["path"], f"deployment_sources.{name}.path")
        return cls(
            name=name,
            format=str(value.get("format", "csv")).lower(),
            path=(project_root / raw_path).resolve(),
            required_columns=tuple(str(item) for item in value["required_columns"]),
            workload_column=str(value["workload_column"]),
            node_column=str(value["node_column"]),
            has_header=bool(value.get("has_header", True)),
            delimiter=str(value.get("delimiter", ",")),
        )

    def validate(self) -> None:
        if self.format not in {"csv", "parquet"}:
            raise ConfigError(f"deployment source {self.name!r} format must be csv or parquet")
        if not self.required_columns:
            raise ConfigError(f"deployment source {self.name!r} has no required columns")
        if {self.workload_column, self.node_column}.difference(self.required_columns):
            raise ConfigError(
                f"deployment source {self.name!r} required_columns omit its mapping fields"
            )
        if self.format == "csv" and not self.has_header:
            raise ConfigError("headerless deployment CSV is not supported without named columns")
        if len(self.delimiter) != 1:
            raise ConfigError(f"deployment source {self.name!r} delimiter must be one character")


@dataclass(frozen=True)
class LinkFeatureSourceConfig:
    name: str
    format: str
    path: Path
    required_columns: tuple[str, ...]
    has_header: bool = True
    delimiter: str = ","

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
        project_root: Path,
    ) -> "LinkFeatureSourceConfig":
        raw_path = _relative_path(value["path"], f"link_feature_sources.{name}.path")
        return cls(
            name=name,
            format=str(value.get("format", "csv")).lower(),
            path=(project_root / raw_path).resolve(),
            required_columns=tuple(str(item) for item in value["required_columns"]),
            has_header=bool(value.get("has_header", True)),
            delimiter=str(value.get("delimiter", ",")),
        )

    def validate(self) -> None:
        if self.format not in {"csv", "parquet"}:
            raise ConfigError(f"link feature source {self.name!r} format must be csv or parquet")
        expected = {
            "time_bucket",
            "src_node",
            "dst_node",
            "bandwidth",
            "delay",
            "loss",
            "availability",
        }
        if set(self.required_columns) != expected:
            raise ConfigError(
                f"link feature source {self.name!r} must declare the seven canonical fields"
            )
        if self.format == "csv" and not self.has_header:
            raise ConfigError("headerless link feature CSV is not supported")
        if len(self.delimiter) != 1:
            raise ConfigError(f"link feature source {self.name!r} delimiter must be one character")


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    project_root: Path
    schema_version: int
    resample_seconds: int
    start_seconds: int
    end_seconds_exclusive: int
    split_ratios: tuple[float, float, float]
    history_lengths: tuple[int, ...]
    forecast_horizons: tuple[int, ...]
    protocol: Mapping[str, Any]
    spatial: Mapping[str, Any]
    training: Mapping[str, Any]
    runtime: Mapping[str, Any]
    max_causal_fill_steps: int
    min_history_coverage: float
    max_workloads: int
    selection_strategy: str
    candidate_pool_size: int
    candidate_metadata_path: Path | None
    portable_dataset_path: Path | None
    require_exact_workload_count: bool
    origin_splits: tuple[str, ...]
    origin_stride_steps: int
    include_boundary_origin: bool
    require_all_forecast_targets_observed: bool
    source_priority: tuple[str, ...]
    sources: Mapping[str, SourceConfig]
    deployment_priority: tuple[str, ...]
    deployment_sources: Mapping[str, DeploymentSourceConfig]
    link_feature_priority: tuple[str, ...]
    link_feature_sources: Mapping[str, LinkFeatureSourceConfig]
    output_dir: Path

    @property
    def total_steps(self) -> int:
        return (self.end_seconds_exclusive - self.start_seconds) // self.resample_seconds

    @property
    def max_history(self) -> int:
        return max(self.history_lengths)

    @property
    def max_forecast_horizon(self) -> int:
        return max(self.forecast_horizons)

    @property
    def resource_names(self) -> tuple[str, ...]:
        first = self.sources[self.source_priority[0]]
        return first.resource_names

    @property
    def link_mode(self) -> str:
        return str(self.spatial["link_mode"])

    @property
    def requires_real_link_features(self) -> bool:
        return self.link_mode == "real_link_quality"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"unsupported schema_version {self.schema_version}")
        if self.resample_seconds != 60:
            raise ConfigError("this experiment requires a continuous 60-second time axis")
        if self.start_seconds < 0 or self.end_seconds_exclusive <= self.start_seconds:
            raise ConfigError("time_axis bounds are invalid")
        span = self.end_seconds_exclusive - self.start_seconds
        if span % self.resample_seconds:
            raise ConfigError("time_axis span must be divisible by resample_seconds")
        if any(ratio <= 0.0 for ratio in self.split_ratios):
            raise ConfigError("split ratios must be positive")
        if not math.isclose(sum(self.split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError("split ratios must sum to one")
        _validate_positive_unique_grid(self.history_lengths, "history_lengths")
        _validate_positive_unique_grid(self.forecast_horizons, "forecast_horizons")
        if tuple(sorted(self.history_lengths)) != self.history_lengths:
            raise ConfigError("history_lengths must be sorted ascending")
        if tuple(sorted(self.forecast_horizons)) != self.forecast_horizons:
            raise ConfigError("forecast_horizons must be sorted ascending")
        if self.max_history + self.max_forecast_horizon > self.total_steps:
            raise ConfigError("max history plus max horizon exceeds the full time axis")
        required_protocol = {
            "experiment_name",
            "forecast_strategy",
            "seed",
            "num_states",
            "max_order",
            "time_decay",
            "message_passing_steps",
            "einstein_strength",
            "backoff_blend",
            "time_period_steps",
            "prototype_length",
        }
        missing_protocol = required_protocol.difference(self.protocol)
        if missing_protocol:
            raise ConfigError(f"protocol is missing {sorted(missing_protocol)}")
        if self.protocol["forecast_strategy"] not in {"recursive", "direct_multi_step"}:
            raise ConfigError(
                "forecast_strategy must be recursive or direct_multi_step"
            )
        if "link_mode" not in self.spatial:
            raise ConfigError("spatial.link_mode is required")
        if self.link_mode not in LINK_MODES:
            raise ConfigError(
                "spatial.link_mode must be one of " + ", ".join(sorted(LINK_MODES))
            )
        if int(self.protocol["seed"]) != 7:
            raise ConfigError("this experiment must run only seed=7")
        if int(self.protocol["num_states"]) != 8:
            raise ConfigError("this experiment requires fixed K=8")
        if int(self.protocol["max_order"]) != 3:
            raise ConfigError("this experiment requires P=3")
        if int(self.protocol["message_passing_steps"]) != 2:
            raise ConfigError("this experiment requires R=2")
        if float(self.protocol["time_decay"]) != 0.02:
            raise ConfigError("this experiment requires time_decay=0.02")
        if int(self.protocol["max_order"]) > min(self.history_lengths):
            raise ConfigError("max_order must not exceed the smallest history length")
        if int(self.protocol["prototype_length"]) <= 0:
            raise ConfigError("prototype_length must be positive")
        if int(self.training.get("epochs", 0)) <= 0:
            raise ConfigError("training.epochs must be positive")
        if float(self.training.get("learning_rate", 0.0)) <= 0.0:
            raise ConfigError("training.learning_rate must be positive")
        if int(self.runtime.get("recursive_checkpoint_interval_h", 0)) <= 0:
            raise ConfigError("runtime.recursive_checkpoint_interval_h must be positive")
        if self.max_causal_fill_steps < 0:
            raise ConfigError("max_causal_fill_steps must be non-negative")
        if not 0.0 <= self.min_history_coverage <= 1.0:
            raise ConfigError("min_history_coverage must be in [0, 1]")
        if self.max_workloads <= 0:
            raise ConfigError("max_workloads must be positive")
        if self.selection_strategy not in {
            "valid_row_count",
            "lexical",
            "seeded_metadata_candidates",
        }:
            raise ConfigError(
                "selection.strategy must be valid_row_count, lexical, or "
                "seeded_metadata_candidates"
            )
        if self.selection_strategy == "seeded_metadata_candidates":
            if self.candidate_pool_size < self.max_workloads:
                raise ConfigError("selection.candidate_pool_size must be >= max_workloads")
            if self.candidate_metadata_path is None:
                raise ConfigError("selection.candidate_metadata_path is required")
        if self.portable_dataset_path is not None and self.portable_dataset_path.suffix.lower() != ".npz":
            raise ConfigError("portable_dataset_path must point to an NPZ file")
        if self.origin_stride_steps <= 0:
            raise ConfigError("origins.stride_steps must be positive")
        allowed_splits = {"train", "validation", "test"}
        if not self.origin_splits or not set(self.origin_splits).issubset(allowed_splits):
            raise ConfigError("origins.splits contains an unsupported split name")
        if not self.source_priority:
            raise ConfigError("source_priority must not be empty")
        if set(self.source_priority).difference(self.sources):
            raise ConfigError("source_priority references an undefined source")
        for source in self.sources.values():
            source.validate()
        expected_names = self.sources[self.source_priority[0]].resource_names
        for source in self.sources.values():
            if source.resource_names != expected_names:
                raise ConfigError("all data sources must expose identical ordered resource names")
        for source in self.deployment_sources.values():
            source.validate()
        if not self.deployment_priority or not self.deployment_sources:
            raise ConfigError("at least one deployment source is required")
        if set(self.deployment_priority).difference(self.deployment_sources):
            raise ConfigError("deployment_priority references an undefined source")
        for source in self.link_feature_sources.values():
            source.validate()
        if self.requires_real_link_features:
            if not self.link_feature_priority or not self.link_feature_sources:
                raise ConfigError(
                    "real_link_quality mode requires the canonical link feature source contract"
                )
        if set(self.link_feature_priority).difference(self.link_feature_sources):
            raise ConfigError("link_feature_priority references an undefined source")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    project_rel = _relative_path(raw.get("project_root", "."), "project_root")
    project_root = (config_path.parent / project_rel).resolve()

    time_axis = raw["time_axis"]
    split = raw["split"]
    grid = raw["grid"]
    missing = raw["missing"]
    selection = raw["selection"]
    origins = raw["origins"]
    sources = {
        str(name): SourceConfig.from_mapping(str(name), value, project_root)
        for name, value in raw["sources"].items()
    }
    deployment_sources = {
        str(name): DeploymentSourceConfig.from_mapping(str(name), value, project_root)
        for name, value in raw.get("deployment_sources", {}).items()
    }
    link_feature_sources = {
        str(name): LinkFeatureSourceConfig.from_mapping(str(name), value, project_root)
        for name, value in raw.get("link_feature_sources", {}).items()
    }
    output_rel = _relative_path(raw["output_dir"], "output_dir")
    candidate_metadata_value = selection.get("candidate_metadata_path")
    candidate_metadata_path = None
    if candidate_metadata_value is not None:
        candidate_metadata_rel = _relative_path(
            candidate_metadata_value,
            "selection.candidate_metadata_path",
        )
        candidate_metadata_path = (project_root / candidate_metadata_rel).resolve()
    portable_dataset_value = raw.get("portable_dataset_path")
    portable_dataset_path = None
    if portable_dataset_value is not None:
        portable_dataset_rel = _relative_path(
            portable_dataset_value,
            "portable_dataset_path",
        )
        portable_dataset_path = (project_root / portable_dataset_rel).resolve()
    cfg = ExperimentConfig(
        config_path=config_path,
        project_root=project_root,
        schema_version=int(raw.get("schema_version", 1)),
        resample_seconds=int(time_axis["resample_seconds"]),
        start_seconds=int(time_axis["start_seconds"]),
        end_seconds_exclusive=int(time_axis["end_seconds_exclusive"]),
        split_ratios=(
            float(split["train_ratio"]),
            float(split["validation_ratio"]),
            float(split["test_ratio"]),
        ),
        history_lengths=tuple(int(item) for item in grid["history_lengths"]),
        forecast_horizons=tuple(int(item) for item in grid["forecast_horizons"]),
        protocol=dict(raw["protocol"]),
        spatial=dict(raw.get("spatial", {})),
        training=dict(raw["training"]),
        runtime=dict(raw["runtime"]),
        max_causal_fill_steps=int(missing["max_causal_fill_steps"]),
        min_history_coverage=float(missing.get("min_history_coverage", 0.0)),
        max_workloads=int(selection["max_workloads"]),
        selection_strategy=str(selection.get("strategy", "valid_row_count")),
        candidate_pool_size=int(selection.get("candidate_pool_size", selection["max_workloads"])),
        candidate_metadata_path=candidate_metadata_path,
        portable_dataset_path=portable_dataset_path,
        require_exact_workload_count=bool(selection.get("require_exact_count", True)),
        origin_splits=tuple(str(item) for item in origins["splits"]),
        origin_stride_steps=int(origins["stride_steps"]),
        include_boundary_origin=bool(origins.get("include_boundary_origin", True)),
        require_all_forecast_targets_observed=bool(
            origins.get("require_all_forecast_targets_observed", False)
        ),
        source_priority=tuple(str(item) for item in raw["source_priority"]),
        sources=sources,
        deployment_priority=tuple(str(item) for item in raw.get("deployment_priority", ())),
        deployment_sources=deployment_sources,
        link_feature_priority=tuple(str(item) for item in raw.get("link_feature_priority", ())),
        link_feature_sources=link_feature_sources,
        output_dir=(project_root / output_rel).resolve(),
    )
    cfg.validate()
    return cfg


def _relative_path(value: Any, field_name: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ConfigError(f"{field_name} must be a relative path")
    return path


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _validate_positive_unique_grid(values: tuple[int, ...], name: str) -> None:
    if not values or any(item <= 0 for item in values):
        raise ConfigError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ConfigError(f"{name} must not contain duplicates")
