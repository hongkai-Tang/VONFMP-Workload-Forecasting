from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import ConfigError, ExperimentConfig


@dataclass(frozen=True)
class TimeSplit:
    name: str
    start_index: int
    stop_index: int

    @property
    def size(self) -> int:
        return self.stop_index - self.start_index

    def as_slice(self) -> slice:
        return slice(self.start_index, self.stop_index)


@dataclass(frozen=True)
class OriginRecord:
    origin_id: str
    split: str
    origin_index: int
    origin_seconds: int
    history_start_index: int
    forecast_end_index: int
    forecast_end_seconds: int
    h_values: tuple[int, ...]
    target_indices: tuple[int, ...]

    def target_index(self, h: int) -> int:
        try:
            position = self.h_values.index(int(h))
        except ValueError as exc:
            raise KeyError(f"h={h} is not present in this origin record") from exc
        return self.target_indices[position]


@dataclass(frozen=True)
class DataIndex:
    time_seconds: np.ndarray
    splits: tuple[TimeSplit, ...]
    origins: tuple[OriginRecord, ...]

    def split(self, name: str) -> TimeSplit:
        for item in self.splits:
            if item.name == name:
                return item
        raise KeyError(f"unknown split {name!r}")

    def origins_for_split(self, name: str) -> tuple[OriginRecord, ...]:
        return tuple(item for item in self.origins if item.split == name)

    def origin_rows(self) -> Iterable[dict[str, int | str]]:
        for item in self.origins:
            yield {
                "origin_id": item.origin_id,
                "split": item.split,
                "origin_index": item.origin_index,
                "origin_seconds": item.origin_seconds,
                "history_start_index": item.history_start_index,
                "forecast_end_index": item.forecast_end_index,
                "forecast_end_seconds": item.forecast_end_seconds,
            }


def build_time_axis(config: ExperimentConfig) -> np.ndarray:
    axis = np.arange(
        config.start_seconds,
        config.end_seconds_exclusive,
        config.resample_seconds,
        dtype=np.int64,
    )
    if len(axis) != config.total_steps:
        raise ConfigError("continuous time axis has an unexpected size")
    if len(axis) > 1 and not np.all(np.diff(axis) == config.resample_seconds):
        raise ConfigError("time axis is not continuous at the configured interval")
    return axis


def build_time_splits(config: ExperimentConfig) -> tuple[TimeSplit, ...]:
    total = config.total_steps
    train_size = int(total * config.split_ratios[0])
    validation_size = int(total * config.split_ratios[1])
    test_size = total - train_size - validation_size
    if min(train_size, validation_size, test_size) <= 0:
        raise ConfigError("time split produced an empty interval")
    splits = (
        TimeSplit("train", 0, train_size),
        TimeSplit("validation", train_size, train_size + validation_size),
        TimeSplit("test", train_size + validation_size, total),
    )
    for item in splits:
        if item.name in config.origin_splits and item.size < config.max_forecast_horizon:
            raise ConfigError(
                f"split {item.name!r} is shorter than max_forecast_horizon"
            )
    return splits


def build_common_origins(
    config: ExperimentConfig,
    time_seconds: np.ndarray,
    splits: tuple[TimeSplit, ...],
) -> tuple[OriginRecord, ...]:
    """Build one origin grid shared by every configured history length."""

    records: list[OriginRecord] = []
    split_map = {item.name: item for item in splits}
    for split_name in config.origin_splits:
        item = split_map[split_name]
        first_origin = item.start_index - 1 if config.include_boundary_origin else item.start_index
        first_origin = max(first_origin, config.max_history - 1)
        last_origin = item.stop_index - config.max_forecast_horizon - 1
        if first_origin > last_origin:
            raise ConfigError(
                f"split {split_name!r} has no origin supporting max history and forecast_horizon"
            )
        for origin_index in range(
            first_origin,
            last_origin + 1,
            config.origin_stride_steps,
        ):
            target_indices = tuple(origin_index + h for h in config.forecast_horizons)
            if target_indices[-1] >= item.stop_index:
                raise ConfigError("origin target escaped its split")
            origin_seconds = int(time_seconds[origin_index])
            forecast_end_index = target_indices[-1]
            records.append(
                OriginRecord(
                    origin_id=f"{split_name}-{origin_seconds}",
                    split=split_name,
                    origin_index=origin_index,
                    origin_seconds=origin_seconds,
                    history_start_index=origin_index - config.max_history + 1,
                    forecast_end_index=forecast_end_index,
                    forecast_end_seconds=int(time_seconds[forecast_end_index]),
                    h_values=config.forecast_horizons,
                    target_indices=target_indices,
                )
            )
    return tuple(records)


def build_data_index(config: ExperimentConfig) -> DataIndex:
    time_seconds = build_time_axis(config)
    splits = build_time_splits(config)
    origins = build_common_origins(config, time_seconds, splits)
    return DataIndex(time_seconds=time_seconds, splits=splits, origins=origins)
