from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    steps: int
    display_unit: str
    display_divisor: float


@dataclass(frozen=True)
class TaskSpec:
    history_length: int
    horizon: HorizonSpec

    @property
    def task_id(self) -> str:
        return f"L{self.history_length}-{self.horizon.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "history_length": self.history_length,
            "horizon_name": self.horizon.name,
            "horizon_steps": self.horizon.steps,
            "display_unit": self.horizon.display_unit,
            "display_divisor": self.horizon.display_divisor,
        }


class ExperimentConfig:
    def __init__(self, path: Path, raw: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.root = self.path.parent.parent.resolve()
        self.raw = dict(raw)
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).resolve()
        if not source.is_file():
            raise ConfigError(f"配置文件不存在: {source}")
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ConfigError("配置文件根节点必须是对象")
        return cls(source, raw)

    def _validate(self) -> None:
        if int(self.raw.get("schema_version", 0)) != 1:
            raise ConfigError("仅支持 schema_version=1")
        if self.forecast_strategy != "direct_multi_step":
            raise ConfigError("forecast_strategy 必须是 direct_multi_step")
        if self.time_step_seconds != 60:
            raise ConfigError("本实验固定 time_step_seconds=60")
        if self.num_states <= 1:
            raise ConfigError("num_states 必须大于1")
        lengths = self.history_lengths
        if len(lengths) != len(set(lengths)) or any(value <= 0 for value in lengths):
            raise ConfigError("history_lengths 必须是互不重复的正整数")
        if self.priority_history_length not in lengths:
            raise ConfigError("priority_history_length 必须包含在 history_lengths 中")
        remaining = self.remaining_history_order
        expected = [value for value in lengths if value != self.priority_history_length]
        if set(remaining) != set(expected) or len(remaining) != len(expected):
            raise ConfigError("remaining_history_order 必须恰好包含其余L")
        horizons = self.horizons
        if [item.name for item in horizons] != ["long_1p2d", "medium_8h", "short_60m"]:
            raise ConfigError("horizons 顺序必须是 long_1p2d, medium_8h, short_60m")
        if [item.steps for item in horizons] != [1728, 480, 60]:
            raise ConfigError("三个H必须分别为1728、480、60")
        model = self.model
        if int(model.get("max_order", -1)) != 3:
            raise ConfigError("为保持原Ours方法，本实验固定max_order(P)=3")
        if abs(float(model.get("time_decay", -1.0)) - 0.02) > 1e-12:
            raise ConfigError("为保持原Ours方法，本实验固定time_decay(lambda)=0.02")
        if int(model.get("message_passing_steps", -1)) != 2:
            raise ConfigError("为保持原Ours方法，本实验固定message_passing_steps(R)=2")
        if int(model["max_order"]) > min(lengths):
            raise ConfigError("max_order不能超过最短历史窗口")
        if not 0.0 <= float(model.get("backoff_blend", -1.0)) <= 1.0:
            raise ConfigError("backoff_blend必须位于[0,1]")
        ratios = self.split_ratios
        if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ConfigError("train/validation/test 比例之和必须为1")

    def resolve(self, value: str | Path) -> Path:
        result = Path(value)
        return result.resolve() if result.is_absolute() else (self.root / result).resolve()

    @property
    def dataset_path(self) -> Path:
        return self.resolve(str(self.raw["dataset_path"]))

    @property
    def output_dir(self) -> Path:
        return self.resolve(str(self.raw.get("output_dir", "runs")))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 7))

    @property
    def time_step_seconds(self) -> int:
        return int(self.raw.get("time_step_seconds", 60))

    @property
    def expected_workloads(self) -> int:
        return int(self.raw.get("expected_workloads", 200))

    @property
    def forecast_strategy(self) -> str:
        return str(self.raw.get("protocol", {}).get("forecast_strategy", ""))

    @property
    def num_states(self) -> int:
        return int(self.raw.get("protocol", {}).get("num_states", 8))

    @property
    def history_lengths(self) -> list[int]:
        return [int(value) for value in self.raw["protocol"]["history_lengths"]]

    @property
    def priority_history_length(self) -> int:
        return int(self.raw["protocol"]["priority_history_length"])

    @property
    def remaining_history_order(self) -> list[int]:
        return [int(value) for value in self.raw["protocol"]["remaining_history_order"]]

    @property
    def horizons(self) -> list[HorizonSpec]:
        return [
            HorizonSpec(
                name=str(item["name"]),
                steps=int(item["steps"]),
                display_unit=str(item["display_unit"]),
                display_divisor=float(item["display_divisor"]),
            )
            for item in self.raw["protocol"]["horizons"]
        ]

    @property
    def split_ratios(self) -> tuple[float, float, float]:
        split = self.raw["split"]
        return (
            float(split["train_ratio"]),
            float(split["validation_ratio"]),
            float(split["test_ratio"]),
        )

    @property
    def membership(self) -> dict[str, Any]:
        return dict(self.raw["membership"])

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.raw["model"])

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw["training"])

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.raw["evaluation"])

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.raw["runtime"])

    def task_queue(self) -> list[TaskSpec]:
        horizon_by_name = {item.name: item for item in self.horizons}
        ordered_horizons = [
            horizon_by_name["long_1p2d"],
            horizon_by_name["medium_8h"],
            horizon_by_name["short_60m"],
        ]
        tasks = [TaskSpec(self.priority_history_length, horizon) for horizon in ordered_horizons]
        for horizon in ordered_horizons:
            tasks.extend(TaskSpec(length, horizon) for length in self.remaining_history_order)
        if len(tasks) != 24 or len({task.task_id for task in tasks}) != 24:
            raise ConfigError("正式任务队列必须包含24个唯一任务")
        return tasks

    def experiment_signature(self) -> str:
        relevant = {
            "schema_version": self.raw["schema_version"],
            # Keep the configured relative path in the compatibility signature.
            # A D: -> H: move must not invalidate otherwise identical checkpoints.
            "dataset_path": str(self.raw["dataset_path"]),
            "seed": self.seed,
            "time_step_seconds": self.time_step_seconds,
            "split": self.raw["split"],
            "protocol": self.raw["protocol"],
            "membership": self.raw["membership"],
            "model": self.raw["model"],
            "training": self.raw["training"],
            "evaluation": self.raw["evaluation"],
        }
        payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = ["ConfigError", "ExperimentConfig", "HorizonSpec", "TaskSpec"]
