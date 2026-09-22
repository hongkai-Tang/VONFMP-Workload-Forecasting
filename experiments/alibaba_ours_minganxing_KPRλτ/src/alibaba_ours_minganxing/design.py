from __future__ import annotations

"""Deterministic construction of the K/P/R/lambda/tau sensitivity suite."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


STUDIES = ("k", "p", "r", "lambda", "tau")


def _decimal_text(value: float) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"


def _decimal_slug(value: float) -> str:
    return _decimal_text(value).replace("-", "m").replace(".", "p")


def _tau_slug(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


@dataclass(frozen=True)
class Condition:
    condition_id: str
    study: str
    num_states: int
    max_order: int
    message_passing_steps: int
    time_decay_per_minute: float
    granularity_seconds: int
    history_duration_seconds: int = 86_400
    forecast_horizon: int = 1
    seed: int = 7
    is_shared_baseline: bool = False

    @property
    def history_length(self) -> int:
        return self.history_duration_seconds // self.granularity_seconds

    @property
    def time_period_steps(self) -> int:
        return 86_400 // self.granularity_seconds

    @property
    def origin_stride_steps(self) -> int:
        """Evaluate at least once per physical hour without skipping coarse buckets."""

        return max(1, 3600 // self.granularity_seconds)

    @property
    def effective_time_decay_per_step(self) -> float:
        """Convert the physical per-minute lambda to the current bucket unit."""

        return self.time_decay_per_minute * self.granularity_seconds / 60.0

    @property
    def signature(self) -> tuple[int, int, int, float, int, int, int, int]:
        return (
            self.num_states,
            self.max_order,
            self.message_passing_steps,
            round(self.time_decay_per_minute, 12),
            self.granularity_seconds,
            self.history_duration_seconds,
            self.forecast_horizon,
            self.seed,
        )

    @property
    def parameter_name(self) -> str:
        return {
            "baseline": "baseline",
            "k": "num_states",
            "p": "max_order",
            "r": "message_passing_steps",
            "lambda": "time_decay_per_minute",
            "tau": "granularity_seconds",
        }[self.study]

    @property
    def parameter_value(self) -> str:
        if self.study == "k":
            return str(self.num_states)
        if self.study == "p":
            return str(self.max_order)
        if self.study == "r":
            return str(self.message_passing_steps)
        if self.study == "lambda":
            return _decimal_text(self.time_decay_per_minute)
        if self.study == "tau":
            return str(self.granularity_seconds)
        return "baseline"

    def metadata(self) -> dict[str, Any]:
        return {
            "sensitivity_condition_id": self.condition_id,
            "sensitivity_study": self.study,
            "sensitivity_parameter": self.parameter_name,
            "sensitivity_value": self.parameter_value,
            "sensitivity_shared_baseline": self.is_shared_baseline,
            "sensitivity_num_states": self.num_states,
            "sensitivity_max_order": self.max_order,
            "sensitivity_message_passing_steps": self.message_passing_steps,
            "sensitivity_time_decay_per_minute": self.time_decay_per_minute,
            "sensitivity_effective_time_decay_per_step": self.effective_time_decay_per_step,
            "sensitivity_granularity_seconds": self.granularity_seconds,
            "sensitivity_history_duration_seconds": self.history_duration_seconds,
            "sensitivity_history_length": self.history_length,
            "sensitivity_forecast_horizon_steps": self.forecast_horizon,
            "sensitivity_forecast_duration_seconds": (
                self.forecast_horizon * self.granularity_seconds
            ),
            "sensitivity_origin_stride_steps": self.origin_stride_steps,
            "sensitivity_origin_stride_seconds": (
                self.origin_stride_steps * self.granularity_seconds
            ),
            "sensitivity_seed": self.seed,
            "membership_definition": "single_time_bucket_resource_vector",
            "prediction_strategy": "direct_one_step_no_feedback",
            "automatic_coverage_selection": False,
            "fixed_k_sensitivity": True,
        }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _positive_ints(values: Iterable[Any], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _nonnegative_floats(values: Iterable[Any], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative values")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _baseline(raw: Mapping[str, Any]) -> Condition:
    suite = raw["sensitivity_suite"]
    values = suite["baseline"]
    condition = Condition(
        condition_id="baseline-k08-p03-r02-lambda0p02-tau60s-l1440-h1",
        study="baseline",
        num_states=int(values["num_states"]),
        max_order=int(values["max_order"]),
        message_passing_steps=int(values["message_passing_steps"]),
        time_decay_per_minute=float(values["time_decay_per_minute"]),
        granularity_seconds=int(values["granularity_seconds"]),
        history_duration_seconds=int(values["history_duration_seconds"]),
        forecast_horizon=int(values["forecast_horizon_steps"]),
        seed=int(suite["execution"]["seed"]),
        is_shared_baseline=True,
    )
    _validate_condition(condition)
    return condition


def _validate_condition(condition: Condition) -> None:
    if condition.study not in ("baseline", *STUDIES):
        raise ValueError(f"unknown study {condition.study!r}")
    if condition.num_states <= 1:
        raise ValueError(f"{condition.condition_id}: K must be greater than one")
    if condition.max_order <= 0 or condition.max_order > condition.history_length:
        raise ValueError(
            f"{condition.condition_id}: P={condition.max_order} must be in "
            f"[1, history_steps={condition.history_length}]"
        )
    if condition.message_passing_steps <= 0:
        raise ValueError(f"{condition.condition_id}: R must be positive")
    if not math.isfinite(condition.time_decay_per_minute) or condition.time_decay_per_minute < 0:
        raise ValueError(f"{condition.condition_id}: lambda must be finite and non-negative")
    if condition.granularity_seconds not in {60, 1800, 3600, 7200}:
        raise ValueError(f"{condition.condition_id}: unsupported tau")
    if condition.history_duration_seconds != 86_400:
        raise ValueError("this suite fixes the physical history duration to 24 hours")
    if condition.history_duration_seconds % condition.granularity_seconds:
        raise ValueError("history duration must be divisible by tau")
    if condition.forecast_horizon != 1:
        raise ValueError("this suite is strictly direct next-bucket prediction")
    if condition.seed < 0:
        raise ValueError("the experiment seed must be a non-negative integer")


def build_conditions(
    config_path: str | Path,
    studies: Iterable[str] | None = None,
    *,
    include_baseline: bool = True,
) -> list[Condition]:
    raw = _read_json(Path(config_path).resolve())
    suite = raw.get("sensitivity_suite")
    if not isinstance(suite, Mapping):
        raise ValueError("configuration is missing sensitivity_suite")
    if int(suite.get("repeat_count", 1)) != 1:
        raise ValueError("the formal grid uses one deterministic run per condition")
    configured_seed = int(suite["execution"]["seed"])
    protocol_seed = int(raw["protocol"]["seed"])
    if protocol_seed != configured_seed:
        raise ValueError(
            "protocol.seed must match sensitivity_suite.execution.seed"
        )
    if not bool(suite.get("fixed_k", False)):
        raise ValueError("K must be manually fixed for each K condition")
    if bool(suite.get("automatic_coverage_selection", True)):
        raise ValueError("automatic coverage selection must be disabled")

    requested = tuple(STUDIES if studies is None else studies)
    unknown = set(requested).difference(STUDIES)
    if unknown:
        raise ValueError(f"unknown studies: {sorted(unknown)}")

    baseline = _baseline(raw)
    study_config = suite["studies"]
    conditions: list[Condition] = [baseline] if include_baseline else []
    seen = {baseline.signature}

    def add(condition: Condition) -> None:
        _validate_condition(condition)
        if condition.signature == baseline.signature:
            return
        if condition.signature in seen:
            raise ValueError(f"duplicate effective condition: {condition.condition_id}")
        seen.add(condition.signature)
        conditions.append(condition)

    def changed(**changes: Any) -> dict[str, Any]:
        values = {
            "num_states": baseline.num_states,
            "max_order": baseline.max_order,
            "message_passing_steps": baseline.message_passing_steps,
            "time_decay_per_minute": baseline.time_decay_per_minute,
            "granularity_seconds": baseline.granularity_seconds,
            "history_duration_seconds": baseline.history_duration_seconds,
            "forecast_horizon": baseline.forecast_horizon,
            "seed": baseline.seed,
        }
        values.update(changes)
        return values

    if "k" in requested:
        for value in _positive_ints(study_config["k"]["values"], "k.values"):
            add(Condition(f"k-k{value:02d}", "k", **changed(num_states=value)))
    if "p" in requested:
        for value in _positive_ints(study_config["p"]["values"], "p.values"):
            add(Condition(f"p-p{value:02d}", "p", **changed(max_order=value)))
    if "r" in requested:
        for value in _positive_ints(study_config["r"]["values"], "r.values"):
            add(
                Condition(
                    f"r-r{value:02d}",
                    "r",
                    **changed(message_passing_steps=value),
                )
            )
    if "lambda" in requested:
        for value in _nonnegative_floats(
            study_config["lambda"]["values"], "lambda.values"
        ):
            add(
                Condition(
                    f"lambda-{_decimal_slug(value)}",
                    "lambda",
                    **changed(time_decay_per_minute=value),
                )
            )
    if "tau" in requested:
        for value in _positive_ints(study_config["tau"]["seconds"], "tau.seconds"):
            add(
                Condition(
                    f"tau-{_tau_slug(value)}",
                    "tau",
                    **changed(granularity_seconds=value),
                )
            )
    return conditions


def select_condition(config_path: str | Path, condition_id: str) -> Condition:
    for condition in build_conditions(config_path):
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"unknown condition {condition_id!r}")


def condition_rows(conditions: Iterable[Condition]) -> list[dict[str, Any]]:
    return [condition.metadata() for condition in conditions]


__all__ = [
    "Condition",
    "STUDIES",
    "build_conditions",
    "condition_rows",
    "select_condition",
]
