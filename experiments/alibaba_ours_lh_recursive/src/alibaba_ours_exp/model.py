from __future__ import annotations

"""PyTorch implementation of the fixed-K Alibaba Ours experiment model.

The module is intentionally self-contained so the experiment can be moved to a
server without importing the earlier NumPy prototype.  It keeps the model's
four modelling branches explicit: fixed-K fuzzy shapes, neighbour-aware
Einstein modulation, two-hop gated message passing, and a dynamic transition
mixed with physical-time variable-order BackOff.
"""

import bisect
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _simplex(values: Tensor, eps: float) -> Tensor:
    values = torch.clamp(values, min=0.0)
    total = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / max(values.shape[-1], 1))
    return torch.where(total > eps, values / total.clamp_min(eps), uniform)


def _as_batch_times(times: Tensor, workloads: int, length: int) -> Tensor:
    if times.ndim == 1:
        if times.shape[0] != length:
            raise ValueError("one-dimensional times must match the time dimension")
        return times.unsqueeze(0).expand(workloads, -1)
    if times.ndim == 2 and times.shape == (workloads, length):
        return times
    raise ValueError("times must have shape (time,) or (workloads, time)")


@dataclass
class OursModelConfig:
    resource_dim: int = 3
    num_states: int = 8
    history_window: int = 48
    prototype_length: int = 12
    membership_temperature: float = 0.25
    prototype_sigma: float = 1.0
    trainable_prototypes: bool = False
    einstein_strength: float = 0.5
    max_order: int = 3
    backoff_decay: float = 0.02
    backoff_time_unit: float = 1.0
    backoff_smoothing: float = 1.0
    backoff_threshold: float = 2.0
    backoff_temperature: float = 1.0
    backoff_max_age: float | None = None
    backoff_blend: float = 0.5
    time_period: float = 1440.0
    transition_hidden_dim: int = 32
    message_passing_steps: int = 2
    message_hidden_dim: int = 16
    link_feature_dim: int = 4
    dropout: float = 0.0
    eps: float = 1e-8
    random_state: int = 7

    def __post_init__(self) -> None:
        if self.resource_dim <= 0:
            raise ValueError("resource_dim must be positive")
        if self.num_states <= 1:
            raise ValueError("num_states must be greater than one")
        if self.history_window <= 0 or self.prototype_length <= 0:
            raise ValueError("history_window and prototype_length must be positive")
        if not 1 <= self.max_order <= self.history_window:
            raise ValueError("max_order must be in [1, history_window]")
        if self.backoff_time_unit <= 0 or self.time_period <= 0:
            raise ValueError("physical time scales must be positive")
        if not 0.0 <= self.backoff_blend <= 1.0:
            raise ValueError("backoff_blend must be in [0, 1]")
        if not 0.0 <= self.einstein_strength < 1.0:
            raise ValueError("einstein_strength must be in [0, 1)")
        if self.message_passing_steps < 0:
            raise ValueError("message_passing_steps must be non-negative")


class PhysicalTimeBackoff:
    """Decay-weighted variable-order BackOff indexed by physical timestamps."""

    def __init__(
        self,
        num_states: int,
        max_order: int,
        decay: float,
        time_unit: float,
        smoothing: float,
        threshold: float,
        temperature: float,
        max_age: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.num_states = int(num_states)
        self.max_order = int(max_order)
        self.decay = float(decay)
        self.time_unit = float(time_unit)
        self.smoothing = float(smoothing)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.max_age = None if max_age is None else float(max_age)
        self.eps = float(eps)
        self.events: list[dict[tuple[int, ...], list[tuple[float, np.ndarray]]]] = [
            defaultdict(list) for _ in range(self.max_order + 1)
        ]
        self._event_starts: list[dict[tuple[int, ...], int]] = [
            defaultdict(int) for _ in range(self.max_order + 1)
        ]
        self._indices: list[
            dict[tuple[int, ...], tuple[float, np.ndarray, np.ndarray]]
        ] = [dict() for _ in range(self.max_order + 1)]
        self._frozen_for_queries = False
        self._streaming = False
        self._stream_states: dict[
            tuple[int, tuple[int, ...]], tuple[float, np.ndarray, int, int]
        ] = {}
        self._latest_event_time: float | None = None
        self.fitted = False

    def clear(self) -> None:
        self.events = [defaultdict(list) for _ in range(self.max_order + 1)]
        self._event_starts = [defaultdict(int) for _ in range(self.max_order + 1)]
        self._indices = [dict() for _ in range(self.max_order + 1)]
        self._frozen_for_queries = False
        self._streaming = False
        self._stream_states = {}
        self._latest_event_time = None
        self.fitted = False

    def freeze_for_queries(self) -> None:
        """Keep compact immutable indexes and release Python event objects."""

        for order, order_events in enumerate(self.events):
            for context in list(order_events):
                self._index(order, context)
                del order_events[context]
        self._event_starts = [defaultdict(int) for _ in range(self.max_order + 1)]
        self._frozen_for_queries = True

    def enable_streaming(self) -> None:
        """Enable monotonic-time rolling queries for recursive forecasting."""

        if self._frozen_for_queries:
            raise RuntimeError("an immutable indexed BackOff cannot enter streaming mode")
        self._streaming = True
        self._stream_states.clear()

    def _invalidate(self, order: int, context: tuple[int, ...]) -> None:
        self._indices[order].pop(context, None)

    def _active_events(
        self,
        order: int,
        context: tuple[int, ...],
    ) -> list[tuple[float, np.ndarray]]:
        values = self.events[order].get(context, [])
        start = int(self._event_starts[order].get(context, 0))
        return values[start:]

    def _index(
        self,
        order: int,
        context: tuple[int, ...],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        cached = self._indices[order].get(context)
        if cached is not None:
            return cached
        values = self._active_events(order, context)
        if not values:
            cached = (
                0.0,
                np.empty(0, dtype=np.float64),
                np.zeros((1, self.num_states), dtype=np.float64),
            )
            self._indices[order][context] = cached
            return cached
        times = np.fromiter((item[0] for item in values), dtype=np.float64, count=len(values))
        targets = np.stack([item[1] for item in values]).astype(np.float64, copy=False)
        reference_time = float(times[-1])
        if self.decay == 0.0:
            scaled = targets
        else:
            scale = np.exp(self.decay * (times - reference_time) / self.time_unit)
            scaled = targets * scale[:, None]
        prefix = np.zeros((times.size + 1, self.num_states), dtype=np.float64)
        np.cumsum(scaled, axis=0, out=prefix[1:])
        cached = (reference_time, times, prefix)
        self._indices[order][context] = cached
        return cached

    def _prune_before(self, cutoff: float) -> None:
        for order, order_events in enumerate(self.events):
            starts = self._event_starts[order]
            for context in list(order_events):
                values = order_events[context]
                start = int(starts.get(context, 0))
                new_start = bisect.bisect_left(
                    values,
                    cutoff,
                    lo=start,
                    key=lambda item: item[0],
                )
                if new_start != start:
                    starts[context] = new_start
                    self._invalidate(order, context)
                if new_start >= len(values):
                    del order_events[context]
                    starts.pop(context, None)
                    self._indices[order].pop(context, None)
                elif new_start >= 4096 and new_start * 2 >= len(values):
                    order_events[context] = values[new_start:]
                    starts[context] = 0
                    self._invalidate(order, context)

    def _append_events(
        self,
        order: int,
        context: tuple[int, ...],
        additions: list[tuple[float, np.ndarray]],
    ) -> None:
        if not additions:
            return
        additions.sort(key=lambda item: item[0])
        values = self.events[order][context]
        start = int(self._event_starts[order].get(context, 0))
        if start and not self._streaming:
            values = values[start:]
            self.events[order][context] = values
            self._event_starts[order][context] = 0
        values.extend(additions)
        if len(values) > len(additions) and additions[0][0] < values[-len(additions) - 1][0]:
            values.sort(key=lambda item: item[0])
        self._invalidate(order, context)

    def clone(self) -> "PhysicalTimeBackoff":
        cloned = PhysicalTimeBackoff(
            num_states=self.num_states,
            max_order=self.max_order,
            decay=self.decay,
            time_unit=self.time_unit,
            smoothing=self.smoothing,
            threshold=self.threshold,
            temperature=self.temperature,
            max_age=self.max_age,
            eps=self.eps,
        )
        cloned.import_state(self.export_state())
        return cloned

    def append_batch(
        self,
        contexts: Tensor | np.ndarray,
        targets: Tensor | np.ndarray,
        event_times: Tensor | np.ndarray,
    ) -> None:
        """Append one synchronously predicted event per workload.

        This is called only after every workload has completed the current h,
        so no workload can observe another workload's newly predicted event
        during the same recursive step.
        """

        if self._frozen_for_queries:
            raise RuntimeError("cannot append to an immutable indexed BackOff")

        context_array = np.asarray(
            contexts.detach().cpu().numpy() if isinstance(contexts, Tensor) else contexts,
            dtype=np.int64,
        )
        target_array = np.asarray(
            targets.detach().cpu().numpy() if isinstance(targets, Tensor) else targets,
            dtype=np.float64,
        )
        time_array = np.asarray(
            event_times.detach().cpu().numpy() if isinstance(event_times, Tensor) else event_times,
            dtype=np.float64,
        ).reshape(-1)
        if context_array.ndim != 2 or target_array.shape != (context_array.shape[0], self.num_states):
            raise ValueError("contexts and targets have incompatible batch shapes")
        if time_array.shape[0] != context_array.shape[0]:
            raise ValueError("event_times must have one value per workload")
        pending: list[dict[tuple[int, ...], list[tuple[float, np.ndarray]]]] = [
            defaultdict(list) for _ in range(self.max_order + 1)
        ]
        for workload in range(context_array.shape[0]):
            target = np.clip(target_array[workload], 0.0, None)
            target /= max(float(target.sum()), self.eps)
            event_time = float(time_array[workload])
            valid_context = [int(value) for value in context_array[workload] if int(value) >= 0]
            pending[0][()].append((event_time, target.copy()))
            for order in range(1, min(self.max_order, len(valid_context)) + 1):
                pending[order][tuple(valid_context[-order:])].append(
                    (event_time, target.copy())
                )
        for order, order_pending in enumerate(pending):
            for context, additions in order_pending.items():
                self._append_events(order, context, additions)
        if self.max_age is not None and time_array.size:
            self._latest_event_time = float(np.max(time_array))
            if not self._streaming:
                self._prune_before(self._latest_event_time - self.max_age)
        self.fitted = True

    def fit(
        self,
        memberships: Tensor | np.ndarray,
        times: Tensor | np.ndarray,
        mask: Tensor | np.ndarray | None = None,
    ) -> "PhysicalTimeBackoff":
        membership_array = np.asarray(
            memberships.detach().cpu().numpy() if isinstance(memberships, Tensor) else memberships,
            dtype=np.float64,
        )
        if membership_array.ndim != 3 or membership_array.shape[-1] != self.num_states:
            raise ValueError("memberships must have shape (workloads, time, num_states)")
        time_array = np.asarray(
            times.detach().cpu().numpy() if isinstance(times, Tensor) else times,
            dtype=np.float64,
        )
        if time_array.ndim == 1:
            time_array = np.broadcast_to(time_array[None, :], membership_array.shape[:2])
        if time_array.shape != membership_array.shape[:2]:
            raise ValueError("times must have shape (time,) or (workloads, time)")
        if mask is None:
            active = np.isfinite(membership_array).all(axis=-1) & np.isfinite(time_array)
        else:
            active = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, Tensor) else mask, dtype=bool)
            if active.shape != membership_array.shape[:2]:
                raise ValueError("mask must have shape (workloads, time)")
            active &= np.isfinite(membership_array).all(axis=-1) & np.isfinite(time_array)

        self.clear()
        hard_states = np.argmax(membership_array, axis=-1)
        for workload in range(membership_array.shape[0]):
            indices = np.flatnonzero(active[workload])
            if len(indices) == 0:
                continue
            indices = indices[np.argsort(time_array[workload, indices], kind="stable")]
            sequence = hard_states[workload, indices]
            for position, target_index in enumerate(indices):
                target = np.asarray(membership_array[workload, target_index], dtype=np.float64)
                total = float(target.sum())
                if not np.isfinite(total) or total <= self.eps:
                    continue
                target = target / total
                event_time = float(time_array[workload, target_index])
                self.events[0][()].append((event_time, target.copy()))
                for order in range(1, self.max_order + 1):
                    if position < order:
                        break
                    context = tuple(int(value) for value in sequence[position - order : position])
                    self.events[order][context].append((event_time, target.copy()))
        for order, order_events in enumerate(self.events):
            for context, values in order_events.items():
                values.sort(key=lambda item: item[0])
                self._event_starts[order][context] = 0
        finite_times = time_array[np.isfinite(time_array)]
        if finite_times.size:
            self._latest_event_time = float(np.max(finite_times))
        self.fitted = True
        return self

    def _weighted_event_sum(
        self,
        values: list[tuple[float, np.ndarray]],
        start: int,
        stop: int,
        forecast_time: float,
    ) -> np.ndarray:
        if stop <= start:
            return np.zeros(self.num_states, dtype=np.float64)
        selected = values[start:stop]
        times = np.fromiter(
            (item[0] for item in selected), dtype=np.float64, count=len(selected)
        )
        targets = np.stack([item[1] for item in selected]).astype(np.float64, copy=False)
        weights = np.exp(-self.decay * (forecast_time - times) / self.time_unit)
        return np.sum(targets * weights[:, None], axis=0)

    def _streaming_counts(
        self,
        order: int,
        context: tuple[int, ...],
        forecast_time: float,
    ) -> np.ndarray:
        values = self.events[order].get(context, [])
        key = (order, context)
        state = self._stream_states.get(key)
        if state is None or forecast_time < state[0]:
            _, times, prefix = self._index(order, context)
            upper = int(np.searchsorted(times, forecast_time, side="left"))
            lower = 0
            if self.max_age is not None:
                lower = int(
                    np.searchsorted(times, forecast_time - self.max_age, side="left")
                )
            reference_time = self._indices[order][context][0]
            scaled = prefix[upper] - prefix[lower]
            factor = (
                1.0
                if self.decay == 0.0
                else float(
                    np.exp(
                        -self.decay
                        * (forecast_time - reference_time)
                        / self.time_unit
                    )
                )
            )
            counts = scaled * factor
            self._stream_states[key] = (forecast_time, counts, upper, lower)
            self._event_starts[order][context] = lower
            return counts

        last_time, previous_counts, include_cursor, expire_cursor = state
        decay_factor = (
            1.0
            if self.decay == 0.0
            else float(
                np.exp(-self.decay * (forecast_time - last_time) / self.time_unit)
            )
        )
        counts = previous_counts * decay_factor
        upper = bisect.bisect_left(
            values,
            forecast_time,
            lo=include_cursor,
            key=lambda item: item[0],
        )
        counts += self._weighted_event_sum(values, include_cursor, upper, forecast_time)
        lower = expire_cursor
        if self.max_age is not None:
            lower = bisect.bisect_left(
                values,
                forecast_time - self.max_age,
                lo=expire_cursor,
                key=lambda item: item[0],
            )
            counts -= self._weighted_event_sum(values, expire_cursor, lower, forecast_time)
        counts = np.maximum(counts, 0.0)
        self._stream_states[key] = (forecast_time, counts, upper, lower)
        self._event_starts[order][context] = lower
        return counts

    def _distribution(
        self,
        order: int,
        context: tuple[int, ...],
        forecast_time: float,
    ) -> tuple[np.ndarray, float]:
        if self._streaming:
            weighted_counts = self._streaming_counts(order, context, forecast_time)
        else:
            reference_time, times, prefix = self._index(order, context)
            upper = int(np.searchsorted(times, forecast_time, side="left"))
            lower = 0
            if self.max_age is not None:
                lower = int(
                    np.searchsorted(times, forecast_time - self.max_age, side="left")
                )
            scaled_counts = prefix[upper] - prefix[lower]
            if self.decay == 0.0:
                weighted_counts = scaled_counts
            else:
                factor = float(
                    np.exp(-self.decay * (forecast_time - reference_time) / self.time_unit)
                )
                weighted_counts = scaled_counts * factor
        support = float(weighted_counts.sum())
        smoothed = weighted_counts + self.smoothing
        distribution = smoothed / max(float(smoothed.sum()), self.eps)
        return distribution, support

    def predict(
        self,
        context: Sequence[int],
        forecast_time: float,
        *,
        _distribution_cache: dict[
            tuple[int, tuple[int, ...], float], tuple[np.ndarray, float]
        ] | None = None,
    ) -> dict[str, np.ndarray | float | int]:
        hard_context = tuple(int(value) for value in context)
        available_order = min(self.max_order, len(hard_context))
        order_distributions = np.zeros((self.max_order + 1, self.num_states), dtype=np.float64)
        supports = np.zeros(self.max_order + 1, dtype=np.float64)
        gates = np.zeros(self.max_order + 1, dtype=np.float64)
        for order in range(self.max_order + 1):
            selected_context = () if order == 0 else hard_context[-order:]
            if order > available_order:
                order_distributions[order] = np.full(self.num_states, 1.0 / self.num_states)
                continue
            cache_key = (order, selected_context, float(forecast_time))
            cached = None if _distribution_cache is None else _distribution_cache.get(cache_key)
            if cached is None:
                cached = self._distribution(order, selected_context, float(forecast_time))
                if _distribution_cache is not None:
                    _distribution_cache[cache_key] = cached
            order_distributions[order], supports[order] = cached
            if order > 0:
                gates[order] = 1.0 / (
                    1.0
                    + np.exp(
                        -(supports[order] - self.threshold) / max(self.temperature, self.eps)
                    )
                )

        order_weights = np.zeros(self.max_order + 1, dtype=np.float64)
        for order in range(1, self.max_order + 1):
            higher_failure = float(np.prod(1.0 - gates[order + 1 :]))
            order_weights[order] = gates[order] * higher_failure
        order_weights[0] = float(np.prod(1.0 - gates[1:]))
        weight_total = float(order_weights.sum())
        if weight_total <= self.eps:
            order_weights[0] = 1.0
            weight_total = 1.0
        order_weights /= weight_total
        distribution = np.sum(order_distributions * order_weights[:, None], axis=0)
        distribution /= max(float(distribution.sum()), self.eps)
        effective_order = int(np.argmax(order_weights))
        return {
            "distribution": distribution,
            "order_distributions": order_distributions,
            "supports": supports,
            "gates": gates,
            "order_weights": order_weights,
            "effective_order": effective_order,
        }

    def predict_batch(
        self,
        contexts: Tensor,
        forecast_times: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        context_array = contexts.detach().cpu().numpy()
        time_array = forecast_times.detach().cpu().numpy().reshape(-1)
        distribution_cache: dict[
            tuple[int, tuple[int, ...], float], tuple[np.ndarray, float]
        ] = {}
        results = [
            self.predict(
                context_array[index].tolist(),
                float(time_array[index]),
                _distribution_cache=distribution_cache,
            )
            for index in range(context_array.shape[0])
        ]
        distributions = torch.as_tensor(
            np.stack([item["distribution"] for item in results]), device=device, dtype=dtype
        )
        diagnostics = {
            "order_distributions": torch.as_tensor(
                np.stack([item["order_distributions"] for item in results]),
                device=device,
                dtype=dtype,
            ),
            "supports": torch.as_tensor(
                np.stack([item["supports"] for item in results]), device=device, dtype=dtype
            ),
            "gates": torch.as_tensor(
                np.stack([item["gates"] for item in results]), device=device, dtype=dtype
            ),
            "order_weights": torch.as_tensor(
                np.stack([item["order_weights"] for item in results]),
                device=device,
                dtype=dtype,
            ),
            "effective_order": torch.as_tensor(
                [item["effective_order"] for item in results], device=device, dtype=torch.long
            ),
        }
        return distributions, diagnostics

    def export_state(self) -> dict[str, Any]:
        serial_events: list[list[dict[str, Any]]] = []
        for order, order_events in enumerate(self.events):
            serial_order: list[dict[str, Any]] = []
            for context in order_events:
                raw_values = order_events[context]
                start = int(self._event_starts[order].get(context, 0))
                if (
                    self._streaming
                    and self.max_age is not None
                    and self._latest_event_time is not None
                ):
                    start = bisect.bisect_left(
                        raw_values,
                        self._latest_event_time - self.max_age,
                        lo=start,
                        key=lambda item: item[0],
                    )
                values = raw_values[start:]
                serial_order.append(
                    {
                        "context": list(context),
                        "times": np.fromiter(
                            (item[0] for item in values),
                            dtype=np.float64,
                            count=len(values),
                        ),
                        "targets": (
                            np.stack([item[1] for item in values]).astype(
                                np.float64, copy=False
                            )
                            if values
                            else np.empty((0, self.num_states), dtype=np.float64)
                        ),
                    }
                )
            serial_events.append(serial_order)
        return {
            "fitted": self.fitted,
            "events": serial_events,
            "streaming": self._streaming,
            "latest_event_time": self._latest_event_time,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        self.clear()
        for order, serial_order in enumerate(state.get("events", [])):
            if order > self.max_order:
                break
            for item in serial_order:
                context = tuple(int(value) for value in item["context"])
                times = item["times"]
                targets = item["targets"]
                self.events[order][context] = [
                    (float(event_time), np.asarray(target, dtype=np.float64))
                    for event_time, target in zip(times, targets)
                ]
                self.events[order][context].sort(key=lambda value: value[0])
                self._event_starts[order][context] = 0
        self.fitted = bool(state.get("fitted", False))
        self._streaming = bool(state.get("streaming", False))
        latest = state.get("latest_event_time")
        if latest is None:
            observed_times = [
                value[0]
                for order_events in self.events
                for values in order_events.values()
                for value in values[-1:]
            ]
            self._latest_event_time = max(observed_times) if observed_times else None
        else:
            self._latest_event_time = float(latest)


class FixedKShapeEncoder(nn.Module):
    """Fixed-count fuzzy shape encoder with optional trainable prototypes."""

    def __init__(self, config: OursModelConfig) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(config.random_state)
        initial = torch.randn(
            config.num_states,
            config.prototype_length,
            config.resource_dim,
            generator=generator,
        ) * 0.05
        self.prototypes = nn.Parameter(initial, requires_grad=config.trainable_prototypes)
        sigma = torch.full((config.num_states,), float(config.prototype_sigma)).log()
        self.log_sigma = nn.Parameter(sigma, requires_grad=config.trainable_prototypes)
        self.register_buffer("resource_weights", torch.ones(config.resource_dim))
        self.temperature = float(config.membership_temperature)
        self.eps = float(config.eps)

    def _resample(self, history: Tensor, mask: Tensor | None) -> Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape (workloads, L, resources)")
        if mask is None:
            valid = torch.isfinite(history).all(dim=-1)
        else:
            if mask.shape != history.shape[:2]:
                raise ValueError("history mask must have shape (workloads, L)")
            valid = mask.bool() & torch.isfinite(history).all(dim=-1)
        safe_history = torch.nan_to_num(history)
        target_length = self.prototypes.shape[1]
        # The formal Alibaba cohort is dense after bounded causal filling.  Use
        # one batched interpolation in that common case instead of invoking
        # F.interpolate once per workload at every training minute.
        if bool(valid.all()):
            if history.shape[1] == 1:
                return safe_history.expand(-1, target_length, -1)
            return F.interpolate(
                safe_history.transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        # Missing histories used to launch F.interpolate once per workload.
        # Group equal valid lengths so at most one batched kernel is launched
        # per distinct length while preserving the original compression rule.
        counts = valid.sum(dim=1)
        output = torch.empty(
            history.shape[0],
            target_length,
            history.shape[-1],
            device=history.device,
            dtype=history.dtype,
        )
        for count_value in torch.unique(counts).detach().cpu().tolist():
            count = int(count_value)
            selected = torch.nonzero(counts == count, as_tuple=False).flatten()
            if count == 0:
                output[selected] = 0.0
                continue
            grouped_history = safe_history[selected]
            grouped_valid = valid[selected]
            sequence = grouped_history[grouped_valid].reshape(
                selected.numel(), count, history.shape[-1]
            )
            if count == 1:
                shaped = sequence.expand(-1, target_length, -1)
            else:
                shaped = F.interpolate(
                    sequence.transpose(1, 2),
                    size=target_length,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
            output[selected] = shaped
        return output

    def forward(self, history: Tensor, mask: Tensor | None = None) -> Tensor:
        shaped = self._resample(history, mask)
        difference = shaped[:, None, :, :] - self.prototypes[None, :, :, :]
        weighted = difference.square() * self.resource_weights.view(1, 1, 1, -1)
        distance = weighted.mean(dim=(-1, -2))
        sigma = self.log_sigma.exp().clamp_min(self.eps)
        logits = -distance / (2.0 * sigma.square().unsqueeze(0))
        return torch.softmax(logits / max(self.temperature, self.eps), dim=-1)

    @torch.no_grad()
    def set_prototypes(self, values: Tensor | np.ndarray) -> None:
        prototype_values = torch.as_tensor(
            values,
            device=self.prototypes.device,
            dtype=self.prototypes.dtype,
        )
        if prototype_values.shape != self.prototypes.shape:
            raise ValueError(f"prototype shape must be {tuple(self.prototypes.shape)}")
        self.prototypes.copy_(prototype_values)

    def decode_resources(self, memberships: Tensor) -> Tensor:
        representatives = self.prototypes[:, -1, :]
        # Recursive evaluation returns predictions on CPU by default, while a
        # trained decoder may remain on CUDA. Align this public input with the
        # decoder so post-processing cannot mix CPU and CUDA tensors.
        aligned_memberships = torch.as_tensor(
            memberships,
            device=representatives.device,
            dtype=representatives.dtype,
        )
        return aligned_memberships @ representatives


def _workload_graph(
    topology: Tensor | None,
    link_features: Tensor | None,
    workloads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if topology is None:
        return torch.zeros((workloads, workloads), dtype=dtype, device=device)
    topology = topology.to(device=device, dtype=dtype)
    if topology.ndim != 2:
        raise ValueError("topology must be a workload adjacency or node-workload incidence matrix")
    if topology.shape == (workloads, workloads):
        adjacency = topology.clamp_min(0.0).clone()
    elif topology.shape[1] == workloads:
        incidence = topology.clamp_min(0.0)
        if (
            link_features is not None
            and link_features.ndim == 3
            and link_features.shape[:2] == (incidence.shape[0], incidence.shape[0])
        ):
            node_edges = torch.isfinite(link_features).all(dim=-1) & (
                link_features.abs().sum(dim=-1) > 0
            )
            node_edges = node_edges.to(dtype=dtype, device=device)
            node_edges.fill_diagonal_(1.0)
            adjacency = incidence.transpose(0, 1) @ node_edges @ incidence
        else:
            adjacency = incidence.transpose(0, 1) @ incidence
    else:
        raise ValueError("topology dimensions do not match the workload count")
    adjacency.fill_diagonal_(0.0)
    return adjacency


def _workload_link_features(
    link_features: Tensor | None,
    topology: Tensor | None,
    workloads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if link_features is None:
        return None
    features = link_features.to(device=device, dtype=dtype)
    if features.ndim != 3:
        raise ValueError("link_features must have shape (source, target, features)")
    if features.shape[:2] == (workloads, workloads):
        return features
    if topology is None or topology.ndim != 2 or topology.shape[1] != workloads:
        raise ValueError("node-level link features require a node-workload incidence matrix")
    incidence = topology.to(device=device, dtype=dtype).clamp_min(0.0)
    if features.shape[0] != incidence.shape[0] or features.shape[1] != incidence.shape[0]:
        raise ValueError("node-level link feature dimensions must match topology rows")
    projected = torch.einsum("vn,vwf,wm->nmf", incidence, features, incidence)
    normalizer = torch.einsum(
        "vn,vw,wm->nm",
        incidence,
        torch.ones_like(features[..., 0]),
        incidence,
    ).unsqueeze(-1)
    return projected / normalizer.clamp_min(1.0)


class FuzzyGraphMessagePassing(nn.Module):
    """Neighbour modulation followed by gated two-hop synchronous propagation."""

    def __init__(self, config: OursModelConfig) -> None:
        super().__init__()
        self.num_states = config.num_states
        self.resource_dim = config.resource_dim
        self.steps = config.message_passing_steps
        self.hidden_dim = config.message_hidden_dim
        self.link_feature_dim = config.link_feature_dim
        self.einstein_strength = config.einstein_strength
        self.eps = config.eps
        self.theta = nn.Parameter(torch.zeros(config.num_states + 1, config.num_states))
        self.input_projection = nn.Linear(
            config.num_states + config.resource_dim,
            config.message_hidden_dim,
        )
        self.self_layers = nn.ModuleList(
            [nn.Linear(config.message_hidden_dim, config.message_hidden_dim) for _ in range(self.steps)]
        )
        self.message_layers = nn.ModuleList(
            [nn.Linear(config.message_hidden_dim, config.message_hidden_dim, bias=False) for _ in range(self.steps)]
        )
        self.link_gate = nn.Linear(config.link_feature_dim, 1)
        self.residual_head = nn.Linear(config.message_hidden_dim, config.num_states)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        memberships: Tensor,
        resource_representation: Tensor,
        topology: Tensor | None,
        link_features: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        workloads = memberships.shape[0]
        adjacency = _workload_graph(
            topology,
            link_features,
            workloads,
            memberships.dtype,
            memberships.device,
        )
        projected_links = _workload_link_features(
            link_features,
            topology,
            workloads,
            memberships.dtype,
            memberships.device,
        )
        has_real_link_features = projected_links is not None
        if projected_links is None:
            # topology_only mode: preserve the observed deployment topology,
            # but do not fabricate bandwidth/delay/loss/availability values.
            # A unit edge weight is the neutral, unweighted propagation rule;
            # it is not a measured link gate and must be reported as N/A.
            link_strength = torch.ones_like(adjacency)
        else:
            if projected_links.shape[-1] != self.link_feature_dim:
                raise ValueError(
                    f"link feature dimension must be {self.link_feature_dim}, got {projected_links.shape[-1]}"
                )
            link_strength = torch.sigmoid(self.link_gate(projected_links).squeeze(-1))
        gated_adjacency = adjacency * link_strength
        row_total = gated_adjacency.sum(dim=1, keepdim=True)
        normalized = gated_adjacency / row_total.clamp_min(self.eps)
        has_neighbour = (row_total > self.eps).to(memberships.dtype)

        neighbour_membership = normalized @ memberships
        degree = (adjacency > 0).to(memberships.dtype).sum(dim=1, keepdim=True)
        degree = degree / max(workloads - 1, 1)
        modulation_features = torch.cat([neighbour_membership, degree], dim=-1)
        modulation = torch.tanh(modulation_features @ self.theta)
        centered = 2.0 * memberships - 1.0
        influence = self.einstein_strength * modulation * has_neighbour
        denominator = (1.0 + centered * influence).clamp_min(self.eps)
        adjusted = 0.5 * (1.0 + (centered + influence) / denominator)
        adjusted = _simplex(adjusted, self.eps)

        hidden = torch.tanh(
            self.input_projection(torch.cat([adjusted, resource_representation], dim=-1))
        )
        for h in range(self.steps):
            message = normalized @ self.message_layers[h](hidden)
            hidden = torch.tanh(self.self_layers[h](hidden) + message)
            hidden = self.dropout(hidden)
        residual = self.residual_head(hidden) * has_neighbour
        propagated = torch.softmax(torch.log(adjusted.clamp_min(self.eps)) + residual, dim=-1)
        if has_real_link_features:
            link_gate_mean = torch.where(
                adjacency > 0,
                link_strength,
                torch.zeros_like(link_strength),
            ).sum(dim=1) / (adjacency > 0).sum(dim=1).clamp_min(1)
        else:
            link_gate_mean = torch.full(
                (workloads,),
                float("nan"),
                dtype=memberships.dtype,
                device=memberships.device,
            )
        diagnostics = {
            "adjacency": adjacency,
            "normalized_adjacency": normalized,
            "neighbour_count": (adjacency > 0).sum(dim=1),
            "link_gate_mean": link_gate_mean,
            "link_features_present": torch.full(
                (workloads,),
                has_real_link_features,
                dtype=torch.bool,
                device=memberships.device,
            ),
            "unweighted_topology": torch.full(
                (workloads,),
                topology is not None and not has_real_link_features,
                dtype=torch.bool,
                device=memberships.device,
            ),
            "modulation": modulation,
            "modulation_l1": (propagated - memberships).abs().sum(dim=-1),
            "message_context_norm": hidden.norm(dim=-1),
        }
        return propagated, diagnostics


class DynamicTransition(nn.Module):
    """Membership- and physical-time-conditioned row-stochastic transition."""

    def __init__(self, config: OursModelConfig) -> None:
        super().__init__()
        self.num_states = config.num_states
        self.time_period = float(config.time_period)
        self.eps = float(config.eps)
        self.network = nn.Sequential(
            nn.Linear(config.num_states + 2, config.transition_hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transition_hidden_dim, config.num_states * config.num_states),
        )
        self.register_buffer(
            "transition_prior",
            torch.full((config.num_states, config.num_states), 1.0 / config.num_states),
        )

    @torch.no_grad()
    def set_prior(self, counts: Tensor, smoothing: float = 1.0) -> None:
        if counts.shape != self.transition_prior.shape:
            raise ValueError("transition counts must have shape (K, K)")
        smoothed = counts.to(self.transition_prior) + float(smoothing)
        self.transition_prior.copy_(smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(self.eps))

    def forward(self, memberships: Tensor, forecast_times: Tensor) -> tuple[Tensor, Tensor]:
        forecast_times = forecast_times.reshape(-1).to(device=memberships.device, dtype=memberships.dtype)
        if forecast_times.shape[0] == 1 and memberships.shape[0] > 1:
            forecast_times = forecast_times.expand(memberships.shape[0])
        if forecast_times.shape[0] != memberships.shape[0]:
            raise ValueError("forecast_times must have one value per workload")
        phase = 2.0 * torch.pi * forecast_times / self.time_period
        features = torch.cat(
            [memberships, torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)],
            dim=-1,
        )
        learned = self.network(features).view(-1, self.num_states, self.num_states)
        prior = torch.log(self.transition_prior.clamp_min(self.eps)).unsqueeze(0)
        matrix = torch.softmax(learned + prior, dim=-1)
        distribution = torch.einsum("nk,nkj->nj", memberships, matrix)
        return _simplex(distribution, self.eps), matrix


class AlibabaOursModel(nn.Module):
    """End-to-end fixed-K model used by the L by forecast_horizon experiment."""

    def __init__(self, config: OursModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or OursModelConfig()
        torch.manual_seed(self.config.random_state)
        self.shape_encoder = FixedKShapeEncoder(self.config)
        self.graph_block = FuzzyGraphMessagePassing(self.config)
        self.dynamic_transition = DynamicTransition(self.config)
        effective_backoff_max_age = (
            self.config.backoff_max_age
            if self.config.backoff_max_age is not None
            else self.config.history_window * self.config.backoff_time_unit
        )
        self.backoff = PhysicalTimeBackoff(
            num_states=self.config.num_states,
            max_order=self.config.max_order,
            decay=self.config.backoff_decay,
            time_unit=self.config.backoff_time_unit,
            smoothing=self.config.backoff_smoothing,
            threshold=self.config.backoff_threshold,
            temperature=self.config.backoff_temperature,
            max_age=effective_backoff_max_age,
            eps=self.config.eps,
        )

    def encode_membership(self, history: Tensor, mask: Tensor | None = None) -> Tensor:
        history = history[:, -self.config.history_window :, :]
        selected_mask = None if mask is None else mask[:, -self.config.history_window :]
        return self.shape_encoder(history, selected_mask)

    def encode_history_memberships(self, history: Tensor, mask: Tensor | None = None) -> Tensor:
        history = history[:, -self.config.history_window :, :]
        selected_mask = None if mask is None else mask[:, -self.config.history_window :]
        memberships: list[Tensor] = []
        for h in range(history.shape[1]):
            prefix_mask = None if selected_mask is None else selected_mask[:, : h + 1]
            memberships.append(self.shape_encoder(history[:, : h + 1, :], prefix_mask))
        return torch.stack(memberships, dim=1)

    @torch.no_grad()
    def fit_backoff(self, memberships: Tensor, times: Tensor, mask: Tensor | None = None) -> None:
        self.backoff.fit(memberships, times, mask)

    @torch.no_grad()
    def set_transition_prior_from_memberships(
        self,
        memberships: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        if memberships.ndim != 3 or memberships.shape[-1] != self.config.num_states:
            raise ValueError("memberships must have shape (workloads, time, K)")
        if mask is None:
            active = torch.ones(memberships.shape[:2], dtype=torch.bool, device=memberships.device)
        else:
            active = mask.bool().to(memberships.device)
        valid = active[:, :-1] & active[:, 1:]
        if valid.any():
            previous = memberships[:, :-1][valid]
            following = memberships[:, 1:][valid]
            counts = previous.transpose(0, 1) @ following
        else:
            counts = torch.zeros(
                self.config.num_states,
                self.config.num_states,
                device=memberships.device,
                dtype=memberships.dtype,
            )
        self.dynamic_transition.set_prior(counts)

    def step_from_membership(
        self,
        memberships: Tensor,
        hard_contexts: Tensor,
        forecast_times: Tensor,
        topology: Tensor | None = None,
        link_features: Tensor | None = None,
        resource_representation: Tensor | None = None,
        backoff_model: PhysicalTimeBackoff | None = None,
        backoff_prediction: tuple[Tensor, Mapping[str, Tensor]] | None = None,
    ) -> dict[str, Any]:
        memberships = _simplex(memberships, self.config.eps)
        if resource_representation is None:
            resource_representation = self.shape_encoder.decode_resources(memberships)
        spatial_membership, graph_diagnostics = self.graph_block(
            memberships,
            resource_representation,
            topology,
            link_features,
        )
        dynamic_distribution, transition_matrix = self.dynamic_transition(
            spatial_membership,
            forecast_times,
        )
        selected_backoff = self.backoff if backoff_model is None else backoff_model
        if backoff_prediction is not None:
            backoff_distribution, supplied_diagnostics = backoff_prediction
            backoff_distribution = backoff_distribution.to(
                device=memberships.device,
                dtype=memberships.dtype,
            )
            backoff_diagnostics = {
                name: value.to(device=memberships.device)
                for name, value in supplied_diagnostics.items()
            }
        elif selected_backoff.fitted:
            backoff_distribution, backoff_diagnostics = selected_backoff.predict_batch(
                hard_contexts,
                forecast_times,
                device=memberships.device,
                dtype=memberships.dtype,
            )
        else:
            backoff_distribution = memberships
            batch = memberships.shape[0]
            orders = self.config.max_order + 1
            backoff_diagnostics = {
                "order_distributions": memberships[:, None, :].expand(-1, orders, -1),
                "supports": torch.zeros(batch, orders, device=memberships.device, dtype=memberships.dtype),
                "gates": torch.zeros(batch, orders, device=memberships.device, dtype=memberships.dtype),
                "order_weights": F.one_hot(
                    torch.zeros(batch, dtype=torch.long, device=memberships.device),
                    num_classes=orders,
                ).to(memberships.dtype),
                "effective_order": torch.zeros(batch, dtype=torch.long, device=memberships.device),
            }
        blend = float(self.config.backoff_blend)
        final_distribution = _simplex(
            (1.0 - blend) * dynamic_distribution + blend * backoff_distribution,
            self.config.eps,
        )
        return {
            "final": final_distribution,
            "dynamic": dynamic_distribution,
            "backoff": backoff_distribution,
            "spatial_membership": spatial_membership,
            "transition_matrix": transition_matrix,
            "graph_diagnostics": graph_diagnostics,
            "backoff_diagnostics": backoff_diagnostics,
        }

    def forward(
        self,
        history: Tensor,
        times: Tensor,
        mask: Tensor | None = None,
        topology: Tensor | None = None,
        link_features: Tensor | None = None,
        hard_contexts: Tensor | None = None,
        forecast_times: Tensor | None = None,
        backoff_prediction: tuple[Tensor, Mapping[str, Tensor]] | None = None,
    ) -> dict[str, Any]:
        if history.ndim != 3 or history.shape[-1] != self.config.resource_dim:
            raise ValueError("history must have shape (workloads, L, resource_dim)")
        history = history[:, -self.config.history_window :, :]
        selected_mask = None if mask is None else mask[:, -self.config.history_window :]
        selected_times = _as_batch_times(
            times[:, -self.config.history_window :] if times.ndim == 2 else times[-self.config.history_window :],
            history.shape[0],
            history.shape[1],
        ).to(history.device)
        base_membership = self.shape_encoder(history, selected_mask)
        if hard_contexts is None:
            history_memberships = self.encode_history_memberships(history, selected_mask)
            hard_contexts = history_memberships.argmax(dim=-1)
        if forecast_times is None:
            if selected_times.shape[1] > 1:
                cadence = torch.median(selected_times[:, 1:] - selected_times[:, :-1], dim=1).values
            else:
                cadence = torch.ones(history.shape[0], device=history.device, dtype=history.dtype)
            forecast_times = selected_times[:, -1].to(history.dtype) + cadence.to(history.dtype)
        result = self.step_from_membership(
            base_membership,
            hard_contexts,
            forecast_times,
            topology,
            link_features,
            resource_representation=history[:, -1, :],
            backoff_prediction=backoff_prediction,
        )
        result["base_membership"] = base_membership
        return result

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model_state": self.state_dict(),
                "backoff_state": self.backoff.export_state(),
                "metadata": dict(metadata or {}),
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["AlibabaOursModel", dict[str, Any]]:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
        model = cls(OursModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.backoff.import_state(checkpoint.get("backoff_state", {}))
        return model, dict(checkpoint.get("metadata", {}))


__all__ = [
    "AlibabaOursModel",
    "FixedKShapeEncoder",
    "OursModelConfig",
    "PhysicalTimeBackoff",
]
