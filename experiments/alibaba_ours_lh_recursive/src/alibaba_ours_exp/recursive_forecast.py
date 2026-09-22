from __future__ import annotations

"""Leakage-safe synchronous recursive forecasting for the Alibaba experiment."""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from .model import AlibabaOursModel, PhysicalTimeBackoff


@dataclass
class RecursiveForecastResult:
    """All intermediate h predictions plus the requested forecast horizons."""

    predictions: Tensor
    predicted_states: Tensor
    forecast_times: Tensor
    initial_membership: Tensor
    step_latency_seconds: Tensor
    forecast_horizons: tuple[int, ...]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def at_h(self, h: int) -> Tensor:
        if h < 1 or h > self.predictions.shape[1]:
            raise ValueError("h is outside the computed forecast horizon")
        return self.predictions[:, h - 1]

    @property
    def requested_predictions(self) -> dict[int, Tensor]:
        return {h: self.at_h(h) for h in self.forecast_horizons}

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions,
            "predicted_states": self.predicted_states,
            "forecast_times": self.forecast_times,
            "initial_membership": self.initial_membership,
            "step_latency_seconds": self.step_latency_seconds,
            "forecast_horizons": self.forecast_horizons,
            "requested_predictions": self.requested_predictions,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }


def _normalize_forecast_horizons(forecast_horizon: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(forecast_horizon, int):
        values = (int(forecast_horizon),)
    else:
        values = tuple(sorted({int(value) for value in forecast_horizon}))
    if not values or values[0] <= 0:
        raise ValueError("forecast_horizon values must be positive")
    return values


def _history_times(
    values: Tensor | np.ndarray,
    workloads: int,
    length: int,
    device: torch.device,
) -> Tensor:
    times = torch.as_tensor(values, dtype=torch.float32, device=device)
    if times.ndim == 1:
        if times.shape[0] < length:
            raise ValueError("history_times is shorter than L")
        return times[-length:].unsqueeze(0).expand(workloads, -1).clone()
    if times.ndim == 2 and times.shape[0] == workloads and times.shape[1] >= length:
        return times[:, -length:].clone()
    raise ValueError("history_times must have shape (time,) or (workloads, time)")


def _future_times(
    history_times: Tensor,
    history_mask: Tensor,
    maximum_h: int,
    values: Tensor | np.ndarray | None,
) -> Tensor:
    workloads = history_times.shape[0]
    if values is not None:
        supplied = torch.as_tensor(values, dtype=history_times.dtype, device=history_times.device)
        if supplied.ndim == 1:
            if supplied.shape[0] < maximum_h:
                raise ValueError("future_times must cover the maximum requested h")
            return supplied[:maximum_h].unsqueeze(0).expand(workloads, -1).clone()
        if supplied.ndim == 2 and supplied.shape[0] == workloads and supplied.shape[1] >= maximum_h:
            return supplied[:, :maximum_h].clone()
        raise ValueError("future_times must have shape (h,) or (workloads, h)")

    result = torch.empty(
        workloads,
        maximum_h,
        dtype=history_times.dtype,
        device=history_times.device,
    )
    for workload in range(workloads):
        valid_times = history_times[workload, history_mask[workload]]
        if valid_times.numel() == 0:
            raise ValueError("every workload needs at least one valid history timestamp")
        if valid_times.numel() > 1:
            cadence = torch.median(valid_times[1:] - valid_times[:-1])
            if cadence <= 0:
                raise ValueError("history timestamps must increase in physical time")
        else:
            cadence = torch.tensor(1.0, device=history_times.device, dtype=history_times.dtype)
        result[workload] = valid_times[-1] + cadence * torch.arange(
            1,
            maximum_h + 1,
            device=history_times.device,
            dtype=history_times.dtype,
        )
    return result


def _right_aligned_contexts(states: Tensor, history_mask: Tensor, length: int) -> Tensor:
    contexts = torch.full(
        (states.shape[0], length),
        -1,
        dtype=torch.long,
        device=states.device,
    )
    for workload in range(states.shape[0]):
        valid_states = states[workload, history_mask[workload]][-length:]
        if valid_states.numel() > 0:
            contexts[workload, -valid_states.numel() :] = valid_states
    return contexts


def _last_valid_resources(history: Tensor, history_mask: Tensor) -> Tensor:
    values: list[Tensor] = []
    for workload in range(history.shape[0]):
        indices = torch.nonzero(history_mask[workload], as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("every workload needs at least one valid history resource vector")
        values.append(history[workload, indices[-1]])
    return torch.stack(values)


def _freeze_topology(
    topology: Tensor | np.ndarray | None,
    workloads: int,
    device: torch.device,
) -> Tensor | None:
    if topology is None:
        return None
    frozen = torch.as_tensor(topology, dtype=torch.float32, device=device).detach().clone()
    if frozen.ndim != 2:
        raise ValueError("recursive forecasting requires one origin topology snapshot")
    if frozen.shape != (workloads, workloads) and frozen.shape[1] != workloads:
        raise ValueError("origin topology dimensions do not match the workload count")
    return frozen


def _freeze_links(
    link_features: Tensor | np.ndarray | None,
    device: torch.device,
) -> Tensor | None:
    if link_features is None:
        return None
    frozen = torch.as_tensor(link_features, dtype=torch.float32, device=device).detach().clone()
    if frozen.ndim != 3:
        raise ValueError("recursive forecasting requires one origin link-feature snapshot")
    return frozen


def _detached_diagnostics(values: dict[str, Tensor], return_cpu: bool) -> dict[str, Tensor]:
    detached: dict[str, Tensor] = {}
    for name, value in values.items():
        selected = value.detach()
        detached[name] = selected.cpu() if return_cpu else selected.clone()
    return detached


@torch.no_grad()
@torch.inference_mode()
def recursive_forecast(
    model: AlibabaOursModel,
    history_resources: Tensor | np.ndarray,
    history_times: Tensor | np.ndarray,
    forecast_horizon: int | Iterable[int],
    history_mask: Tensor | np.ndarray | None = None,
    topology: Tensor | np.ndarray | None = None,
    link_features: Tensor | np.ndarray | None = None,
    future_times: Tensor | np.ndarray | None = None,
    *,
    device: str | torch.device | None = None,
    return_cpu: bool = True,
    diagnostic_horizons: Iterable[int] | None = None,
    checkpoint_interval_h: int = 60,
    checkpoint_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    resume_state: Mapping[str, Any] | None = None,
) -> RecursiveForecastResult:
    """Forecast all workloads synchronously using exactly L observed history.

    The soft final distribution is fed into the next h.  Only its argmax is
    appended to the BackOff context.  Origin topology and link features are
    cloned once and never replaced with future observations.
    """

    forecast_horizons = _normalize_forecast_horizons(forecast_horizon)
    maximum_h = max(forecast_horizons)
    if checkpoint_interval_h <= 0:
        raise ValueError("checkpoint_interval_h must be positive")
    diagnostic_set = set(
        forecast_horizons
        if diagnostic_horizons is None
        else _normalize_forecast_horizons(diagnostic_horizons)
    )
    if device is None:
        try:
            selected_device = next(model.parameters()).device
        except StopIteration:
            selected_device = torch.device("cpu")
    else:
        selected_device = torch.device(device)
    model = model.to(selected_device)
    model.eval()

    resources = torch.as_tensor(
        history_resources,
        dtype=torch.float32,
        device=selected_device,
    )
    if resources.ndim != 3 or resources.shape[-1] != model.config.resource_dim:
        raise ValueError("history_resources must have shape (workloads, time, resource_dim)")
    if resources.shape[1] < model.config.history_window:
        raise ValueError("recursive forecasting requires at least L history points")
    resources = resources[:, -model.config.history_window :, :].clone()
    workloads = resources.shape[0]
    times = _history_times(
        history_times,
        workloads,
        model.config.history_window,
        selected_device,
    )
    finite = torch.isfinite(resources).all(dim=-1) & torch.isfinite(times)
    if history_mask is None:
        active = finite
    else:
        supplied_mask = torch.as_tensor(history_mask, dtype=torch.bool, device=selected_device)
        if supplied_mask.ndim != 2 or supplied_mask.shape[0] != workloads:
            raise ValueError("history_mask must have shape (workloads, time)")
        if supplied_mask.shape[1] < model.config.history_window:
            raise ValueError("history_mask is shorter than L")
        active = supplied_mask[:, -model.config.history_window :].clone() & finite
    if not active.any(dim=1).all():
        raise ValueError("every workload must have valid observations inside L")

    frozen_topology = _freeze_topology(topology, workloads, selected_device)
    frozen_links = _freeze_links(link_features, selected_device)
    forecast_time_matrix = _future_times(times, active, maximum_h, future_times)

    history_memberships = model.encode_history_memberships(resources, active)
    initial_membership = model.encode_membership(resources, active)
    hard_contexts = _right_aligned_contexts(
        history_memberships.argmax(dim=-1),
        active,
        model.config.history_window,
    )
    initial_hard_contexts = hard_contexts.clone()
    origin_resources = _last_valid_resources(resources, active)
    current_membership = initial_membership.detach().clone()
    # Rebuild BackOff from exactly this origin's L-sized observed history.
    # This prevents older training events from bypassing the configured window
    # and lets each rolling origin use observations available at that origin.
    template = model.backoff
    local_backoff = PhysicalTimeBackoff(
        num_states=template.num_states,
        max_order=template.max_order,
        decay=template.decay,
        time_unit=template.time_unit,
        smoothing=template.smoothing,
        threshold=template.threshold,
        temperature=template.temperature,
        max_age=template.max_age,
        eps=template.eps,
    )
    local_backoff.fit(history_memberships, times, active)

    predictions: list[Tensor] = []
    predicted_states: list[Tensor] = []
    step_latencies: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    start_h = 0
    if resume_state is not None:
        start_h = int(resume_state.get("h", 0))
        if start_h < 0 or start_h > maximum_h:
            raise ValueError("resume h must be in [0, maximum_h]")
        current_membership = torch.as_tensor(
            resume_state["current_membership"],
            dtype=resources.dtype,
            device=selected_device,
        )
        hard_contexts = torch.as_tensor(
            resume_state["hard_contexts"],
            dtype=torch.long,
            device=selected_device,
        )
        previous_predictions = torch.as_tensor(
            resume_state["predictions"], dtype=resources.dtype, device=selected_device
        )
        previous_states = torch.as_tensor(
            resume_state["predicted_states"], dtype=torch.long, device=selected_device
        )
        predictions = [previous_predictions[:, index] for index in range(start_h)]
        predicted_states = [previous_states[:, index] for index in range(start_h)]
        step_latencies = [float(value) for value in resume_state.get("step_latency_seconds", [])]
        legacy_backoff_state = resume_state.get("backoff_state")
        if legacy_backoff_state is not None:
            local_backoff.import_state(legacy_backoff_state)
            local_backoff.enable_streaming()
        else:
            # Checkpoints keep one copy of predictions instead of four copies
            # of every BackOff order. Replaying is linear in completed h and
            # occurs only when a run is actually resumed.
            local_backoff.enable_streaming()
            replay_contexts = initial_hard_contexts.clone()
            for replay_index in range(start_h):
                local_backoff.append_batch(
                    replay_contexts,
                    previous_predictions[:, replay_index],
                    forecast_time_matrix[:, replay_index],
                )
                replay_contexts[:, :-1] = replay_contexts[:, 1:].clone()
                replay_contexts[:, -1] = previous_states[:, replay_index]
            if not torch.equal(replay_contexts, hard_contexts):
                raise ValueError("recursive checkpoint contexts do not match replayed predictions")
    else:
        local_backoff.enable_streaming()
    for h in range(start_h + 1, maximum_h + 1):
        step_started = time.perf_counter()
        previous_membership = current_membership
        outputs = model.step_from_membership(
            previous_membership,
            hard_contexts,
            forecast_time_matrix[:, h - 1],
            frozen_topology,
            frozen_links,
            resource_representation=origin_resources if h == 1 else None,
            backoff_model=local_backoff,
        )
        next_membership = outputs["final"]
        next_state = next_membership.argmax(dim=-1)
        predictions.append(next_membership)
        predicted_states.append(next_state)
        local_backoff.append_batch(
            hard_contexts,
            next_membership,
            forecast_time_matrix[:, h - 1],
        )
        hard_contexts[:, :-1] = hard_contexts[:, 1:].clone()
        hard_contexts[:, -1] = next_state
        current_membership = next_membership
        step_latencies.append(time.perf_counter() - step_started)

        if h in diagnostic_set:
            confidence = next_membership.max(dim=-1).values
            entropy = -(
                next_membership * torch.log(next_membership.clamp_min(model.config.eps))
            ).sum(dim=-1)
            h_diagnostics: dict[str, Any] = {
                "h": h,
                "forecast_time": forecast_time_matrix[:, h - 1].detach(),
                "final": next_membership,
                "dynamic": outputs["dynamic"].detach(),
                "backoff": outputs["backoff"].detach(),
                "spatial_membership": outputs["spatial_membership"].detach(),
                "predicted_state": next_state.detach(),
                "confidence": confidence.detach(),
                "entropy": entropy.detach(),
                "dynamic_backoff_l1": (
                    outputs["dynamic"] - outputs["backoff"]
                ).abs().sum(dim=-1).detach(),
                "graph": _detached_diagnostics(outputs["graph_diagnostics"], return_cpu),
                "backoff_details": _detached_diagnostics(
                    outputs["backoff_diagnostics"], return_cpu
                ),
            }
            if return_cpu:
                for name in (
                    "forecast_time",
                    "final",
                    "dynamic",
                    "backoff",
                    "spatial_membership",
                    "predicted_state",
                    "confidence",
                    "entropy",
                    "dynamic_backoff_l1",
                ):
                    h_diagnostics[name] = h_diagnostics[name].cpu()
            diagnostics.append(h_diagnostics)
        if checkpoint_callback is not None and (
            h % checkpoint_interval_h == 0 or h == maximum_h
        ):
            checkpoint_callback(
                h,
                {
                    "h": h,
                    "current_membership": current_membership.detach().cpu().numpy(),
                    "hard_contexts": hard_contexts.detach().cpu().numpy(),
                    "predictions": torch.stack(predictions, dim=1).detach().cpu().numpy(),
                    "predicted_states": torch.stack(predicted_states, dim=1).detach().cpu().numpy(),
                    "step_latency_seconds": np.asarray(step_latencies, dtype=np.float64),
                },
            )

    prediction_tensor = torch.stack(predictions, dim=1)
    state_tensor = torch.stack(predicted_states, dim=1)
    output_times = forecast_time_matrix
    output_initial = initial_membership.detach()
    output_latency = torch.as_tensor(step_latencies, dtype=torch.float64)
    if return_cpu:
        prediction_tensor = prediction_tensor.cpu()
        state_tensor = state_tensor.cpu()
        output_times = output_times.cpu()
        output_initial = output_initial.cpu()
        output_latency = output_latency.cpu()
    if frozen_links is not None:
        link_mode = "real_link_quality"
    elif frozen_topology is not None:
        link_mode = "topology_only"
    else:
        link_mode = "spatial_disabled"
    return RecursiveForecastResult(
        predictions=prediction_tensor,
        predicted_states=state_tensor,
        forecast_times=output_times,
        initial_membership=output_initial,
        step_latency_seconds=output_latency,
        forecast_horizons=forecast_horizons,
        diagnostics=diagnostics,
        metadata={
            "history_window": model.config.history_window,
            "num_states": model.config.num_states,
            "max_order": model.config.max_order,
            "message_passing_steps": model.config.message_passing_steps,
            "workloads": workloads,
            "origin_topology_frozen": frozen_topology is not None,
            "origin_links_frozen": frozen_links is not None,
            "link_mode": link_mode,
            "real_link_features": frozen_links is not None,
            "full_ours": frozen_links is not None,
            "topology_source": (
                "deployment_colocation" if frozen_topology is not None else "none"
            ),
            "edge_weighting": (
                "measured_link_gate"
                if frozen_links is not None
                else ("unit_unweighted" if frozen_topology is not None else "none")
            ),
            "link_quality_metrics_applicable": frozen_links is not None,
            "feedback": "soft_membership_with_argmax_backoff",
            "predicted_backoff_updates": True,
            "resumed_from_h": start_h,
        },
    )


__all__ = ["RecursiveForecastResult", "recursive_forecast"]
