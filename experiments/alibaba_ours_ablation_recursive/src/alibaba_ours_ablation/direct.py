from __future__ import annotations

"""Per-slot membership labels and direct multi-step forecasting for ablations."""

import copy
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from alibaba_ours_exp.model import PhysicalTimeBackoff
from alibaba_ours_exp.recursive_forecast import (
    RecursiveForecastResult,
    _right_aligned_contexts,
)
from alibaba_ours_exp.training import TrainingConfig, TrainingResult

from .model import AblationModel


def _active_mask(resources: Tensor, mask: Tensor | None) -> Tensor:
    finite = torch.isfinite(resources).all(dim=-1)
    if mask is None:
        return finite
    if mask.shape != resources.shape[:2]:
        raise ValueError("mask must have shape (workloads, slots)")
    return mask.bool().to(resources.device) & finite


@torch.no_grad()
def initialize_slot_prototypes(
    model: AblationModel,
    resources: Tensor,
    mask: Tensor | None = None,
    *,
    max_candidates: int = 512,
    random_state: int | None = None,
) -> Tensor:
    """Initialize K shared prototypes from individual training slots only."""

    if resources.ndim != 3 or resources.shape[-1] != model.config.resource_dim:
        raise ValueError("resources must have shape (workloads, slots, resource_dim)")
    active = _active_mask(resources, mask)
    positions = torch.nonzero(active, as_tuple=False).cpu().numpy()
    if positions.shape[0] == 0:
        raise ValueError("no valid training slots are available for prototype initialization")
    generator = np.random.default_rng(
        model.config.random_state if random_state is None else int(random_state)
    )
    candidate_count = min(
        positions.shape[0],
        max(int(max_candidates), model.config.num_states),
    )
    selected_positions = positions[
        generator.choice(positions.shape[0], size=candidate_count, replace=False)
    ]
    candidates = torch.stack(
        [resources[int(workload), int(slot)] for workload, slot in selected_positions]
    ).detach().cpu()

    first = int(generator.integers(0, candidates.shape[0]))
    selected = [first]
    nearest = (candidates - candidates[first]).square().mean(dim=-1)
    while len(selected) < model.config.num_states:
        next_index = int(torch.argmax(nearest).item())
        selected.append(next_index)
        distance = (candidates - candidates[next_index]).square().mean(dim=-1)
        nearest = torch.minimum(nearest, distance)
        if len(set(selected)) == candidates.shape[0]:
            selected.append(selected[len(selected) % candidates.shape[0]])

    representatives = candidates[selected[: model.config.num_states]].to(
        device=model.shape_encoder.prototypes.device,
        dtype=model.shape_encoder.prototypes.dtype,
    )
    prototypes = representatives[:, None, :].expand(
        -1,
        model.config.prototype_length,
        -1,
    ).contiguous()
    model.shape_encoder.set_prototypes(prototypes)
    return prototypes


@torch.no_grad()
def compute_slot_membership_series(
    model: AblationModel,
    resources: Tensor,
    mask: Tensor | None = None,
    *,
    chunk_slots: int = 256,
) -> Tensor:
    """Encode each physical slot independently; L never enters label creation."""

    if chunk_slots <= 0:
        raise ValueError("chunk_slots must be positive")
    active = _active_mask(resources, mask)
    output = torch.empty(
        resources.shape[0],
        resources.shape[1],
        model.config.num_states,
        device=resources.device,
        dtype=resources.dtype,
    )
    for start in range(0, resources.shape[1], chunk_slots):
        stop = min(start + chunk_slots, resources.shape[1])
        output[:, start:stop] = model.encode_slot_memberships(
            resources[:, start:stop],
            active[:, start:stop],
        )
    return output


def _time_matrix(times: Tensor, workloads: int, slots: int, device: torch.device) -> Tensor:
    value = torch.as_tensor(times, device=device, dtype=torch.float32)
    if value.ndim == 1 and value.shape[0] == slots:
        return value.unsqueeze(0).expand(workloads, -1)
    if value.ndim == 2 and value.shape == (workloads, slots):
        return value
    raise ValueError("times must have shape (slots,) or (workloads, slots)")


def _masked_topology(topology: Tensor | None, active: Tensor) -> Tensor | None:
    if topology is None:
        return None
    selected = topology
    if selected.ndim == 3:
        raise ValueError("direct ablation training expects a static topology")
    if selected.shape == (active.shape[0], active.shape[0]):
        return selected * (active[:, None] & active[None, :]).to(selected.dtype)
    if selected.ndim == 2 and selected.shape[1] == active.shape[0]:
        return selected * active.to(selected.dtype).unsqueeze(0)
    raise ValueError("topology dimensions do not match workloads")


def _soft_cross_entropy(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    losses = -(target * torch.log(prediction.clamp_min(1e-8))).sum(dim=-1)
    if not valid.any():
        return prediction.sum() * 0.0
    return losses[valid].mean()


def _origin_candidates(length: int, history: int, horizon: int) -> np.ndarray:
    last = length - horizon - 1
    if last < history - 1:
        raise ValueError(
            f"series length {length} cannot support L={history} and direct horizon={horizon}"
        )
    return np.arange(history - 1, last + 1, dtype=np.int64)


def _direct_epoch(
    model: AblationModel,
    resources: Tensor,
    times: Tensor,
    active: Tensor,
    memberships: Tensor,
    topology: Tensor | None,
    link_features: Tensor | None,
    origins: np.ndarray,
    horizons: np.ndarray,
    training_config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    horizon_tensor = torch.as_tensor(horizons, device=resources.device, dtype=torch.long)
    totals: dict[str, float] = {"loss": 0.0, "final_loss": 0.0, "dynamic_loss": 0.0}
    used = 0
    for origin_value in origins:
        origin = int(origin_value)
        targets_at = origin + horizon_tensor
        history_start = origin - model.config.history_window + 1
        history_memberships = memberships[:, history_start : origin + 1]
        history_active = active[:, history_start : origin + 1]
        contexts = _right_aligned_contexts(
            history_memberships.argmax(dim=-1),
            history_active,
            model.config.max_order,
        )
        origin_active = active[:, origin]
        selected_topology = _masked_topology(topology, origin_active)
        outputs = model.direct_multi_step(
            memberships[:, origin],
            contexts,
            times[:, targets_at],
            horizon_tensor,
            selected_topology,
            link_features,
            resource_representation=resources[:, origin],
            backoff_model=model.backoff,
        )
        target = memberships[:, targets_at]
        valid = origin_active[:, None] & active[:, targets_at]
        final_loss = _soft_cross_entropy(outputs["final"], target, valid)
        dynamic_loss = _soft_cross_entropy(outputs["dynamic"], target, valid)
        loss = final_loss + training_config.dynamic_loss_weight * dynamic_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["final_loss"] += float(final_loss.detach().cpu())
        totals["dynamic_loss"] += float(dynamic_loss.detach().cpu())
        used += 1
    if used == 0:
        raise ValueError("no direct multi-step origins were evaluated")
    return {name: value / used for name, value in totals.items()}


def _direct_all_horizons(
    model: AblationModel,
    resources: Tensor,
    times: Tensor,
    active: Tensor,
    memberships: Tensor,
    topology: Tensor | None,
    link_features: Tensor | None,
    origins: np.ndarray,
    maximum_horizon: int,
    horizon_chunk_size: int,
    training_config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Cover every future slot in memory-bounded, mutually independent chunks."""

    totals: dict[str, float] = {"loss": 0.0, "final_loss": 0.0, "dynamic_loss": 0.0}
    total_weight = 0
    for start in range(1, maximum_horizon + 1, horizon_chunk_size):
        stop = min(start + horizon_chunk_size - 1, maximum_horizon)
        horizons = np.arange(start, stop + 1, dtype=np.int64)
        metrics = _direct_epoch(
            model,
            resources,
            times,
            active,
            memberships,
            topology,
            link_features,
            origins,
            horizons,
            training_config,
            optimizer,
        )
        weight = horizons.shape[0]
        for name, value in metrics.items():
            totals[name] += value * weight
        total_weight += weight
    return {name: value / max(total_weight, 1) for name, value in totals.items()}


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # PyTorch 2.0.1 on Windows can misreport an existing parent as missing
        # when the path contains non-ASCII characters.  Passing a Python file
        # handle bypasses the legacy C++ path conversion.
        with temporary.open("wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _torch_load_path(path: Path, *, map_location: torch.device) -> Mapping[str, Any]:
    """Load through a file handle for Windows paths containing Unicode."""

    with path.open("rb") as handle:
        try:
            return torch.load(handle, map_location=map_location, weights_only=False)
        except TypeError:
            handle.seek(0)
            return torch.load(handle, map_location=map_location)


def train_direct_model(
    model: AblationModel,
    resources: Tensor | np.ndarray,
    times: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
    topology: Tensor | np.ndarray | None = None,
    link_features: Tensor | np.ndarray | None = None,
    *,
    config: TrainingConfig | None = None,
    validation: Mapping[str, Any] | None = None,
    epoch_callback: Callable[..., None] | None = None,
    initialize_prototypes: bool = True,
) -> TrainingResult:
    """Train all future slots directly from one observed history context."""

    training_config = config or TrainingConfig()
    device = torch.device(training_config.device)
    if device.type == "cpu":
        torch.set_num_threads(training_config.cpu_threads)
    torch.manual_seed(training_config.random_state)
    np.random.seed(training_config.random_state)
    generator = np.random.default_rng(training_config.random_state)

    model = model.to(device)
    resource_tensor = torch.as_tensor(resources, device=device, dtype=torch.float32)
    mask_tensor = None if mask is None else torch.as_tensor(mask, device=device, dtype=torch.bool)
    active = _active_mask(resource_tensor, mask_tensor)
    time_tensor = _time_matrix(
        torch.as_tensor(times), resource_tensor.shape[0], resource_tensor.shape[1], device
    )
    topology_tensor = (
        None if topology is None else torch.as_tensor(topology, device=device, dtype=torch.float32)
    )
    link_tensor = (
        None
        if link_features is None
        else torch.as_tensor(link_features, device=device, dtype=torch.float32)
    )
    if initialize_prototypes:
        initialize_slot_prototypes(
            model,
            resource_tensor,
            active,
            max_candidates=training_config.prototype_candidates,
            random_state=training_config.random_state,
        )
    model.eval()
    training_memberships = compute_slot_membership_series(model, resource_tensor, active)
    model.fit_backoff(training_memberships, time_tensor, active)
    model.backoff.freeze_for_queries()
    model.set_transition_prior_from_memberships(training_memberships, active)

    maximum_horizon = min(
        model.config.max_forecast_horizon,
        resource_tensor.shape[1] - model.config.history_window,
    )
    train_origins = _origin_candidates(
        resource_tensor.shape[1], model.config.history_window, maximum_horizon
    )
    horizon_batch_size = int(
        getattr(training_config, "direct_horizon_batch_size", min(60, maximum_horizon))
    )

    validation_payload: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, np.ndarray, int] | None = None
    if validation is not None:
        validation_resources = torch.as_tensor(
            validation["resources"], device=device, dtype=torch.float32
        )
        validation_mask_value = validation.get("mask")
        validation_mask = (
            None
            if validation_mask_value is None
            else torch.as_tensor(validation_mask_value, device=device, dtype=torch.bool)
        )
        validation_active = _active_mask(validation_resources, validation_mask)
        validation_times = _time_matrix(
            torch.as_tensor(validation["times"]),
            validation_resources.shape[0],
            validation_resources.shape[1],
            device,
        )
        model.eval()
        validation_memberships = compute_slot_membership_series(
            model, validation_resources, validation_active
        )
        validation_horizon = min(
            model.config.max_forecast_horizon,
            validation_resources.shape[1] - model.config.history_window,
        )
        validation_origins = _origin_candidates(
            validation_resources.shape[1],
            model.config.history_window,
            validation_horizon,
        )
        validation_limit = int(getattr(training_config, "direct_validation_origins", 32))
        if validation_origins.shape[0] > validation_limit:
            positions = np.linspace(
                0, validation_origins.shape[0] - 1, validation_limit, dtype=np.int64
            )
            validation_origins = validation_origins[positions]
        validation_payload = (
            validation_resources,
            validation_times,
            validation_active,
            validation_memberships,
            None
            if validation.get("topology") is None
            else torch.as_tensor(validation["topology"], device=device, dtype=torch.float32),
            None
            if validation.get("link_features") is None
            else torch.as_tensor(validation["link_features"], device=device, dtype=torch.float32),
            validation_origins,
            validation_horizon,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    result = TrainingResult(model=model, training_memberships=None)
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    start_epoch = 1
    resume_path = (
        None
        if training_config.resume_checkpoint_path is None
        else Path(training_config.resume_checkpoint_path)
    )
    # A fresh ablation run has no per-history model directory yet.  Create it
    # before the first checkpoint write; this remains local to direct ablation
    # training and does not alter the shared L×h/sensitivity training path.
    if resume_path is not None:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
    if training_config.resume_existing and resume_path is not None and resume_path.exists():
        checkpoint = _torch_load_path(resume_path, map_location=device)
        if checkpoint.get("run_identity_digest") != training_config.run_identity_digest:
            raise ValueError("direct training checkpoint belongs to a different run identity")
        if checkpoint.get("relevant_data_digest") != training_config.relevant_data_digest:
            raise ValueError("direct training checkpoint belongs to different prepared data")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_state = checkpoint["best_state"]
        result.history = [dict(item) for item in checkpoint.get("history", [])]
        result.best_epoch = int(checkpoint.get("best_epoch", 0))
        result.best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if checkpoint.get("numpy_generator_state") is not None:
            generator.bit_generator.state = checkpoint["numpy_generator_state"]

    for epoch in range(start_epoch, training_config.epochs + 1):
        if (
            training_config.max_origins_per_epoch is not None
            and train_origins.shape[0] > training_config.max_origins_per_epoch
        ):
            epoch_origins = generator.choice(
                train_origins,
                size=training_config.max_origins_per_epoch,
                replace=False,
            )
        else:
            epoch_origins = train_origins.copy()
            generator.shuffle(epoch_origins)
        model.train()
        train_metrics = _direct_all_horizons(
            model,
            resource_tensor,
            time_tensor,
            active,
            training_memberships,
            topology_tensor,
            link_tensor,
            epoch_origins,
            maximum_horizon,
            horizon_batch_size,
            training_config,
            optimizer,
        )
        metrics = {f"train_{name}": value for name, value in train_metrics.items()}
        if validation_payload is not None:
            model.eval()
            with torch.no_grad():
                validation_metrics = _direct_all_horizons(
                    model,
                    validation_payload[0],
                    validation_payload[1],
                    validation_payload[2],
                    validation_payload[3],
                    validation_payload[4],
                    validation_payload[5],
                    validation_payload[6],
                    validation_payload[7],
                    min(horizon_batch_size, validation_payload[7]),
                    training_config,
                    None,
                )
            metrics.update(
                {f"validation_{name}": value for name, value in validation_metrics.items()}
            )
            monitored = validation_metrics["loss"]
        else:
            monitored = train_metrics["loss"]
        metrics["epoch"] = float(epoch)
        metrics["direct_horizons_trained"] = float(maximum_horizon)
        result.history.append(metrics)
        if monitored < result.best_validation_loss - training_config.min_improvement:
            result.best_validation_loss = monitored
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if resume_path is not None:
            _atomic_torch_save(
                resume_path,
                {
                    "epoch": epoch,
                    "run_identity_digest": training_config.run_identity_digest,
                    "relevant_data_digest": training_config.relevant_data_digest,
                    "model_state": model.state_dict(),
                    "best_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "history": result.history,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "stale_epochs": stale_epochs,
                    "numpy_generator_state": generator.bit_generator.state,
                },
            )
        if epoch_callback is not None:
            epoch_callback(epoch, metrics, model)
        if stale_epochs >= training_config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model.backoff.clear()
    model.eval()
    result.model = model
    return result


def _normalize_horizons(values: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, int):
        result = (int(values),)
    else:
        result = tuple(sorted({int(value) for value in values}))
    if not result or result[0] <= 0:
        raise ValueError("forecast horizons must be positive")
    return result


def _detach_tree(value: Any, return_cpu: bool) -> Any:
    if isinstance(value, Tensor):
        output = value.detach()
        return output.cpu() if return_cpu else output
    if isinstance(value, Mapping):
        return {str(key): _detach_tree(item, return_cpu) for key, item in value.items()}
    return value


def direct_forecast(
    model: AblationModel,
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
    """Directly predict every future slot from the same observed history."""

    requested = _normalize_horizons(forecast_horizon)
    maximum_h = max(requested)
    diagnostics_at = set(
        requested if diagnostic_horizons is None else _normalize_horizons(diagnostic_horizons)
    )
    selected_device = (
        next(model.parameters()).device if device is None else torch.device(device)
    )
    model = model.to(selected_device)
    model.eval()
    resources = torch.as_tensor(
        history_resources, device=selected_device, dtype=torch.float32
    )
    if resources.ndim != 3 or resources.shape[1] < model.config.history_window:
        raise ValueError("history_resources must provide at least L observed slots")
    resources = resources[:, -model.config.history_window :]
    supplied_mask = (
        None
        if history_mask is None
        else torch.as_tensor(history_mask, device=selected_device, dtype=torch.bool)[
            :, -model.config.history_window :
        ]
    )
    active = _active_mask(resources, supplied_mask)
    workloads = resources.shape[0]
    times = _time_matrix(
        torch.as_tensor(history_times)[-model.config.history_window :],
        workloads,
        model.config.history_window,
        selected_device,
    )
    if future_times is None:
        cadence = torch.ones(workloads, device=selected_device)
        forecast_time_matrix = times[:, -1:] + cadence[:, None] * torch.arange(
            1, maximum_h + 1, device=selected_device, dtype=torch.float32
        )[None, :]
    else:
        future = torch.as_tensor(future_times, device=selected_device, dtype=torch.float32)
        if future.ndim == 1 and future.shape[0] >= maximum_h:
            forecast_time_matrix = future[:maximum_h].unsqueeze(0).expand(workloads, -1)
        elif future.ndim == 2 and future.shape[0] == workloads and future.shape[1] >= maximum_h:
            forecast_time_matrix = future[:, :maximum_h]
        else:
            raise ValueError("future_times must cover every direct forecast slot")

    with torch.no_grad():
        history_memberships = compute_slot_membership_series(model, resources, active)
        initial_membership = history_memberships[:, -1]
        contexts = _right_aligned_contexts(
            history_memberships.argmax(dim=-1), active, model.config.max_order
        )
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
        local_backoff.freeze_for_queries()

        topology_tensor = (
            None
            if topology is None
            else torch.as_tensor(topology, device=selected_device, dtype=torch.float32)
        )
        topology_tensor = _masked_topology(topology_tensor, active[:, -1])
        links_tensor = (
            None
            if link_features is None
            else torch.as_tensor(link_features, device=selected_device, dtype=torch.float32)
        )

        predictions: list[Tensor] = []
        predicted_states: list[Tensor] = []
        step_latencies: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        start_h = 0
        if resume_state is not None:
            start_h = int(resume_state.get("h", 0))
            if start_h > 0:
                previous = torch.as_tensor(
                    resume_state["predictions"], device=selected_device, dtype=torch.float32
                )
                previous_states = torch.as_tensor(
                    resume_state["predicted_states"], device=selected_device, dtype=torch.long
                )
                predictions.extend(previous[:, index] for index in range(start_h))
                predicted_states.extend(previous_states[:, index] for index in range(start_h))
                step_latencies.extend(
                    float(value) for value in resume_state.get("step_latency_seconds", [])
                )

        chunk_size = max(1, min(128, checkpoint_interval_h))
        for first_h in range(start_h + 1, maximum_h + 1, chunk_size):
            last_h = min(first_h + chunk_size - 1, maximum_h)
            horizons = torch.arange(
                first_h, last_h + 1, device=selected_device, dtype=torch.long
            )
            started = time.perf_counter()
            outputs = model.direct_multi_step(
                initial_membership,
                contexts,
                forecast_time_matrix[:, first_h - 1 : last_h],
                horizons,
                topology_tensor,
                links_tensor,
                resource_representation=resources[:, -1],
                backoff_model=local_backoff,
            )
            elapsed_per_h = (time.perf_counter() - started) / horizons.numel()
            for offset, h in enumerate(range(first_h, last_h + 1)):
                prediction = outputs["final"][:, offset]
                state = prediction.argmax(dim=-1)
                predictions.append(prediction)
                predicted_states.append(state)
                step_latencies.append(elapsed_per_h)
                if h in diagnostics_at:
                    confidence = prediction.max(dim=-1).values
                    entropy = -(
                        prediction * torch.log(prediction.clamp_min(model.config.eps))
                    ).sum(dim=-1)
                    backoff_details = {
                        name: value[:, offset]
                        for name, value in outputs["backoff_diagnostics"].items()
                    }
                    diagnostic = {
                        "h": h,
                        "forecast_time": forecast_time_matrix[:, h - 1],
                        "final": prediction,
                        "dynamic": outputs["dynamic"][:, offset],
                        "backoff": outputs["backoff"][:, offset],
                        "spatial_membership": outputs["spatial_membership"],
                        "predicted_state": state,
                        "confidence": confidence,
                        "entropy": entropy,
                        "dynamic_backoff_l1": (
                            outputs["dynamic"][:, offset] - outputs["backoff"][:, offset]
                        ).abs().sum(dim=-1),
                        "graph": _detach_tree(outputs["graph_diagnostics"], return_cpu),
                        "backoff_details": _detach_tree(backoff_details, return_cpu),
                    }
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
                        diagnostic[name] = _detach_tree(diagnostic[name], return_cpu)
                    diagnostics.append(diagnostic)
            if checkpoint_callback is not None:
                checkpoint_callback(
                    last_h,
                    {
                        "h": last_h,
                        "direct_multi_step": True,
                        "predictions": torch.stack(predictions, dim=1).cpu().numpy(),
                        "predicted_states": torch.stack(predicted_states, dim=1).cpu().numpy(),
                        "step_latency_seconds": np.asarray(step_latencies, dtype=np.float64),
                    },
                )

        prediction_tensor = torch.stack(predictions, dim=1)
        state_tensor = torch.stack(predicted_states, dim=1)
        latency_tensor = torch.as_tensor(step_latencies, dtype=torch.float64)
        output_times = forecast_time_matrix
        output_initial = initial_membership
        if return_cpu:
            prediction_tensor = prediction_tensor.cpu()
            state_tensor = state_tensor.cpu()
            latency_tensor = latency_tensor.cpu()
            output_times = output_times.cpu()
            output_initial = output_initial.cpu()

    return RecursiveForecastResult(
        predictions=prediction_tensor,
        predicted_states=state_tensor,
        forecast_times=output_times,
        initial_membership=output_initial,
        step_latency_seconds=latency_tensor,
        forecast_horizons=requested,
        diagnostics=diagnostics,
        metadata={
            "history_window": model.config.history_window,
            "num_states": model.config.num_states,
            "forecast_strategy": "direct_multi_step",
            "membership_definition": "per_slot",
            "feedback": "none",
            "predicted_backoff_updates": False,
            "workloads": workloads,
        },
    )


def true_slot_memberships(
    model: AblationModel,
    resources: np.ndarray,
    active: np.ndarray,
    origin: Any,
    h_values: Iterable[int],
    workload_positions: np.ndarray,
) -> np.ndarray:
    """Build each target from exactly one observed slot, independent of L."""

    horizons = np.asarray([int(value) for value in h_values], dtype=np.int64)
    target_indices = origin.origin_index + horizons
    selected_resources = resources[workload_positions][:, target_indices]
    selected_active = active[workload_positions][:, target_indices]
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        memberships = compute_slot_membership_series(
            model,
            torch.as_tensor(selected_resources, device=device, dtype=torch.float32),
            torch.as_tensor(selected_active, device=device, dtype=torch.bool),
        )
    return memberships.detach().cpu().numpy()


__all__ = [
    "compute_slot_membership_series",
    "direct_forecast",
    "initialize_slot_prototypes",
    "train_direct_model",
    "true_slot_memberships",
]
