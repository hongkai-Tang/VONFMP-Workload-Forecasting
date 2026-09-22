from __future__ import annotations

"""Training utilities for the fixed-L Alibaba Ours experiment."""

import copy
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .model import AlibabaOursModel
from workload_fmm.dtw import dtw_distance


EpochCallback = Callable[..., None]


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    origin_stride: int = 1
    max_origins_per_epoch: int | None = 128
    dynamic_loss_weight: float = 0.25
    adaptive_order_loss_weight: float = 0.25
    early_stopping_patience: int = 5
    min_improvement: float = 1e-5
    prototype_candidates: int = 512
    device: str = "cpu"
    cpu_threads: int = 1
    random_state: int = 7
    checkpoint_path: str | Path | None = None
    resume_checkpoint_path: str | Path | None = None
    resume_existing: bool = False
    run_identity_digest: str | None = None
    relevant_data_digest: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.origin_stride <= 0:
            raise ValueError("learning_rate and origin_stride must be positive")
        if self.max_origins_per_epoch is not None and self.max_origins_per_epoch <= 0:
            raise ValueError("max_origins_per_epoch must be positive when provided")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")


@dataclass
class TrainingResult:
    model: AlibabaOursModel
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float("inf")
    training_memberships: Tensor | None = None


def _tensor_tree_to_numpy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, Mapping):
        return {str(key): _tensor_tree_to_numpy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [_tensor_tree_to_numpy(item) for item in value]
    return value


def _model_state_from_numpy(value: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {
        str(key): torch.as_tensor(item, device=device)
        for key, item in value.items()
    }


def _optimizer_state_from_numpy(value: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    restored = copy.deepcopy(dict(value))
    state = restored.get("state", {})
    restored["state"] = {
        int(key): {
            str(name): torch.as_tensor(item, device=device)
            if isinstance(item, np.ndarray)
            else item
            for name, item in payload.items()
        }
        for key, payload in state.items()
    }
    return restored


def _time_matrix(times: Tensor, workloads: int, length: int, device: torch.device) -> Tensor:
    times = times.to(device)
    if times.ndim == 1:
        if times.shape[0] != length:
            raise ValueError("times must match the resource time dimension")
        return times.unsqueeze(0).expand(workloads, -1)
    if times.ndim == 2 and times.shape == (workloads, length):
        return times
    raise ValueError("times must have shape (time,) or (workloads, time)")


def _active_mask(resources: Tensor, mask: Tensor | None) -> Tensor:
    finite = torch.isfinite(resources).all(dim=-1)
    if mask is None:
        return finite
    if mask.shape != resources.shape[:2]:
        raise ValueError("mask must have shape (workloads, time)")
    return mask.bool().to(resources.device) & finite


def _snapshot(values: Tensor | None, origin: int, expected_static_rank: int) -> Tensor | None:
    if values is None:
        return None
    if values.ndim == expected_static_rank:
        return values
    if values.ndim == expected_static_rank + 1:
        return values[origin]
    raise ValueError("time-varying graph inputs must have one leading time dimension")


def _mask_topology(topology: Tensor | None, active: Tensor) -> Tensor | None:
    if topology is None:
        return None
    if topology.ndim != 2:
        raise ValueError("topology snapshot must be two-dimensional")
    selected = topology.clone()
    if selected.shape == (active.shape[0], active.shape[0]):
        pair_mask = active[:, None] & active[None, :]
        return selected * pair_mask.to(selected.dtype)
    if selected.shape[1] == active.shape[0]:
        return selected * active.to(selected.dtype).unsqueeze(0)
    raise ValueError("topology dimensions do not match the workload count")


@torch.no_grad()
def initialize_fixed_k_prototypes(
    model: AlibabaOursModel,
    resources: Tensor,
    mask: Tensor | None = None,
    *,
    max_candidates: int = 512,
    random_state: int | None = None,
) -> Tensor:
    """Initialize exactly K prototypes with deterministic farthest-point sampling."""

    if resources.ndim != 3 or resources.shape[-1] != model.config.resource_dim:
        raise ValueError("resources must have shape (workloads, time, resource_dim)")
    active = _active_mask(resources, mask)
    history_window = model.config.history_window
    generator = np.random.default_rng(
        model.config.random_state if random_state is None else int(random_state)
    )
    origins = np.arange(max(history_window - 1, 0), resources.shape[1])
    generator.shuffle(origins)
    candidates: list[Tensor] = []
    for origin in origins:
        start = max(0, int(origin) - history_window + 1)
        histories = resources[:, start : int(origin) + 1]
        histories_mask = active[:, start : int(origin) + 1]
        valid_workloads = torch.nonzero(
            histories_mask.any(dim=1),
            as_tuple=False,
        ).flatten().cpu().numpy()
        generator.shuffle(valid_workloads)
        for workload in valid_workloads:
            shaped = model.shape_encoder._resample(
                histories[int(workload) : int(workload) + 1],
                histories_mask[int(workload) : int(workload) + 1],
            )[0]
            candidates.append(shaped.detach().cpu())
            if len(candidates) >= max(max_candidates, model.config.num_states):
                break
        if len(candidates) >= max(max_candidates, model.config.num_states):
            break
    if not candidates:
        raise ValueError("no valid histories are available for prototype initialization")
    candidate_tensor = torch.stack(candidates)
    first = int(generator.integers(0, candidate_tensor.shape[0]))
    selected = [first]
    candidates_numpy = candidate_tensor.numpy()
    weights = np.ones(model.config.resource_dim, dtype=np.float64)
    nearest_distance = torch.as_tensor(
        [
            dtw_distance(candidate, candidates_numpy[first], weights=weights, window=5)
            for candidate in candidates_numpy
        ],
        dtype=torch.float32,
    )
    while len(selected) < model.config.num_states:
        next_index = int(torch.argmax(nearest_distance).item())
        selected.append(next_index)
        distance = torch.as_tensor(
            [
                dtw_distance(candidate, candidates_numpy[next_index], weights=weights, window=5)
                for candidate in candidates_numpy
            ],
            dtype=torch.float32,
        )
        nearest_distance = torch.minimum(nearest_distance, distance)
        if len(set(selected)) == candidate_tensor.shape[0] and len(selected) < model.config.num_states:
            selected.append(selected[len(selected) % candidate_tensor.shape[0]])
    prototypes = candidate_tensor[selected[: model.config.num_states]].to(
        device=model.shape_encoder.prototypes.device,
        dtype=model.shape_encoder.prototypes.dtype,
    )
    model.shape_encoder.set_prototypes(prototypes)
    return prototypes


@torch.no_grad()
def compute_membership_series(
    model: AlibabaOursModel,
    resources: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Encode every physical time using no more than the configured L history."""

    active = _active_mask(resources, mask)
    workloads, total_steps, resource_dim = resources.shape
    history_window = model.config.history_window
    result = torch.empty(
        workloads,
        total_steps,
        model.config.num_states,
        device=resources.device,
        dtype=resources.dtype,
    )
    prefix_steps = min(max(history_window - 1, 0), total_steps)
    for h in range(prefix_steps):
        start = max(0, h - model.config.history_window + 1)
        result[:, h] = model.encode_membership(
            resources[:, start : h + 1],
            active[:, start : h + 1],
        )
    if total_steps >= history_window:
        windows = resources.unfold(1, history_window, 1).permute(0, 1, 3, 2)
        mask_windows = active.unfold(1, history_window, 1)
        window_count = windows.shape[1]
        elements_per_window = max(workloads * history_window * resource_dim, 1)
        chunk_size = max(1, min(128, 4_000_000 // elements_per_window))
        for offset in range(0, window_count, chunk_size):
            stop = min(offset + chunk_size, window_count)
            chunk = stop - offset
            encoded = model.encode_membership(
                windows[:, offset:stop].reshape(
                    workloads * chunk,
                    history_window,
                    resource_dim,
                ),
                mask_windows[:, offset:stop].reshape(workloads * chunk, history_window),
            )
            result[:, history_window - 1 + offset : history_window - 1 + stop] = encoded.reshape(
                workloads,
                chunk,
                model.config.num_states,
            )
    return result


def soft_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    active: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    losses = -(target * torch.log(prediction.clamp_min(eps))).sum(dim=-1)
    if active is not None:
        losses = losses[active]
    if losses.numel() == 0:
        return prediction.sum() * 0.0
    return losses.mean()


def multi_order_adaptive_soft_cross_entropy(
    outputs: Mapping[str, Any],
    target: Tensor,
    active: Tensor,
    *,
    dynamic_weight: float = 0.25,
    adaptive_order_weight: float = 0.25,
    blend: float = 0.5,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train with final, dynamic, and BackOff-order-adaptive soft targets."""

    final_loss = soft_cross_entropy(outputs["final"], target, active, eps)
    dynamic_loss = soft_cross_entropy(outputs["dynamic"], target, active, eps)
    order_distributions = outputs["backoff_diagnostics"]["order_distributions"]
    order_weights = outputs["backoff_diagnostics"]["order_weights"].detach()
    dynamic_expanded = outputs["dynamic"].unsqueeze(1).expand_as(order_distributions)
    adaptive_predictions = (
        (1.0 - float(blend)) * dynamic_expanded
        + float(blend) * order_distributions.detach()
    )
    target_expanded = target.unsqueeze(1).expand_as(adaptive_predictions)
    order_losses = -(
        target_expanded * torch.log(adaptive_predictions.clamp_min(eps))
    ).sum(dim=-1)
    adaptive_per_workload = (order_losses * order_weights).sum(dim=-1)
    adaptive_loss = (
        adaptive_per_workload[active].mean()
        if active.any()
        else outputs["final"].sum() * 0.0
    )
    total = (
        final_loss
        + float(dynamic_weight) * dynamic_loss
        + float(adaptive_order_weight) * adaptive_loss
    )
    return total, {
        "loss": total.detach(),
        "final_loss": final_loss.detach(),
        "dynamic_loss": dynamic_loss.detach(),
        "adaptive_order_loss": adaptive_loss.detach(),
    }


def _origins(length: int, history_window: int, config: TrainingConfig) -> list[int]:
    values = list(range(history_window - 1, length - 1, config.origin_stride))
    if config.max_origins_per_epoch is not None and len(values) > config.max_origins_per_epoch:
        selected = np.linspace(
            0,
            len(values) - 1,
            num=config.max_origins_per_epoch,
            dtype=int,
        )
        values = [values[int(index)] for index in selected]
    return values


BackoffPrediction = tuple[Tensor, dict[str, Tensor]]


@torch.no_grad()
def _precompute_backoff_predictions(
    model: AlibabaOursModel,
    memberships: Tensor,
    times: Tensor,
    origins: list[int],
) -> dict[int, BackoffPrediction]:
    """Cache the fixed BackOff branch once instead of once per epoch."""

    if not model.backoff.fitted:
        return {}
    context_states = memberships.argmax(dim=-1)
    cached: dict[int, BackoffPrediction] = {}
    for origin in origins:
        start = origin - model.config.history_window + 1
        distribution, diagnostics = model.backoff.predict_batch(
            context_states[:, start : origin + 1],
            times[:, origin + 1],
            device=memberships.device,
            dtype=memberships.dtype,
        )
        cached[origin] = (
            distribution.detach(),
            {name: value.detach() for name, value in diagnostics.items()},
        )
    return cached


def _run_origins(
    model: AlibabaOursModel,
    resources: Tensor,
    times: Tensor,
    active: Tensor,
    memberships: Tensor,
    topology: Tensor | None,
    link_features: Tensor | None,
    origins: list[int],
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    backoff_predictions: Mapping[int, BackoffPrediction] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    totals = {"loss": 0.0, "final_loss": 0.0, "dynamic_loss": 0.0, "adaptive_order_loss": 0.0}
    processed = 0
    context_states = memberships.argmax(dim=-1)
    fixed_memberships = not any(
        parameter.requires_grad for parameter in model.shape_encoder.parameters()
    )
    for origin in origins:
        start = origin - model.config.history_window + 1
        target_active = active[:, origin] & active[:, origin + 1]
        if not target_active.any():
            continue
        topology_snapshot = _snapshot(topology, origin, 2)
        topology_snapshot = _mask_topology(topology_snapshot, active[:, origin])
        link_snapshot = _snapshot(link_features, origin, 3)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            cached_backoff = (
                None
                if backoff_predictions is None
                else backoff_predictions.get(origin)
            )
            if fixed_memberships:
                outputs = model.step_from_membership(
                    memberships[:, origin],
                    context_states[:, start : origin + 1],
                    times[:, origin + 1],
                    topology_snapshot,
                    link_snapshot,
                    resource_representation=resources[:, origin],
                    backoff_prediction=cached_backoff,
                )
            else:
                outputs = model(
                    resources[:, start : origin + 1],
                    times[:, start : origin + 1],
                    active[:, start : origin + 1],
                    topology_snapshot,
                    link_snapshot,
                    hard_contexts=context_states[:, start : origin + 1],
                    forecast_times=times[:, origin + 1],
                    backoff_prediction=cached_backoff,
                )
            loss, components = multi_order_adaptive_soft_cross_entropy(
                outputs,
                memberships[:, origin + 1],
                target_active,
                dynamic_weight=config.dynamic_loss_weight,
                adaptive_order_weight=config.adaptive_order_loss_weight,
                blend=model.config.backoff_blend,
                eps=model.config.eps,
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        for name in totals:
            totals[name] += float(components[name].item())
        processed += 1
    if processed == 0:
        raise ValueError("no valid one-step training origins were found")
    return {name: value / processed for name, value in totals.items()}


def _invoke_epoch_callback(
    callback: EpochCallback | None,
    epoch: int,
    metrics: Mapping[str, float],
    model: AlibabaOursModel,
) -> None:
    if callback is None:
        return
    try:
        parameter_count = len(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        parameter_count = 3
    if parameter_count <= 2:
        callback(epoch, metrics)
    else:
        callback(epoch, metrics, model)


def train_model(
    model: AlibabaOursModel,
    resources: Tensor | np.ndarray,
    times: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
    topology: Tensor | np.ndarray | None = None,
    link_features: Tensor | np.ndarray | None = None,
    *,
    config: TrainingConfig | None = None,
    validation: Mapping[str, Any] | None = None,
    epoch_callback: EpochCallback | None = None,
    initialize_prototypes: bool = True,
) -> TrainingResult:
    """Train Theta, dynamic transition, and two-hop propagation on CPU by default."""

    training_config = config or TrainingConfig()
    device = torch.device(training_config.device)
    if device.type == "cpu":
        torch.set_num_threads(training_config.cpu_threads)
    torch.manual_seed(training_config.random_state)
    np.random.seed(training_config.random_state)
    model = model.to(device)
    resource_tensor = torch.as_tensor(resources, dtype=torch.float32, device=device)
    if resource_tensor.ndim != 3:
        raise ValueError("resources must have shape (workloads, time, resource_dim)")
    time_tensor = _time_matrix(
        torch.as_tensor(times),
        resource_tensor.shape[0],
        resource_tensor.shape[1],
        device,
    ).to(torch.float32)
    mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=device)
    active = _active_mask(resource_tensor, mask_tensor)
    topology_tensor = None if topology is None else torch.as_tensor(topology, dtype=torch.float32, device=device)
    link_tensor = (
        None
        if link_features is None
        else torch.as_tensor(link_features, dtype=torch.float32, device=device)
    )

    if initialize_prototypes:
        initialize_fixed_k_prototypes(
            model,
            resource_tensor,
            active,
            max_candidates=training_config.prototype_candidates,
            random_state=training_config.random_state,
        )
    model.eval()
    training_memberships = compute_membership_series(model, resource_tensor, active).detach()
    model.fit_backoff(training_memberships, time_tensor, active)
    model.set_transition_prior_from_memberships(training_memberships, active)

    train_origins = _origins(
        resource_tensor.shape[1],
        model.config.history_window,
        training_config,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    validation_payload: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, list[int]] | None = None
    if validation is not None:
        validation_resources = torch.as_tensor(
            validation["resources"], dtype=torch.float32, device=device
        )
        validation_times = _time_matrix(
            torch.as_tensor(validation["times"]),
            validation_resources.shape[0],
            validation_resources.shape[1],
            device,
        ).to(torch.float32)
        validation_mask_value = validation.get("mask")
        validation_mask = (
            None
            if validation_mask_value is None
            else torch.as_tensor(validation_mask_value, dtype=torch.bool, device=device)
        )
        validation_active = _active_mask(validation_resources, validation_mask)
        model.eval()
        validation_memberships = compute_membership_series(
            model,
            validation_resources,
            validation_active,
        ).detach()
        validation_topology_value = validation.get("topology")
        validation_links_value = validation.get("link_features")
        validation_topology = (
            None
            if validation_topology_value is None
            else torch.as_tensor(validation_topology_value, dtype=torch.float32, device=device)
        )
        validation_links = (
            None
            if validation_links_value is None
            else torch.as_tensor(validation_links_value, dtype=torch.float32, device=device)
        )
        validation_origins = _origins(
            validation_resources.shape[1],
            model.config.history_window,
            training_config,
        )
        validation_payload = (
            validation_resources,
            validation_times,
            validation_active,
            validation_memberships,
            validation_topology,
            validation_links,
            validation_origins,
        )

    result = TrainingResult(model=model, training_memberships=training_memberships.detach().cpu())
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    start_epoch = 1
    resume_path = (
        None
        if training_config.resume_checkpoint_path is None
        else Path(training_config.resume_checkpoint_path)
    )
    checkpoint_config = {
        "model": vars(model.config),
        "run_identity_digest": training_config.run_identity_digest,
        "training": {
            key: value
            for key, value in vars(training_config).items()
            if key not in {
                "checkpoint_path",
                "resume_checkpoint_path",
                "resume_existing",
                "run_identity_digest",
                "relevant_data_digest",
                "run_id",
            }
        },
    }
    checkpoint_data = {
        "resource_shape": tuple(int(value) for value in resource_tensor.shape),
        "time_start": float(time_tensor.min().item()),
        "time_stop": float(time_tensor.max().item()),
        "active_count": int(active.sum().item()),
        "relevant_data_digest": training_config.relevant_data_digest,
    }
    if training_config.resume_existing and resume_path is not None and resume_path.exists():
        loaded = load_training_checkpoint(
            resume_path,
            expected_config=checkpoint_config,
            expected_data=checkpoint_data,
        )
        model_bundle = loaded.state["model"]
        model.load_state_dict(_model_state_from_numpy(model_bundle["current"], device))
        model.eval()
        training_memberships = compute_membership_series(model, resource_tensor, active).detach()
        model.fit_backoff(training_memberships, time_tensor, active)
        model.set_transition_prior_from_memberships(training_memberships, active)
        result.training_memberships = training_memberships.detach().cpu()
        if validation_payload is not None:
            refreshed_validation_memberships = compute_membership_series(
                model,
                validation_payload[0],
                validation_payload[2],
            ).detach()
            validation_payload = (
                validation_payload[0],
                validation_payload[1],
                validation_payload[2],
                refreshed_validation_memberships,
                validation_payload[4],
                validation_payload[5],
                validation_payload[6],
            )
        best_state = _model_state_from_numpy(model_bundle["best"], device)
        optimizer_payload = loaded.state.get("optimizer")
        if optimizer_payload is not None:
            optimizer.load_state_dict(_optimizer_state_from_numpy(optimizer_payload, device))
        metadata = loaded.metadata.extra
        result.history = [dict(item) for item in metadata.get("history", [])]
        result.best_epoch = int(metadata.get("best_epoch", 0))
        result.best_validation_loss = float(metadata.get("best_validation_loss", float("inf")))
        stale_epochs = int(metadata.get("stale_epochs", 0))
        start_epoch = int(loaded.metadata.epoch or 0) + 1
    model.backoff.freeze_for_queries()
    training_backoff_predictions = _precompute_backoff_predictions(
        model,
        training_memberships,
        time_tensor,
        train_origins,
    )
    validation_backoff_predictions: dict[int, BackoffPrediction] | None = None
    if validation_payload is not None:
        validation_backoff_predictions = _precompute_backoff_predictions(
            model,
            validation_payload[3],
            validation_payload[1],
            validation_payload[6],
        )
    # The formal recursive evaluator rebuilds BackOff from each origin's
    # leakage-safe L-sized history.  Training now uses the immutable caches
    # above, so retaining millions of Python event objects would only increase
    # RAM and make every best-model checkpoint unnecessarily large.
    model.backoff.clear()

    generator = np.random.default_rng(training_config.random_state)
    for epoch in range(start_epoch, training_config.epochs + 1):
        generator.shuffle(train_origins)
        model.train()
        train_metrics = _run_origins(
            model,
            resource_tensor,
            time_tensor,
            active,
            training_memberships,
            topology_tensor,
            link_tensor,
            train_origins,
            training_config,
            optimizer,
            training_backoff_predictions,
        )
        metrics = {f"train_{name}": value for name, value in train_metrics.items()}
        if validation_payload is not None:
            model.eval()
            with torch.no_grad():
                validation_metrics = _run_origins(
                    model,
                    validation_payload[0],
                    validation_payload[1],
                    validation_payload[2],
                    validation_payload[3],
                    validation_payload[4],
                    validation_payload[5],
                    validation_payload[6],
                    training_config,
                    None,
                    validation_backoff_predictions,
                )
            metrics.update({f"validation_{name}": value for name, value in validation_metrics.items()})
            monitored = validation_metrics["loss"]
        else:
            monitored = train_metrics["loss"]
        metrics["epoch"] = float(epoch)
        result.history.append(metrics)

        if monitored < result.best_validation_loss - training_config.min_improvement:
            result.best_validation_loss = monitored
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            if training_config.checkpoint_path is not None:
                model.save(
                    training_config.checkpoint_path,
                    metadata={"epoch": epoch, "metrics": metrics},
                )
        else:
            stale_epochs += 1
        if resume_path is not None:
            save_training_checkpoint(
                resume_path,
                model_state={
                    "current": _tensor_tree_to_numpy(model.state_dict()),
                    "best": _tensor_tree_to_numpy(best_state),
                },
                optimizer_state=_tensor_tree_to_numpy(optimizer.state_dict()),
                rng_state={
                    "torch": torch.get_rng_state().cpu().numpy(),
                },
                epoch=epoch,
                global_step=epoch * len(train_origins),
                config=checkpoint_config,
                data=checkpoint_data,
                run_id=training_config.run_id or "seed-7",
                extra={
                    "history": result.history,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "stale_epochs": stale_epochs,
                },
            )
        _invoke_epoch_callback(epoch_callback, epoch, metrics, model)
        if stale_epochs >= training_config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    result.model = model
    return result


__all__ = [
    "EpochCallback",
    "TrainingConfig",
    "TrainingResult",
    "compute_membership_series",
    "initialize_fixed_k_prototypes",
    "multi_order_adaptive_soft_cross_entropy",
    "soft_cross_entropy",
    "train_model",
]
