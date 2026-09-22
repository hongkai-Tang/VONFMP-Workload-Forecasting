from __future__ import annotations

"""Direct multi-step version of the Alibaba Ours predictor.

The fixed-K fuzzy memberships are computed once from the training split by
``data.py`` and are shared by every L/H task.  This module keeps the predictive
branches of Ours explicit: Einstein neighbour modulation, synchronous
multi-hop graph propagation, a physical-time dynamic transition, and the
variable-order physical-time BackOff prior.  The only forecasting change is
that every t+h is queried from the same real history ending at t; predictions
are never appended to the history or fed into a later horizon.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _simplex(values: Tensor, eps: float) -> Tensor:
    values = torch.clamp(values, min=0.0)
    total = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / max(values.shape[-1], 1))
    return torch.where(total > eps, values / total.clamp_min(eps), uniform)


@dataclass(frozen=True)
class DirectModelConfig:
    input_dim: int
    resource_dim: int
    num_states: int
    history_length: int
    horizon_steps: int
    workload_count: int
    max_order: int = 3
    backoff_decay: float = 0.02
    backoff_time_unit_steps: float = 1.0
    backoff_smoothing: float = 1.0
    backoff_threshold: float = 2.0
    backoff_temperature: float = 1.0
    backoff_blend: float = 0.5
    time_period_steps: float = 1440.0
    einstein_strength: float = 0.5
    message_passing_steps: int = 2
    message_hidden_dim: int = 8
    transition_hidden_dim: int = 32
    transition_horizon_chunk: int = 240
    dropout: float = 0.0
    time_step_seconds: int = 60
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.resource_dim <= 0 or self.num_states <= 1:
            raise ValueError("resource_dim must be positive and num_states must exceed one")
        if self.input_dim != self.resource_dim + self.num_states + 1:
            raise ValueError("input_dim must equal R resources + K memberships + one valid flag")
        if self.history_length <= 0 or self.horizon_steps <= 0:
            raise ValueError("history_length and horizon_steps must be positive")
        if not 1 <= self.max_order <= self.history_length:
            raise ValueError("max_order must be in [1, history_length]")
        if self.backoff_decay < 0.0 or self.backoff_time_unit_steps <= 0.0:
            raise ValueError("BackOff decay must be non-negative and its time unit positive")
        if self.backoff_smoothing < 0.0 or self.backoff_temperature <= 0.0:
            raise ValueError("BackOff smoothing must be non-negative and temperature positive")
        if not 0.0 <= self.backoff_blend <= 1.0:
            raise ValueError("backoff_blend must be in [0, 1]")
        if not 0.0 <= self.einstein_strength < 1.0:
            raise ValueError("einstein_strength must be in [0, 1)")
        if self.message_passing_steps < 0 or self.transition_horizon_chunk <= 0:
            raise ValueError("message steps must be non-negative and transition chunk positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_workload_adjacency(deployment: Tensor, workloads: int) -> Tensor:
    """Create the observed topology-only workload adjacency.

    Deployment is a node-workload incidence matrix.  No synthetic link-quality
    attributes are introduced; co-located workloads receive a unit edge.
    """

    deployment = torch.as_tensor(deployment, dtype=torch.float32)
    if deployment.ndim != 2 or deployment.shape[1] != workloads:
        raise ValueError("deployment must have shape (nodes, workloads)")
    adjacency = deployment.clamp_min(0).transpose(0, 1) @ deployment.clamp_min(0)
    adjacency.fill_diagonal_(0.0)
    return (adjacency > 0).to(torch.float32)


class FuzzyGraphMessagePassing(nn.Module):
    """Einstein neighbour modulation followed by synchronous graph updates."""

    def __init__(self, config: DirectModelConfig, adjacency: Tensor) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("adjacency", adjacency)
        self.theta = nn.Parameter(torch.zeros(config.num_states + 1, config.num_states))
        self.input_projection = nn.Linear(
            config.num_states + config.resource_dim,
            config.message_hidden_dim,
        )
        self.self_layers = nn.ModuleList(
            [
                nn.Linear(config.message_hidden_dim, config.message_hidden_dim)
                for _ in range(config.message_passing_steps)
            ]
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Linear(config.message_hidden_dim, config.message_hidden_dim, bias=False)
                for _ in range(config.message_passing_steps)
            ]
        )
        self.residual_head = nn.Linear(config.message_hidden_dim, config.num_states)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        memberships: Tensor,
        resources: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if memberships.ndim != 3:
            raise ValueError("memberships must have shape (batch, workloads, K)")
        adjacency = self.adjacency.to(device=memberships.device, dtype=memberships.dtype)
        pair_active = active[:, :, None] & active[:, None, :]
        batch_adjacency = adjacency[None, :, :] * pair_active.to(memberships.dtype)
        row_total = batch_adjacency.sum(dim=-1, keepdim=True)
        normalized = batch_adjacency / row_total.clamp_min(self.config.eps)
        has_neighbour = row_total > self.config.eps

        neighbour_membership = torch.einsum("bnm,bmk->bnk", normalized, memberships)
        degree = (batch_adjacency > 0).sum(dim=-1, keepdim=True).to(memberships.dtype)
        degree = degree / max(self.config.workload_count - 1, 1)
        modulation = torch.tanh(
            torch.cat([neighbour_membership, degree], dim=-1) @ self.theta
        )
        centered = 2.0 * memberships - 1.0
        influence = (
            float(self.config.einstein_strength)
            * modulation
            * has_neighbour.to(memberships.dtype)
        )
        denominator = (1.0 + centered * influence).clamp_min(self.config.eps)
        adjusted = _simplex(0.5 * (1.0 + (centered + influence) / denominator), self.config.eps)

        hidden = torch.tanh(
            self.input_projection(torch.cat([adjusted, resources], dim=-1))
        )
        for self_layer, message_layer in zip(self.self_layers, self.message_layers):
            message = torch.einsum(
                "bnm,bmd->bnd", normalized, message_layer(hidden)
            )
            hidden = self.dropout(torch.tanh(self_layer(hidden) + message))
        residual = self.residual_head(hidden) * has_neighbour.to(memberships.dtype)
        propagated = torch.softmax(
            torch.log(adjusted.clamp_min(self.config.eps)) + residual,
            dim=-1,
        )
        return propagated, {
            "neighbour_count": (batch_adjacency > 0).sum(dim=-1),
            "modulation_l1": (propagated - memberships).abs().sum(dim=-1),
            "message_context_norm": hidden.norm(dim=-1),
        }


class DynamicTransition(nn.Module):
    """Membership- and physical-time-conditioned stochastic transition."""

    def __init__(self, config: DirectModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.num_states + 2, config.transition_hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transition_hidden_dim, config.num_states * config.num_states),
        )
        self.register_buffer(
            "transition_prior",
            torch.full(
                (config.num_states, config.num_states),
                1.0 / config.num_states,
            ),
        )

    @torch.no_grad()
    def set_prior_from_memberships(
        self,
        memberships: Tensor,
        valid: Tensor,
        smoothing: float = 1.0,
    ) -> None:
        if memberships.ndim != 3 or memberships.shape[-1] != self.config.num_states:
            raise ValueError("memberships must have shape (workloads, time, K)")
        if valid.shape != memberships.shape[:2]:
            raise ValueError("valid must have shape (workloads, time)")
        active = valid[:, :-1].bool() & valid[:, 1:].bool()
        if bool(active.any()):
            previous = memberships[:, :-1][active].to(torch.float64)
            following = memberships[:, 1:][active].to(torch.float64)
            counts = previous.transpose(0, 1) @ following
        else:
            counts = torch.zeros(
                self.config.num_states,
                self.config.num_states,
                dtype=torch.float64,
                device=memberships.device,
            )
        counts = counts + float(smoothing)
        prior = counts / counts.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        self.transition_prior.copy_(prior.to(self.transition_prior))

    def forward(self, memberships: Tensor, origin_time_seconds: Tensor) -> Tensor:
        batch, workloads, states = memberships.shape
        if origin_time_seconds.shape != (batch,):
            raise ValueError("origin_time_seconds must have shape (batch,)")
        prior_logits = torch.log(
            self.transition_prior.to(memberships).clamp_min(self.config.eps)
        ).view(1, 1, 1, states, states)
        origin_steps = origin_time_seconds.to(memberships) / float(
            self.config.time_step_seconds
        )
        outputs: list[Tensor] = []
        chunk_size = int(self.config.transition_horizon_chunk)
        for start in range(0, self.config.horizon_steps, chunk_size):
            stop = min(start + chunk_size, self.config.horizon_steps)
            horizon = torch.arange(
                start + 1,
                stop + 1,
                device=memberships.device,
                dtype=memberships.dtype,
            )
            forecast_steps = origin_steps[:, None] + horizon[None, :]
            phase = 2.0 * torch.pi * forecast_steps / float(
                self.config.time_period_steps
            )
            base = memberships[:, :, None, :].expand(-1, -1, stop - start, -1)
            sin_phase = torch.sin(phase)[:, None, :, None].expand(-1, workloads, -1, -1)
            cos_phase = torch.cos(phase)[:, None, :, None].expand(-1, workloads, -1, -1)
            features = torch.cat([base, sin_phase, cos_phase], dim=-1)
            learned = self.network(features).view(
                batch,
                workloads,
                stop - start,
                states,
                states,
            )
            matrix = torch.softmax(learned + prior_logits, dim=-1)
            outputs.append(torch.einsum("bnk,bnhkj->bnhj", memberships, matrix))
        return _simplex(torch.cat(outputs, dim=2), self.config.eps)


class DirectPhysicalTimeBackoff(nn.Module):
    """Vectorized direct queries for the original physical-time BackOff rule.

    For every horizon, counts are built only from the real L-slot history.
    The old recursive implementation appended each predicted state before the
    next query; this implementation deliberately never performs that append.
    """

    def __init__(self, config: DirectModelConfig) -> None:
        super().__init__()
        self.config = config

    @torch.no_grad()
    def _order_distribution(
        self,
        memberships: Tensor,
        valid: Tensor,
        hard_states: Tensor,
        order: int,
    ) -> tuple[Tensor, Tensor]:
        batch, workloads, length, states = memberships.shape
        contexts = states**order if order else 1
        event_length = length
        targets = memberships
        if order:
            # Match PhysicalTimeBackoff.fit exactly when slots are missing:
            # contexts use the previous *valid* events, not necessarily the
            # previous contiguous physical slots.
            valid_long = valid.to(torch.long)
            previous_valid_count = valid_long.cumsum(dim=-1) - valid_long
            compressed = torch.zeros(
                batch * workloads * length,
                dtype=torch.long,
                device=memberships.device,
            )
            series_offset = (
                torch.arange(
                    batch * workloads,
                    device=memberships.device,
                    dtype=torch.long,
                ).view(batch, workloads, 1)
                * length
            )
            compressed_indices = series_offset + previous_valid_count
            flat_valid = valid.reshape(-1)
            compressed[compressed_indices.reshape(-1)[flat_valid]] = hard_states.reshape(-1)[
                flat_valid
            ]
            compressed = compressed.view(batch, workloads, length)
            codes = torch.zeros(
                (batch, workloads, event_length),
                dtype=torch.long,
                device=memberships.device,
            )
            for offset in range(order):
                compressed_position = (
                    previous_valid_count - order + offset
                ).clamp(min=0, max=length - 1)
                context_state = torch.gather(
                    compressed,
                    dim=-1,
                    index=compressed_position,
                )
                codes = codes * states + context_state
            event_valid = valid & (previous_valid_count >= order)
            total_valid = valid_long.sum(dim=-1)
            query_codes = torch.zeros(
                (batch, workloads),
                dtype=torch.long,
                device=memberships.device,
            )
            for offset in range(order):
                compressed_position = (total_valid - order + offset).clamp(
                    min=0,
                    max=length - 1,
                )
                query_codes = query_codes * states + torch.gather(
                    compressed,
                    dim=-1,
                    index=compressed_position.unsqueeze(-1),
                ).squeeze(-1)
            query_valid = total_valid >= order
        else:
            codes = torch.zeros(
                (batch, workloads, event_length),
                dtype=torch.long,
                device=memberships.device,
            )
            query_codes = torch.zeros(
                (batch, workloads),
                dtype=torch.long,
                device=memberships.device,
            )
            query_valid = torch.ones_like(query_codes, dtype=torch.bool)
            event_valid = valid

        event_positions = torch.arange(
            0,
            length,
            device=memberships.device,
            dtype=torch.long,
        )
        event_age = (length - 1 - event_positions).to(memberships.dtype)
        reference_decay = torch.exp(
            -float(self.config.backoff_decay)
            * event_age
            / float(self.config.backoff_time_unit_steps)
        )
        weighted_targets = targets * reference_decay.view(1, 1, event_length, 1)

        batch_offsets = (
            torch.arange(batch, device=memberships.device, dtype=torch.long)
            .view(batch, 1, 1)
            * contexts
            * length
        )
        group_indices = (
            batch_offsets
            + codes * length
            + event_positions.view(1, 1, event_length)
        )
        grouped = torch.zeros(
            (batch * contexts * length, states),
            device=memberships.device,
            dtype=memberships.dtype,
        )
        flat_valid = event_valid.reshape(-1)
        if bool(flat_valid.any()):
            grouped.index_add_(
                0,
                group_indices.reshape(-1)[flat_valid],
                weighted_targets.reshape(-1, states)[flat_valid],
            )
        # At t+h, BackOff max_age=L retains events whose history position is
        # j >= h-1.  A reverse cumulative sum makes all horizons one gather.
        tail_counts = grouped.view(batch, contexts, length, states)
        tail_counts = tail_counts.flip(2).cumsum(2).flip(2)

        horizon = torch.arange(
            1,
            self.config.horizon_steps + 1,
            device=memberships.device,
            dtype=torch.long,
        )
        start_positions = horizon - 1
        active_horizon = start_positions < length
        safe_positions = start_positions.clamp(max=length - 1)
        query_indices = (
            torch.arange(batch, device=memberships.device, dtype=torch.long)
            .view(batch, 1, 1)
            * contexts
            * length
            + query_codes[:, :, None] * length
            + safe_positions.view(1, 1, -1)
        )
        counts = tail_counts.reshape(-1, states)[query_indices]
        forecast_decay = torch.exp(
            -float(self.config.backoff_decay)
            * horizon.to(memberships.dtype)
            / float(self.config.backoff_time_unit_steps)
        )
        counts = counts * forecast_decay.view(1, 1, -1, 1)
        counts = counts * active_horizon.view(1, 1, -1, 1)
        counts = counts * query_valid[:, :, None, None]
        support = counts.sum(dim=-1)
        smoothed = counts + float(self.config.backoff_smoothing)
        distribution = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        return distribution, support

    @torch.no_grad()
    def forward(self, memberships: Tensor, valid: Tensor) -> dict[str, Tensor]:
        if memberships.ndim != 4:
            raise ValueError("memberships must have shape (batch, workloads, L, K)")
        if valid.shape != memberships.shape[:3]:
            raise ValueError("valid must have shape (batch, workloads, L)")
        if memberships.shape[2] != self.config.history_length:
            raise ValueError("history width does not match configured L")
        memberships = _simplex(torch.nan_to_num(memberships), self.config.eps)
        hard_states = memberships.argmax(dim=-1)
        distributions: list[Tensor] = []
        supports: list[Tensor] = []
        for order in range(self.config.max_order + 1):
            distribution, support = self._order_distribution(
                memberships,
                valid.bool(),
                hard_states,
                order,
            )
            distributions.append(distribution)
            supports.append(support)
        order_distributions = torch.stack(distributions, dim=-2)
        support_tensor = torch.stack(supports, dim=-1)
        gates = torch.zeros_like(support_tensor)
        gates[..., 1:] = torch.sigmoid(
            (support_tensor[..., 1:] - float(self.config.backoff_threshold))
            / float(self.config.backoff_temperature)
        )
        weights = torch.zeros_like(gates)
        for order in range(1, self.config.max_order + 1):
            higher_failure = torch.prod(1.0 - gates[..., order + 1 :], dim=-1)
            weights[..., order] = gates[..., order] * higher_failure
        weights[..., 0] = torch.prod(1.0 - gates[..., 1:], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        distribution = torch.sum(order_distributions * weights[..., None], dim=-2)
        return {
            "distribution": _simplex(distribution, self.config.eps),
            "order_distributions": order_distributions,
            "supports": support_tensor,
            "gates": gates,
            "order_weights": weights,
            "effective_order": weights.argmax(dim=-1),
        }


class DirectMembershipForecaster(nn.Module):
    """Ours predictor that emits all future slots without recursive feedback."""

    def __init__(self, config: DirectModelConfig, deployment: Tensor) -> None:
        super().__init__()
        self.config = config
        adjacency = normalized_workload_adjacency(deployment, config.workload_count)
        self.graph_block = FuzzyGraphMessagePassing(config, adjacency)
        self.dynamic_transition = DynamicTransition(config)
        self.backoff = DirectPhysicalTimeBackoff(config)

    @torch.no_grad()
    def set_transition_prior_from_memberships(
        self,
        memberships: Tensor,
        valid: Tensor,
    ) -> None:
        self.dynamic_transition.set_prior_from_memberships(memberships, valid)

    def forward(
        self,
        history_features: Tensor,
        history_valid: Tensor,
        origin_time_seconds: Tensor,
        *,
        return_components: bool = False,
    ) -> Tensor | dict[str, Any]:
        if history_features.ndim != 4:
            raise ValueError("history_features must have shape (batch, workloads, L, features)")
        if history_valid.shape != history_features.shape[:3]:
            raise ValueError("history_valid must have shape (batch, workloads, L)")
        batch, workloads, length, features = history_features.shape
        if workloads != self.config.workload_count or length != self.config.history_length:
            raise ValueError("history workload count or length does not match model configuration")
        if features != self.config.input_dim:
            raise ValueError("history feature width does not match model configuration")

        resource_stop = self.config.resource_dim
        membership_stop = resource_stop + self.config.num_states
        history_resources = torch.nan_to_num(history_features[..., :resource_stop])
        history_memberships = history_features[..., resource_stop:membership_stop]
        base_membership = _simplex(history_memberships[:, :, -1, :], self.config.eps)
        current_resources = history_resources[:, :, -1, :]
        current_active = history_valid[:, :, -1].bool()
        spatial, graph_diagnostics = self.graph_block(
            base_membership,
            current_resources,
            current_active,
        )
        dynamic = self.dynamic_transition(spatial, origin_time_seconds)
        backoff = self.backoff(history_memberships, history_valid)
        blend = float(self.config.backoff_blend)
        final = _simplex(
            (1.0 - blend) * dynamic + blend * backoff["distribution"],
            self.config.eps,
        )
        if not return_components:
            return final
        return {
            "final": final,
            "dynamic": dynamic,
            "backoff": backoff["distribution"],
            "base_membership": base_membership,
            "spatial_membership": spatial,
            "graph_diagnostics": graph_diagnostics,
            "backoff_diagnostics": {
                name: value
                for name, value in backoff.items()
                if name != "distribution"
            },
        }


def _soft_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    active: Tensor,
    eps: float,
) -> Tensor:
    normalized_target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    losses = -(normalized_target * torch.log(prediction.clamp_min(eps))).sum(dim=-1)
    return losses[active].mean()


def direct_ours_loss(
    outputs: Mapping[str, Any],
    target: Tensor,
    valid: Tensor,
    *,
    dynamic_weight: float = 0.25,
    adaptive_order_weight: float = 0.25,
    brier_weight: float = 0.0,
    blend: float = 0.5,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    prediction = outputs["final"]
    if prediction.shape != target.shape or valid.shape != prediction.shape[:-1]:
        raise ValueError("prediction/target/valid shapes are incompatible")
    active = valid.bool() & torch.isfinite(target).all(dim=-1)
    if not bool(active.any()):
        raise ValueError("batch contains no valid future targets")
    normalized_target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    final_loss = _soft_cross_entropy(prediction, normalized_target, active, eps)
    dynamic_loss = _soft_cross_entropy(outputs["dynamic"], normalized_target, active, eps)

    diagnostics = outputs["backoff_diagnostics"]
    order_distributions = diagnostics["order_distributions"].detach()
    order_weights = diagnostics["order_weights"].detach()
    dynamic_expanded = outputs["dynamic"].unsqueeze(-2).expand_as(order_distributions)
    adaptive_predictions = (
        (1.0 - float(blend)) * dynamic_expanded
        + float(blend) * order_distributions
    )
    order_losses = -(
        normalized_target.unsqueeze(-2)
        * torch.log(adaptive_predictions.clamp_min(eps))
    ).sum(dim=-1)
    adaptive_loss = (order_losses * order_weights).sum(dim=-1)[active].mean()
    brier = (prediction - normalized_target).square().sum(dim=-1)[active].mean()
    loss = (
        final_loss
        + float(dynamic_weight) * dynamic_loss
        + float(adaptive_order_weight) * adaptive_loss
        + float(brier_weight) * brier
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "final_soft_cross_entropy": float(final_loss.detach().cpu()),
        "dynamic_soft_cross_entropy": float(dynamic_loss.detach().cpu()),
        "adaptive_order_soft_cross_entropy": float(adaptive_loss.detach().cpu()),
        "brier_soft": float(brier.detach().cpu()),
        "valid_targets": int(active.sum().detach().cpu()),
    }


def direct_membership_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    brier_weight: float = 0.25,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Compatibility loss retained for simple model/unit tests."""

    if prediction.shape != target.shape or valid.shape != prediction.shape[:-1]:
        raise ValueError("prediction/target/valid shapes are incompatible")
    active = valid.bool() & torch.isfinite(target).all(dim=-1)
    if not bool(active.any()):
        raise ValueError("batch contains no valid future targets")
    normalized_target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    cross_entropy = -(
        normalized_target * torch.log(prediction.clamp_min(eps))
    ).sum(dim=-1)
    brier = (prediction - normalized_target).square().sum(dim=-1)
    ce_loss = cross_entropy[active].mean()
    brier_loss = brier[active].mean()
    loss = ce_loss + float(brier_weight) * brier_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "soft_cross_entropy": float(ce_loss.detach().cpu()),
        "brier_soft": float(brier_loss.detach().cpu()),
        "valid_targets": int(active.sum().detach().cpu()),
    }


__all__ = [
    "DirectMembershipForecaster",
    "DirectModelConfig",
    "DirectPhysicalTimeBackoff",
    "direct_membership_loss",
    "direct_ours_loss",
    "normalized_workload_adjacency",
]
