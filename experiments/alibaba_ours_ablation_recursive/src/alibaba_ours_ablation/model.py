from __future__ import annotations

"""Exactly-scoped model substitutions used by the ablation experiment."""

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from alibaba_ours_exp.model import (
    AlibabaOursModel,
    OursModelConfig,
    _simplex,
    _workload_graph,
)


ABLATION_VARIANTS = (
    "ns_mem",
    "hypergraph",
    "dyn_trans",
    "vo_markov",
    "multi_hop",
)

DISPLAY_NAMES = {
    "ns_mem": "NS-Mem",
    "hypergraph": "HyperGraph",
    "dyn_trans": "Dyn-Trans",
    "vo_markov": "VO-Markov",
    "multi_hop": "Multi-Hop",
}

_ALIASES = {
    "ns-mem": "ns_mem",
    "nsmem": "ns_mem",
    "hyper-graph": "hypergraph",
    "dyn-trans": "dyn_trans",
    "dyntrans": "dyn_trans",
    "vo-markov": "vo_markov",
    "vomarkov": "vo_markov",
    "multi-hop": "multi_hop",
    "multihop": "multi_hop",
}


def canonical_variant(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in ABLATION_VARIANTS:
        allowed = ", ".join(ABLATION_VARIANTS)
        raise ValueError(f"unknown ablation variant {value!r}; choose one of: {allowed}")
    return normalized


def pairwise_adjacency(deployment: np.ndarray) -> np.ndarray:
    """Ordinary unweighted pair graph induced by shared deployment nodes."""

    incidence = (np.asarray(deployment) > 0).astype(np.float32)
    if incidence.ndim != 2:
        raise ValueError("deployment must have shape (nodes, workloads)")
    workloads = incidence.shape[1]
    adjacency = np.zeros((workloads, workloads), dtype=np.float32)
    # Avoid invoking a second OpenMP/BLAS runtime after PyTorch on Windows.
    for edge in incidence:
        members = np.flatnonzero(edge)
        if members.size > 1:
            adjacency[np.ix_(members, members)] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def hypergraph_adjacency(deployment: np.ndarray) -> np.ndarray:
    """Cardinality-normalized clique expansion retaining hyperedge structure.

    Each deployment node is a workload hyperedge.  A node containing ``d``
    workloads contributes ``1/(d-1)`` to every off-diagonal pair, so a large
    co-location group is not treated as ``d*(d-1)`` independent observations.
    """

    incidence = (np.asarray(deployment) > 0).astype(np.float32)
    if incidence.ndim != 2:
        raise ValueError("deployment must have shape (nodes, workloads)")
    workloads = incidence.shape[1]
    adjacency = np.zeros((workloads, workloads), dtype=np.float32)
    for edge in incidence:
        members = np.flatnonzero(edge)
        if members.size > 1:
            adjacency[np.ix_(members, members)] += 1.0 / float(members.size - 1)
    np.fill_diagonal(adjacency, 0.0)
    maximum = float(adjacency.max(initial=0.0))
    if maximum > 0.0:
        adjacency /= maximum
    return adjacency.astype(np.float32, copy=False)


@dataclass
class AblationModelConfig(OursModelConfig):
    variant: str = "ns_mem"
    max_forecast_horizon: int = 1728
    direct_hidden_dim: int = 32
    membership_definition: str = "per_slot"
    diagnostic_max_order: int = 3

    def __post_init__(self) -> None:
        self.variant = canonical_variant(self.variant)
        if self.max_forecast_horizon <= 0:
            raise ValueError("max_forecast_horizon must be positive")
        if self.direct_hidden_dim <= 0:
            raise ValueError("direct_hidden_dim must be positive")
        if self.membership_definition != "per_slot":
            raise ValueError("ablation memberships must be defined per slot")
        if self.variant == "vo_markov":
            self.max_order = 1
        if self.variant == "multi_hop":
            self.message_passing_steps = 1
        super().__post_init__()
        if self.diagnostic_max_order < self.max_order:
            raise ValueError("diagnostic_max_order must cover the effective max_order")


class AblationModel(AlibabaOursModel):
    """Ours model with one and only one mechanism replaced."""

    config: AblationModelConfig

    def __init__(self, config: AblationModelConfig | None = None) -> None:
        super().__init__(config or AblationModelConfig())
        hidden = self.config.direct_hidden_dim
        self.direct_origin_projection = nn.Linear(self.config.num_states, hidden)
        self.direct_horizon_projection = nn.Linear(4, hidden)
        self.direct_output = nn.Linear(hidden, self.config.num_states)

    def encode_slot_memberships(
        self,
        resources: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode every 60-second slot independently with the shared K prototypes."""

        if resources.ndim != 3 or resources.shape[-1] != self.config.resource_dim:
            raise ValueError("resources must have shape (workloads, slots, resource_dim)")
        if mask is not None and mask.shape != resources.shape[:2]:
            raise ValueError("mask must have shape (workloads, slots)")
        workloads, slots, resource_dim = resources.shape
        flattened = resources.reshape(workloads * slots, 1, resource_dim)
        flattened_mask = None if mask is None else mask.reshape(workloads * slots, 1)
        encoded = self.shape_encoder(flattened, flattened_mask)
        return encoded.reshape(workloads, slots, self.config.num_states)

    def encode_membership(self, history: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return the membership of the final slot, never an L-slot aggregate label."""

        if history.ndim != 3 or history.shape[1] == 0:
            raise ValueError("history must contain at least one slot")
        selected_mask = None if mask is None else mask[:, -1:]
        return self.encode_slot_memberships(history[:, -1:, :], selected_mask)[:, 0]

    def encode_history_memberships(
        self,
        history: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.encode_slot_memberships(history, mask)

    def _spatial_distribution(
        self,
        memberships: Tensor,
        resource_representation: Tensor,
        topology: Tensor | None,
        link_features: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.variant != "ns_mem":
            return self.graph_block(
                memberships,
                resource_representation,
                topology,
                link_features,
            )

        # NS-Mem: target membership depends only on its own historical state.
        workloads = memberships.shape[0]
        adjacency = _workload_graph(
            topology,
            link_features,
            workloads,
            memberships.dtype,
            memberships.device,
        )
        zeros = torch.zeros(workloads, device=memberships.device, dtype=memberships.dtype)
        return memberships, {
            "adjacency": adjacency,
            "normalized_adjacency": torch.zeros_like(adjacency),
            "neighbour_count": (adjacency > 0).sum(dim=1),
            "link_gate_mean": torch.full_like(zeros, float("nan")),
            "link_features_present": torch.zeros(workloads, dtype=torch.bool, device=memberships.device),
            "unweighted_topology": torch.full(
                (workloads,), topology is not None, dtype=torch.bool, device=memberships.device
            ),
            "modulation": torch.zeros_like(memberships),
            "modulation_l1": zeros,
            "message_context_norm": zeros,
        }

    def _dynamic_distribution(
        self,
        spatial_membership: Tensor,
        forecast_times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.variant != "dyn_trans":
            return self.dynamic_transition(spatial_membership, forecast_times)

        # Dyn-Trans: use one global row-stochastic matrix, independent of time
        # and current neighbourhood/load state.
        prior = self.dynamic_transition.transition_prior.to(spatial_membership)
        matrix = prior.unsqueeze(0).expand(spatial_membership.shape[0], -1, -1)
        distribution = torch.einsum("nk,nkj->nj", spatial_membership, matrix)
        return _simplex(distribution, self.config.eps), matrix

    def _first_order_backoff(
        self,
        distribution: Tensor,
        diagnostics: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.variant != "vo_markov":
            return distribution, dict(diagnostics)
        order_distributions = diagnostics["order_distributions"]
        supports = diagnostics["supports"]
        if order_distributions.shape[1] != 2:
            raise RuntimeError("VO-Markov requires exactly zeroth and first-order distributions")
        use_first = supports[:, 1] > 0
        selected = torch.where(
            use_first[:, None],
            order_distributions[:, 1, :],
            order_distributions[:, 0, :],
        )
        effective = use_first.to(dtype=torch.long)
        updated = dict(diagnostics)
        updated["effective_order"] = effective
        updated["order_weights"] = F.one_hot(effective, num_classes=2).to(selected.dtype)
        return _simplex(selected, self.config.eps), updated

    def step_from_membership(
        self,
        memberships: Tensor,
        hard_contexts: Tensor,
        forecast_times: Tensor,
        topology: Tensor | None = None,
        link_features: Tensor | None = None,
        resource_representation: Tensor | None = None,
        backoff_model: Any | None = None,
        backoff_prediction: tuple[Tensor, Mapping[str, Tensor]] | None = None,
    ) -> dict[str, Any]:
        memberships = _simplex(memberships, self.config.eps)
        if resource_representation is None:
            resource_representation = self.shape_encoder.decode_resources(memberships)
        spatial, graph_diagnostics = self._spatial_distribution(
            memberships,
            resource_representation,
            topology,
            link_features,
        )
        dynamic, transition_matrix = self._dynamic_distribution(spatial, forecast_times)
        selected_backoff = self.backoff if backoff_model is None else backoff_model
        if backoff_prediction is not None:
            backoff, supplied = backoff_prediction
            backoff = backoff.to(device=memberships.device, dtype=memberships.dtype)
            backoff_diagnostics = {
                name: value.to(device=memberships.device) for name, value in supplied.items()
            }
        elif selected_backoff.fitted:
            backoff, backoff_diagnostics = selected_backoff.predict_batch(
                hard_contexts,
                forecast_times,
                device=memberships.device,
                dtype=memberships.dtype,
            )
        else:
            backoff = memberships
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
        backoff, backoff_diagnostics = self._first_order_backoff(backoff, backoff_diagnostics)
        blend = float(self.config.backoff_blend)
        final = _simplex((1.0 - blend) * dynamic + blend * backoff, self.config.eps)
        return {
            "final": final,
            "dynamic": dynamic,
            "backoff": backoff,
            "spatial_membership": spatial,
            "transition_matrix": transition_matrix,
            "graph_diagnostics": graph_diagnostics,
            "backoff_diagnostics": backoff_diagnostics,
        }

    def direct_multi_step(
        self,
        origin_membership: Tensor,
        hard_contexts: Tensor,
        forecast_times: Tensor,
        horizon_steps: Tensor,
        topology: Tensor | None = None,
        link_features: Tensor | None = None,
        resource_representation: Tensor | None = None,
        backoff_model: Any | None = None,
    ) -> dict[str, Any]:
        """Predict multiple horizons independently from one observed origin.

        No output at horizon j is inserted into the input of another horizon.
        The graph, transition and BackOff branches therefore see the same true
        origin context for every requested future slot.
        """

        origin_membership = _simplex(origin_membership, self.config.eps)
        if forecast_times.ndim != 2 or forecast_times.shape[0] != origin_membership.shape[0]:
            raise ValueError("forecast_times must have shape (workloads, horizons)")
        if horizon_steps.ndim != 1 or horizon_steps.shape[0] != forecast_times.shape[1]:
            raise ValueError("horizon_steps must match the forecast time dimension")
        if int(horizon_steps.min().item()) < 1:
            raise ValueError("horizon steps must be positive")
        if int(horizon_steps.max().item()) > self.config.max_forecast_horizon:
            raise ValueError("horizon exceeds max_forecast_horizon")
        if hard_contexts.ndim != 2 or hard_contexts.shape[0] != origin_membership.shape[0]:
            raise ValueError("hard_contexts must have shape (workloads, context)")

        if resource_representation is None:
            resource_representation = self.shape_encoder.decode_resources(origin_membership)
        spatial, graph_diagnostics = self._spatial_distribution(
            origin_membership,
            resource_representation,
            topology,
            link_features,
        )

        workloads, horizons = forecast_times.shape
        repeated_spatial = spatial[:, None, :].expand(-1, horizons, -1)
        flat_spatial = repeated_spatial.reshape(workloads * horizons, self.config.num_states)
        flat_times = forecast_times.reshape(workloads * horizons)
        dynamic, _ = self._dynamic_distribution(flat_spatial, flat_times)

        selected_backoff = self.backoff if backoff_model is None else backoff_model
        repeated_contexts = hard_contexts[:, None, :].expand(-1, horizons, -1)
        flat_contexts = repeated_contexts.reshape(workloads * horizons, hard_contexts.shape[1])
        if selected_backoff.fitted:
            backoff, backoff_diagnostics = selected_backoff.predict_batch(
                flat_contexts,
                flat_times,
                device=origin_membership.device,
                dtype=origin_membership.dtype,
            )
        else:
            backoff = origin_membership[:, None, :].expand(-1, horizons, -1).reshape(
                workloads * horizons,
                self.config.num_states,
            )
            orders = self.config.max_order + 1
            batch = workloads * horizons
            backoff_diagnostics = {
                "order_distributions": backoff[:, None, :].expand(-1, orders, -1),
                "supports": torch.zeros(
                    batch, orders, device=backoff.device, dtype=backoff.dtype
                ),
                "gates": torch.zeros(
                    batch, orders, device=backoff.device, dtype=backoff.dtype
                ),
                "order_weights": F.one_hot(
                    torch.zeros(batch, dtype=torch.long, device=backoff.device),
                    num_classes=orders,
                ).to(backoff.dtype),
                "effective_order": torch.zeros(
                    batch, dtype=torch.long, device=backoff.device
                ),
            }
        backoff, backoff_diagnostics = self._first_order_backoff(
            backoff,
            backoff_diagnostics,
        )
        desired_orders = self.config.diagnostic_max_order + 1
        for name in ("order_distributions", "supports", "gates", "order_weights"):
            value = backoff_diagnostics[name]
            missing_orders = desired_orders - value.shape[1]
            if missing_orders > 0:
                padding_shape = list(value.shape)
                padding_shape[1] = missing_orders
                backoff_diagnostics[name] = torch.cat(
                    (
                        value,
                        torch.zeros(
                            *padding_shape,
                            device=value.device,
                            dtype=value.dtype,
                        ),
                    ),
                    dim=1,
                )

        blend = float(self.config.backoff_blend)
        base = _simplex((1.0 - blend) * dynamic + blend * backoff, self.config.eps)
        origin_hidden = self.direct_origin_projection(origin_membership)[:, None, :]
        normalized_h = horizon_steps.to(origin_membership.dtype) / float(
            self.config.max_forecast_horizon
        )
        horizon_features = torch.stack(
            (
                normalized_h,
                torch.log1p(horizon_steps.to(origin_membership.dtype))
                / math.log1p(self.config.max_forecast_horizon),
                torch.sin(2.0 * math.pi * normalized_h),
                torch.cos(2.0 * math.pi * normalized_h),
            ),
            dim=-1,
        )
        horizon_hidden = self.direct_horizon_projection(horizon_features)[None, :, :]
        correction = self.direct_output(torch.tanh(origin_hidden + horizon_hidden))
        final = torch.softmax(
            torch.log(base.reshape(workloads, horizons, -1).clamp_min(self.config.eps))
            + correction,
            dim=-1,
        )

        reshaped_backoff_diagnostics = {
            name: value.reshape(workloads, horizons, *value.shape[1:])
            for name, value in backoff_diagnostics.items()
        }
        return {
            "final": final,
            "dynamic": dynamic.reshape(workloads, horizons, -1),
            "backoff": backoff.reshape(workloads, horizons, -1),
            "spatial_membership": spatial,
            "graph_diagnostics": graph_diagnostics,
            "backoff_diagnostics": reshaped_backoff_diagnostics,
        }

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        """Save via a file handle so PyTorch 2.0 works with Unicode Windows paths."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            torch.save(
                {
                    "config": asdict(self.config),
                    "model_state": self.state_dict(),
                    "backoff_state": self.backoff.export_state(),
                    "metadata": dict(metadata or {}),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["AblationModel", dict[str, Any]]:
        with Path(path).open("rb") as handle:
            try:
                checkpoint = torch.load(
                    handle, map_location=map_location, weights_only=False
                )
            except TypeError:
                handle.seek(0)
                checkpoint = torch.load(handle, map_location=map_location)
        model = cls(AblationModelConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.backoff.import_state(checkpoint.get("backoff_state", {}))
        return model, dict(checkpoint.get("metadata", {}))


__all__ = [
    "ABLATION_VARIANTS",
    "DISPLAY_NAMES",
    "AblationModel",
    "AblationModelConfig",
    "canonical_variant",
    "hypergraph_adjacency",
    "pairwise_adjacency",
]
