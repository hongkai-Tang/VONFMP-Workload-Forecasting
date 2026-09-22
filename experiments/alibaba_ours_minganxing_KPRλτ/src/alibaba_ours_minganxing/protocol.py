from __future__ import annotations

"""Protocol patches that separate single-bucket labels from history length."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

import alibaba_ours_exp.cli as base_cli
import alibaba_ours_exp.model as base_model
import alibaba_ours_exp.training as base_training


LABEL_DEFINITION = "mu_t = fuzzy_encoder(resource_vector_at_t); L only bounds state context"


def encode_single_bucket_membership(
    self: base_model.AlibabaOursModel,
    history: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Encode only the final physical bucket, independent of configured L."""

    if history.ndim != 3 or history.shape[-1] != self.config.resource_dim:
        raise ValueError("history must have shape (workloads, time, resource_dim)")
    selected = history[:, -1:, :]
    selected_mask = None if mask is None else mask[:, -1:]
    return self.shape_encoder(selected, selected_mask)


def encode_single_bucket_history(
    self: base_model.AlibabaOursModel,
    history: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Encode one membership token per bucket without rolling resource windows."""

    if history.ndim != 3 or history.shape[-1] != self.config.resource_dim:
        raise ValueError("history must have shape (workloads, time, resource_dim)")
    history = history[:, -self.config.history_window :, :]
    selected_mask = None if mask is None else mask[:, -self.config.history_window :]
    workloads, steps, resources = history.shape
    flattened = history.reshape(workloads * steps, 1, resources)
    flattened_mask = (
        None if selected_mask is None else selected_mask.reshape(workloads * steps, 1)
    )
    memberships: list[Tensor] = []
    chunk_size = 8192
    for start in range(0, flattened.shape[0], chunk_size):
        stop = min(start + chunk_size, flattened.shape[0])
        memberships.append(
            self.shape_encoder(
                flattened[start:stop],
                None if flattened_mask is None else flattened_mask[start:stop],
            )
        )
    return torch.cat(memberships, dim=0).reshape(
        workloads, steps, self.config.num_states
    )


@torch.no_grad()
def compute_single_bucket_membership_series(
    model: base_model.AlibabaOursModel,
    resources: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Return mu_t for every t using only the resource vector at t."""

    if resources.ndim != 3 or resources.shape[-1] != model.config.resource_dim:
        raise ValueError("resources must have shape (workloads, time, resource_dim)")
    workloads, steps, resource_dim = resources.shape
    flattened = resources.reshape(workloads * steps, 1, resource_dim)
    flattened_mask = None if mask is None else mask.reshape(workloads * steps, 1)
    memberships: list[Tensor] = []
    chunk_size = 8192
    for start in range(0, flattened.shape[0], chunk_size):
        stop = min(start + chunk_size, flattened.shape[0])
        memberships.append(
            model.shape_encoder(
                flattened[start:stop],
                None if flattened_mask is None else flattened_mask[start:stop],
            )
        )
    return torch.cat(memberships, dim=0).reshape(
        workloads, steps, model.config.num_states
    )


@torch.no_grad()
def initialize_single_bucket_prototypes(
    model: base_model.AlibabaOursModel,
    resources: Tensor,
    mask: Tensor | None = None,
    *,
    max_candidates: int = 512,
    random_state: int | None = None,
) -> Tensor:
    """Initialize K fixed resource-mode prototypes from training buckets only."""

    if resources.ndim != 3 or resources.shape[-1] != model.config.resource_dim:
        raise ValueError("resources must have shape (workloads, time, resource_dim)")
    finite = torch.isfinite(resources).all(dim=-1)
    active = finite if mask is None else finite & mask.bool().to(resources.device)
    active_indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten().cpu().numpy()
    if active_indices.size == 0:
        raise ValueError("no valid training buckets are available for prototype initialization")
    generator = np.random.default_rng(
        model.config.random_state if random_state is None else int(random_state)
    )
    count = min(max(int(max_candidates), model.config.num_states), active_indices.size)
    selected_indices = generator.choice(active_indices, size=count, replace=False)
    tensor_indices = torch.as_tensor(
        selected_indices, device=resources.device, dtype=torch.long
    )
    points = resources.reshape(-1, resources.shape[-1])[tensor_indices].detach().cpu()
    first = int(generator.integers(0, points.shape[0]))
    selected = [first]
    nearest = (points - points[first]).square().mean(dim=-1)
    while len(selected) < model.config.num_states:
        next_index = int(torch.argmax(nearest).item())
        if next_index in selected and len(selected) == points.shape[0]:
            next_index = selected[len(selected) % len(selected)]
        selected.append(next_index)
        distance = (points - points[next_index]).square().mean(dim=-1)
        nearest = torch.minimum(nearest, distance)
    centers = points[selected[: model.config.num_states]].to(
        device=model.shape_encoder.prototypes.device,
        dtype=model.shape_encoder.prototypes.dtype,
    )
    prototypes = centers[:, None, :].expand(
        -1, model.config.prototype_length, -1
    ).contiguous()
    model.shape_encoder.set_prototypes(prototypes)
    return prototypes


def apply_single_bucket_protocol() -> None:
    """Apply idempotent process-local patches to the vendored base experiment."""

    base_model.AlibabaOursModel.encode_membership = encode_single_bucket_membership
    base_model.AlibabaOursModel.encode_history_memberships = encode_single_bucket_history
    base_training.compute_membership_series = compute_single_bucket_membership_series
    base_training.initialize_fixed_k_prototypes = initialize_single_bucket_prototypes
    base_cli.initialize_fixed_k_prototypes = initialize_single_bucket_prototypes
    base_cli.AlibabaOursModel = base_model.AlibabaOursModel


def protocol_manifest() -> dict[str, Any]:
    return {
        "membership_definition": LABEL_DEFINITION,
        "label_window_steps": 1,
        "history_affects_label_definition": False,
        "prediction_strategy": "direct_one_step_no_feedback",
        "predicted_outputs_are_reused": False,
        "prototype_source": "training_split_only_single_bucket_vectors",
        "normalization_source": "training_split_only",
        "neural_parameter_training_scope": "training_split_only",
        "validation_role": "early_stopping_only",
        "test_targets_used_for_training": False,
        "rolling_origin_policy": "fixed_L_observed_history_at_each_origin",
        "backoff_evaluation_policy": "causal_origin_local_fit_from_observed_L_history",
        "workload_cohort": "fixed_precomputed_alibaba_selected_200",
        "workload_cohort_selection_scope": "inherited_full_period_valid_row_count_ranking",
        "workload_selection_limitation": (
            "the supplied 200-workload portable cohort was preselected using full-period "
            "coverage; sensitivity comparisons share the same cohort, but this does not "
            "establish workload-selection-independent generalization"
        ),
    }


__all__ = [
    "LABEL_DEFINITION",
    "apply_single_bucket_protocol",
    "compute_single_bucket_membership_series",
    "encode_single_bucket_history",
    "encode_single_bucket_membership",
    "initialize_single_bucket_prototypes",
    "protocol_manifest",
]
