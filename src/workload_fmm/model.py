from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backoff import VariableOrderBackoff, state_sequences_from_memberships
from .fuzzy import apply_neighbor_modulation
from .graph import active_neighbor_indices
from .message_passing import LinkQualityMessagePassing
from .preprocessing import ResourcePreprocessor
from .shape_set import ShapeSetConfig, build_shape_set, compute_memberships
from .transition import SoftmaxTransitionKernel, TransitionTrainingConfig
from .types import ShapeSet
from .utils import default_mask, ensure_3d_resource_tensor, normalize_simplex


@dataclass
class ModelConfig:
    shape_set: ShapeSetConfig = field(default_factory=ShapeSetConfig)
    resource_names: tuple[str, ...] = ("cpu", "gpu", "mem", "read", "write")
    log_resource_names: tuple[str, ...] = ("read", "write")
    einstein_strength: float = 0.5
    max_order: int = 3
    backoff_gamma: float = 0.02
    backoff_smoothing: float = 1.0
    backoff_threshold: float = 2.0
    backoff_temperature: float = 1.0
    backoff_blend: float = 0.5
    transition_training: TransitionTrainingConfig = field(default_factory=TransitionTrainingConfig)
    message_passing_steps: int = 0
    message_hidden_dim: int = 8
    random_state: int = 7


class NonstationaryFuzzyMarkovModel:
    """End-to-end implementation of the documented methodology."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.preprocessor = ResourcePreprocessor(log_indices=self._log_indices())
        self.shape_set_: ShapeSet | None = None
        self.relation_matrix_: np.ndarray | None = None
        self.memberships_: np.ndarray | None = None
        self.states_: np.ndarray | None = None
        self.mask_: np.ndarray | None = None
        self.deployment_: np.ndarray | None = None
        self.transition_: SoftmaxTransitionKernel | None = None
        self.backoff_: VariableOrderBackoff | None = None
        self.message_passing_: LinkQualityMessagePassing | None = None

    def _log_indices(self) -> tuple[int, ...]:
        names = list(self.config.resource_names)
        return tuple(i for i, name in enumerate(names) if name in set(self.config.log_resource_names))

    def _resource_weights(self) -> np.ndarray:
        dim = len(self.config.resource_names)
        weights = self.config.shape_set.resource_weights
        if weights is None:
            return np.ones(dim, dtype=float)
        arr = np.asarray(weights, dtype=float)
        if arr.shape != (dim,):
            raise ValueError("shape_set.resource_weights length must match ModelConfig.resource_names")
        return arr

    def fit(
        self,
        resources: np.ndarray,
        mask: np.ndarray | None = None,
        deployment: np.ndarray | None = None,
        link_features: np.ndarray | None = None,
        edge_mask: np.ndarray | None = None,
        relation_matrix: np.ndarray | None = None,
    ) -> "NonstationaryFuzzyMarkovModel":
        raw = ensure_3d_resource_tensor(resources, len(self.config.resource_names))
        active = default_mask(raw, mask)
        x = self.preprocessor.fit_transform(raw, active)
        self.mask_ = active
        self.deployment_ = None if deployment is None else np.asarray(deployment, dtype=float)

        self.shape_set_ = build_shape_set(x, active, self.config.shape_set)
        base_mu = compute_memberships(
            x,
            self.shape_set_,
            active,
            weights=self._resource_weights(),
            dtw_window=self.config.shape_set.dtw_window,
            normalize=True,
        )
        mu = base_mu.copy()
        num_states = self.shape_set_.num_states
        if relation_matrix is None:
            self.relation_matrix_ = np.zeros((num_states + 1, num_states), dtype=float)
        else:
            theta = np.asarray(relation_matrix, dtype=float)
            if theta.shape != (num_states + 1, num_states):
                raise ValueError("relation_matrix must have shape (states + 1, states)")
            self.relation_matrix_ = theta

        if self.deployment_ is not None:
            for n in range(raw.shape[0]):
                for t in np.flatnonzero(active[n]):
                    neighbors, overlaps = active_neighbor_indices(self.deployment_, active, n, int(t))
                    if len(neighbors) == 0:
                        continue
                    mu[n, t] = apply_neighbor_modulation(
                        base_mu[n, t],
                        base_mu[neighbors, t],
                        overlaps,
                        self.relation_matrix_,
                        strength=self.config.einstein_strength,
                        normalize=True,
                    )

        if (
            self.config.message_passing_steps > 0
            and self.deployment_ is not None
            and link_features is not None
        ):
            self.message_passing_ = LinkQualityMessagePassing(
                steps=self.config.message_passing_steps,
                hidden_dim=self.config.message_hidden_dim,
                seed=self.config.random_state,
            ).fit_link_stats(link_features, edge_mask)
            for t in range(raw.shape[1]):
                active_t = active[:, t]
                if not np.any(active_t):
                    continue
                workload_repr = np.concatenate([x[:, t], mu[:, t]], axis=1)
                context = self.message_passing_.compute_context(
                    workload_repr,
                    self.deployment_,
                    link_features,
                    edge_mask,
                )
                modulated = self.message_passing_.residual_modulate_membership(mu[:, t], context)
                mu[active_t, t] = modulated[active_t]

        mu = normalize_simplex(mu, axis=-1)
        self.memberships_ = mu
        states = np.argmax(mu, axis=-1).astype(int)
        self.states_ = states
        sequences = state_sequences_from_memberships(mu, active)

        self.transition_ = SoftmaxTransitionKernel(num_states, seed=self.config.random_state).fit_counts(sequences)
        self.transition_.train_supervised(mu, states, active, self.config.transition_training)
        self.backoff_ = VariableOrderBackoff(
            max_order=self.config.max_order,
            gamma=self.config.backoff_gamma,
            smoothing=self.config.backoff_smoothing,
            thresholds=self.config.backoff_threshold,
            gate_temperature=self.config.backoff_temperature,
        ).fit(sequences, num_states=num_states)
        return self

    def transform_memberships(self, resources: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.shape_set_ is None:
            raise RuntimeError("fit must be called before transform_memberships")
        raw = ensure_3d_resource_tensor(resources, len(self.config.resource_names))
        active = default_mask(raw, mask)
        x = self.preprocessor.transform(raw)
        return compute_memberships(
            x,
            self.shape_set_,
            active,
            weights=self._resource_weights(),
            dtw_window=self.config.shape_set.dtw_window,
            normalize=True,
        )

    def predict_next(
        self,
        workload_index: int,
        time_index: int,
        return_parts: bool = False,
    ) -> dict[str, np.ndarray] | np.ndarray:
        if self.memberships_ is None or self.states_ is None or self.transition_ is None or self.backoff_ is None:
            raise RuntimeError("fit must be called before predict_next")
        n = int(workload_index)
        t = int(time_index)
        mu = self.memberships_[n, t]
        dynamic = self.transition_.predict(
            mu,
            time_index=t,
            time_period=self.config.transition_training.time_period,
        )
        context = self.states_[n, max(0, t - self.config.max_order + 1) : t + 1]
        backoff = self.backoff_.predict(context)
        blend = float(np.clip(self.config.backoff_blend, 0.0, 1.0))
        final = normalize_simplex((1.0 - blend) * dynamic + blend * backoff)
        if return_parts:
            return {"final": final, "dynamic": dynamic, "backoff": backoff, "membership": mu}
        return final

    @property
    def num_states(self) -> int:
        if self.shape_set_ is None:
            raise RuntimeError("fit must be called first")
        return self.shape_set_.num_states
