from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import normalize_deployment
from .utils import sigmoid, softmax


@dataclass
class LinkQualityStats:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8


class LinkQualityMessagePassing:
    """Link-quality-aware r-hop workload-node-link-node-workload propagation."""

    def __init__(
        self,
        steps: int = 2,
        hidden_dim: int = 8,
        seed: int = 7,
        eps: float = 1e-8,
    ) -> None:
        self.steps = int(steps)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.eps = float(eps)
        self.stats_: LinkQualityStats | None = None
        self.gate_weight_: np.ndarray | None = None
        self.gate_bias_: float = 0.0
        self.w_h_: np.ndarray | None = None
        self.w_self_: list[np.ndarray] = []
        self.w_msg_: list[np.ndarray] = []
        self.w_residual_: np.ndarray | None = None
        self.b_residual_: np.ndarray | None = None

    def fit_link_stats(self, link_features: np.ndarray, edge_mask: np.ndarray | None = None) -> "LinkQualityMessagePassing":
        p = np.asarray(link_features, dtype=float)
        if p.ndim != 3 or p.shape[-1] != 4:
            raise ValueError("link_features must have shape (nodes, nodes, 4)")
        mask = np.isfinite(p).all(axis=-1) if edge_mask is None else np.asarray(edge_mask, dtype=bool)
        flat = p[mask]
        if flat.size == 0:
            raise ValueError("no link features available for normalization")
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.stats_ = LinkQualityStats(mean=mean, std=std, eps=self.eps)
        if self.gate_weight_ is None:
            self.gate_weight_ = np.ones(4, dtype=float)
        return self

    def _phi(self, link_features: np.ndarray) -> np.ndarray:
        if self.stats_ is None:
            raise RuntimeError("fit_link_stats must be called before propagation")
        z = (np.asarray(link_features, dtype=float) - self.stats_.mean) / (self.stats_.std + self.eps)
        # bandwidth and availability are positive; delay and loss are negative.
        return np.stack([z[..., 0], -z[..., 1], -z[..., 2], z[..., 3]], axis=-1)

    def link_gates(self, link_features: np.ndarray, edge_mask: np.ndarray | None = None) -> np.ndarray:
        phi = self._phi(link_features)
        if self.gate_weight_ is None:
            self.gate_weight_ = np.ones(4, dtype=float)
        raw = phi @ self.gate_weight_ + self.gate_bias_
        gates = sigmoid(raw)
        if edge_mask is not None:
            gates = np.where(np.asarray(edge_mask, dtype=bool), gates, 0.0)
        gates = np.where(np.isfinite(gates), gates, 0.0)
        incoming = gates.sum(axis=0, keepdims=True)
        return np.divide(gates, np.maximum(incoming, self.eps), out=np.zeros_like(gates), where=incoming > self.eps)

    def _init_projection(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        if self.w_h_ is None or self.w_h_.shape != (input_dim, self.hidden_dim):
            self.w_h_ = rng.normal(0.0, 1.0 / max(input_dim, 1), size=(input_dim, self.hidden_dim))
        if not self.w_self_:
            self.w_self_ = []
            self.w_msg_ = []
            for _ in range(self.steps):
                self.w_self_.append(np.eye(self.hidden_dim))
                self.w_msg_.append(rng.normal(0.0, 1.0 / max(self.hidden_dim, 1), size=(self.hidden_dim, self.hidden_dim)))

    def _init_residual(self, output_states: int) -> None:
        residual_dim = output_states + self.hidden_dim
        if self.w_residual_ is None or self.w_residual_.shape != (residual_dim, output_states):
            # Neutral by default. Users can replace it with trained parameters.
            self.w_residual_ = np.zeros((residual_dim, output_states), dtype=float)
            self.b_residual_ = np.zeros(output_states, dtype=float)

    def compute_context(
        self,
        workload_repr: np.ndarray,
        deployment: np.ndarray,
        link_features: np.ndarray,
        edge_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        h_work = np.asarray(workload_repr, dtype=float)
        if h_work.ndim != 2:
            raise ValueError("workload_repr must have shape (workloads, features)")
        hbar = normalize_deployment(deployment)
        if hbar.shape[1] != h_work.shape[0]:
            raise ValueError("deployment columns must match number of workloads")
        self._init_projection(h_work.shape[1])
        assert self.w_h_ is not None
        qbar = self.link_gates(link_features, edge_mask)
        node = hbar @ h_work @ self.w_h_
        for step in range(self.steps):
            self_part = node @ self.w_self_[step]
            msg_part = qbar.T @ (node @ self.w_msg_[step])
            node = np.tanh(self_part + msg_part)
        return hbar.T @ node

    def residual_modulate_membership(self, membership: np.ndarray, context: np.ndarray) -> np.ndarray:
        mu = np.asarray(membership, dtype=float)
        ctx = np.asarray(context, dtype=float)
        if mu.ndim != 2 or ctx.ndim != 2 or mu.shape[0] != ctx.shape[0]:
            raise ValueError("membership and context must have matching workload rows")
        if ctx.shape[1] != self.hidden_dim:
            raise ValueError("context feature dimension must match hidden_dim")
        self._init_residual(output_states=mu.shape[1])
        assert self.w_residual_ is not None and self.b_residual_ is not None
        features = np.concatenate([mu, ctx], axis=1)
        gate = np.tanh(features @ self.w_residual_ + self.b_residual_)
        return softmax(np.log(np.maximum(mu, self.eps)) + gate, axis=1)
