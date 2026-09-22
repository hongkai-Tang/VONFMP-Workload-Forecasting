from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import normalize_simplex, softmax, time_encoding


@dataclass
class TransitionTrainingConfig:
    epochs: int = 0
    learning_rate: float = 0.05
    l2: float = 1e-4
    time_period: float = 24.0


class SoftmaxTransitionKernel:
    """Dynamic row-stochastic transition kernel with a NumPy linear scorer."""

    def __init__(self, num_states: int, seed: int = 7, eps: float = 1e-12) -> None:
        self.num_states = int(num_states)
        self.seed = int(seed)
        self.eps = float(eps)
        self.counts_ = np.zeros((self.num_states, self.num_states), dtype=float)
        self.weight_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit_counts(self, sequences: list[np.ndarray]) -> "SoftmaxTransitionKernel":
        self.counts_[:] = 0.0
        for seq in sequences:
            y = np.asarray(seq, dtype=int)
            for prev, nxt in zip(y[:-1], y[1:]):
                self.counts_[int(prev), int(nxt)] += 1.0
        return self

    def empirical_matrix(self, smoothing: float = 1.0) -> np.ndarray:
        counts = self.counts_ + float(smoothing)
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), self.eps)

    def _features(self, membership: np.ndarray, current_state: int, time_index: int, time_period: float) -> np.ndarray:
        ratios = self.empirical_matrix(smoothing=1.0)[int(current_state)]
        t_feat = time_encoding(time_index, period=time_period)
        return np.concatenate([np.asarray(membership, dtype=float), np.asarray(t_feat).reshape(-1), ratios])

    def _ensure_params(self, feature_dim: int) -> None:
        if self.weight_ is None or self.weight_.shape != (feature_dim, self.num_states):
            rng = np.random.default_rng(self.seed)
            self.weight_ = rng.normal(0.0, 0.01, size=(feature_dim, self.num_states))
            self.bias_ = np.zeros(self.num_states, dtype=float)

    def transition_matrix(
        self,
        membership: np.ndarray,
        time_index: int = 0,
        time_period: float = 24.0,
    ) -> np.ndarray:
        mu = normalize_simplex(np.asarray(membership, dtype=float))
        empirical = self.empirical_matrix(smoothing=1.0)
        rows = []
        for state in range(self.num_states):
            feat = self._features(mu, state, time_index, time_period)
            if self.weight_ is None or self.bias_ is None:
                scores = np.log(np.maximum(empirical[state], self.eps))
            else:
                scores = feat @ self.weight_ + self.bias_ + np.log(np.maximum(empirical[state], self.eps))
            rows.append(softmax(scores))
        return np.vstack(rows)

    def predict(
        self,
        membership: np.ndarray,
        time_index: int = 0,
        time_period: float = 24.0,
    ) -> np.ndarray:
        pi = normalize_simplex(membership)
        return normalize_simplex(pi @ self.transition_matrix(pi, time_index, time_period))

    def train_supervised(
        self,
        memberships: np.ndarray,
        states: np.ndarray,
        mask: np.ndarray,
        config: TransitionTrainingConfig | None = None,
    ) -> "SoftmaxTransitionKernel":
        cfg = config or TransitionTrainingConfig()
        if cfg.epochs <= 0:
            return self
        mu = np.asarray(memberships, dtype=float)
        y = np.asarray(states, dtype=int)
        active = np.asarray(mask, dtype=bool)
        examples: list[tuple[np.ndarray, int, int, int]] = []
        for n in range(mu.shape[0]):
            idx = np.flatnonzero(active[n])
            for pos in idx[:-1]:
                if not active[n, pos + 1]:
                    continue
                cur = int(y[n, pos])
                target = int(y[n, pos + 1])
                examples.append((mu[n, pos], cur, int(pos), target))
        if not examples:
            return self
        feat0 = self._features(examples[0][0], examples[0][1], examples[0][2], cfg.time_period)
        self._ensure_params(len(feat0))
        assert self.weight_ is not None and self.bias_ is not None
        for _ in range(cfg.epochs):
            grad_w = np.zeros_like(self.weight_)
            grad_b = np.zeros_like(self.bias_)
            for membership, cur, time_idx, target in examples:
                feat = self._features(membership, cur, time_idx, cfg.time_period)
                empirical_log = np.log(np.maximum(self.empirical_matrix(smoothing=1.0)[cur], self.eps))
                probs = softmax(feat @ self.weight_ + self.bias_ + empirical_log)
                delta = probs
                delta[target] -= 1.0
                grad_w += np.outer(feat, delta)
                grad_b += delta
            scale = 1.0 / len(examples)
            grad_w = grad_w * scale + cfg.l2 * self.weight_
            grad_b = grad_b * scale
            self.weight_ -= cfg.learning_rate * grad_w
            self.bias_ -= cfg.learning_rate * grad_b
        return self
