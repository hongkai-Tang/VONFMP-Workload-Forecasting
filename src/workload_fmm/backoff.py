from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .utils import normalize_simplex, sigmoid


@dataclass
class BackoffPrediction:
    distribution: np.ndarray
    order_distributions: list[np.ndarray]
    gates: np.ndarray
    weights: np.ndarray


class VariableOrderBackoff:
    """Decay-weighted variable-order Markov BackOff estimator."""

    def __init__(
        self,
        max_order: int = 3,
        gamma: float = 0.02,
        smoothing: float | list[float] = 1.0,
        thresholds: float | list[float] = 2.0,
        gate_temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        self.max_order = int(max_order)
        self.gamma = float(gamma)
        self.smoothing = smoothing
        self.thresholds = thresholds
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)
        self.num_states_: int | None = None
        self.events_: list[dict[tuple[int, ...], dict[int, list[int]]]] = [
            defaultdict(lambda: defaultdict(list)) for _ in range(self.max_order + 1)
        ]
        self.max_index_: int = 0

    def _lambda(self, order: int) -> float:
        if isinstance(self.smoothing, list):
            return float(self.smoothing[min(order - 1, len(self.smoothing) - 1)])
        return float(self.smoothing)

    def _threshold(self, order: int) -> float:
        if isinstance(self.thresholds, list):
            return float(self.thresholds[min(order - 1, len(self.thresholds) - 1)])
        return float(self.thresholds)

    def fit(self, sequences: list[np.ndarray], num_states: int | None = None) -> "VariableOrderBackoff":
        if not sequences:
            raise ValueError("sequences must not be empty")
        if num_states is None:
            num_states = int(max(int(np.max(seq)) for seq in sequences if len(seq) > 0) + 1)
        self.num_states_ = int(num_states)
        self.events_ = [defaultdict(lambda: defaultdict(list)) for _ in range(self.max_order + 1)]
        global_index = 0
        for seq in sequences:
            y = np.asarray(seq, dtype=int)
            for pos in range(len(y)):
                target = int(y[pos])
                self.events_[0][()][target].append(global_index)
                for order in range(1, self.max_order + 1):
                    if pos - order < 0:
                        continue
                    context = tuple(int(v) for v in y[pos - order : pos])
                    self.events_[order][context][target].append(global_index)
                global_index += 1
        self.max_index_ = global_index
        return self

    def _counts(self, order: int, context: tuple[int, ...], current_index: int) -> tuple[np.ndarray, float]:
        if self.num_states_ is None:
            raise RuntimeError("fit must be called before predict")
        counts = np.zeros(self.num_states_, dtype=float)
        state_events = self.events_[order].get(context, {})
        for state, positions in state_events.items():
            age = current_index - np.asarray(positions, dtype=float)
            counts[int(state)] = float(np.exp(-self.gamma * np.maximum(age, 0.0)).sum())
        return counts, float(counts.sum())

    def predict(
        self,
        context: list[int] | np.ndarray,
        current_index: int | None = None,
        return_details: bool = False,
    ) -> np.ndarray | BackoffPrediction:
        if self.num_states_ is None:
            raise RuntimeError("fit must be called before predict")
        current = self.max_index_ if current_index is None else int(current_index)
        ctx = tuple(int(v) for v in context)
        max_available = min(self.max_order, len(ctx))

        order_distributions: list[np.ndarray] = []
        counts0, total0 = self._counts(0, (), current)
        p0 = normalize_simplex(counts0 + self.eps)
        order_distributions.append(p0)
        totals = np.zeros(self.max_order + 1, dtype=float)
        totals[0] = total0

        for order in range(1, self.max_order + 1):
            if order > max_available:
                order_distributions.append(order_distributions[-1])
                continue
            sub_ctx = ctx[-order:]
            counts, total = self._counts(order, sub_ctx, current)
            lam = max(self._lambda(order), self.eps)
            lower = order_distributions[order - 1]
            dist = (counts + lam * lower) / max(total + lam, self.eps)
            order_distributions.append(normalize_simplex(dist))
            totals[order] = total

        gates = np.zeros(self.max_order + 1, dtype=float)
        for order in range(1, self.max_order + 1):
            if order <= max_available:
                gates[order] = sigmoid((totals[order] - self._threshold(order)) / max(self.gate_temperature, self.eps))

        weights = np.zeros(self.max_order + 1, dtype=float)
        weights[self.max_order] = gates[self.max_order]
        for order in range(1, self.max_order):
            higher_fail = float(np.prod(1.0 - gates[order + 1 : self.max_order + 1]))
            weights[order] = gates[order] * higher_fail
        weights[0] = float(np.prod(1.0 - gates[1 : self.max_order + 1]))
        weights = normalize_simplex(weights)

        mixed = np.zeros(self.num_states_, dtype=float)
        for order, weight in enumerate(weights):
            mixed += weight * order_distributions[order]
        mixed = normalize_simplex(mixed)
        if return_details:
            return BackoffPrediction(mixed, order_distributions, gates, weights)
        return mixed


def state_sequences_from_memberships(memberships: np.ndarray, mask: np.ndarray | None = None) -> list[np.ndarray]:
    mu = np.asarray(memberships, dtype=float)
    if mu.ndim != 3:
        raise ValueError("memberships must have shape (workloads, time, states)")
    active = np.ones(mu.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    sequences: list[np.ndarray] = []
    for n in range(mu.shape[0]):
        run: list[int] = []
        for t, is_active in enumerate(active[n]):
            if is_active:
                run.append(t)
            elif run:
                sequences.append(np.argmax(mu[n, run], axis=1).astype(int))
                run = []
        if run:
            sequences.append(np.argmax(mu[n, run], axis=1).astype(int))
    return sequences
