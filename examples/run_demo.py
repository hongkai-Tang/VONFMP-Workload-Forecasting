from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workload_fmm import ModelConfig, NonstationaryFuzzyMarkovModel
from workload_fmm.shape_set import ShapeSetConfig
from workload_fmm.transition import TransitionTrainingConfig


def make_synthetic_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    workloads = 6
    steps = 48
    resources = np.zeros((workloads, steps, 5), dtype=float)
    mask = np.ones((workloads, steps), dtype=bool)
    base_patterns = np.asarray(
        [
            [0.25, 0.10, 0.35, 20.0, 15.0],
            [0.75, 0.20, 0.55, 35.0, 20.0],
            [0.40, 0.80, 0.65, 30.0, 40.0],
        ],
        dtype=float,
    )
    for n in range(workloads):
        state = n % len(base_patterns)
        for t in range(steps):
            if t % 14 == 0 and t > 0:
                state = (state + 1 + (n % 2)) % len(base_patterns)
            seasonal = 0.05 * np.sin(2 * np.pi * t / 24.0)
            noise = rng.normal(0.0, [0.04, 0.04, 0.03, 3.0, 3.0])
            resources[n, t] = np.maximum(base_patterns[state] + seasonal + noise, 0.0)

    # H has shape (nodes, workloads). Workloads sharing a node are neighbors.
    deployment = np.zeros((4, workloads), dtype=float)
    for n in range(workloads):
        deployment[n % 4, n] = 1.0
        if n % 3 == 0:
            deployment[(n + 1) % 4, n] = 1.0
    return resources, mask, deployment


def main() -> None:
    resources, mask, deployment = make_synthetic_data()
    cfg = ModelConfig(
        shape_set=ShapeSetConfig(
            min_length=3,
            max_length=8,
            split_penalty=0.5,
            coverage_threshold=0.60,
            max_shapes=8,
            dtw_window=3,
            max_candidates=120,
        ),
        max_order=3,
        backoff_blend=0.5,
        transition_training=TransitionTrainingConfig(epochs=30, learning_rate=0.03),
    )
    model = NonstationaryFuzzyMarkovModel(cfg).fit(resources, mask=mask, deployment=deployment)
    pred = model.predict_next(workload_index=0, time_index=30, return_parts=True)
    print("states:", model.num_states)
    print("shape lengths:", model.shape_set_.lengths())
    print("shape coverage:", round(model.shape_set_.coverage, 4))
    print("membership:", np.round(pred["membership"], 4).tolist())
    print("dynamic:", np.round(pred["dynamic"], 4).tolist())
    print("backoff:", np.round(pred["backoff"], 4).tolist())
    print("final:", np.round(pred["final"], 4).tolist())
    print("final_sum:", float(np.sum(pred["final"])))


if __name__ == "__main__":
    main()
