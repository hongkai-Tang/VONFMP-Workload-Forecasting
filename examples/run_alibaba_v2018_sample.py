from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workload_fmm.datasets.alibaba_v2018 import alibaba_v2018_model_config, load_prepared
from workload_fmm.model import NonstationaryFuzzyMarkovModel


def main() -> None:
    resources, mask, deployment, workload_ids, time_ids, _ = load_prepared()
    cfg = alibaba_v2018_model_config()
    cfg.shape_set.max_shapes = 8
    cfg.shape_set.max_candidates = 300
    cfg.shape_set.min_length = 5
    cfg.shape_set.max_length = 30
    cfg.transition_training.epochs = 20
    model = NonstationaryFuzzyMarkovModel(cfg).fit(resources, mask=mask, deployment=deployment)
    workload_index = 0
    active = np.flatnonzero(mask[workload_index])
    time_index = int(active[min(len(active) - 2, max(0, len(active) // 2))])
    pred = model.predict_next(workload_index, time_index, return_parts=True)
    print("workload_id:", workload_ids[workload_index])
    print("time:", time_ids[time_index])
    print("resource_names:", cfg.resource_names)
    print("states:", model.num_states)
    print("membership:", np.round(pred["membership"], 6).tolist())
    print("next_distribution:", np.round(pred["final"], 6).tolist())
    print("sum:", float(np.sum(pred["final"])))


if __name__ == "__main__":
    main()
