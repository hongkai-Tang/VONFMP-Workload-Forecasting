from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alibaba_ours_exp.metrics import atomic_write_parquet
from alibaba_ours_exp.model import AlibabaOursModel, OursModelConfig
from alibaba_ours_minganxing.design import build_conditions
from alibaba_ours_minganxing.enrichment import enrich_run
from alibaba_ours_minganxing.protocol import apply_single_bucket_protocol


class DesignTests(unittest.TestCase):
    def test_exact_twenty_unique_conditions(self) -> None:
        conditions = build_conditions(ROOT / "configs" / "sensitivity.json")
        self.assertEqual(len(conditions), 20)
        self.assertEqual(len({condition.signature for condition in conditions}), 20)
        self.assertEqual(
            {condition.history_length for condition in conditions if condition.study == "tau"},
            {12, 24, 48},
        )
        baseline = conditions[0]
        self.assertEqual(baseline.history_length, 1440)
        self.assertTrue(baseline.is_shared_baseline)

    def test_seed17_config_preserves_the_formal_grid(self) -> None:
        conditions = build_conditions(ROOT / "configs" / "sensitivity_seed17.json")
        self.assertEqual(len(conditions), 20)
        self.assertEqual({condition.seed for condition in conditions}, {17})
        self.assertEqual(len({condition.signature for condition in conditions}), 20)

    def test_label_depends_only_on_last_bucket(self) -> None:
        apply_single_bucket_protocol()
        model = AlibabaOursModel(
            OursModelConfig(
                resource_dim=4,
                num_states=4,
                history_window=10,
                prototype_length=3,
                max_order=3,
            )
        )
        left = torch.randn(2, 10, 4)
        right = torch.randn(2, 10, 4)
        right[:, -1] = left[:, -1]
        left_mu = model.encode_membership(left, torch.ones(2, 10, dtype=torch.bool))
        right_mu = model.encode_membership(right, torch.ones(2, 10, dtype=torch.bool))
        torch.testing.assert_close(left_mu, right_mu)


class EnrichmentTests(unittest.TestCase):
    def test_enrichment_writes_all_metric_levels(self) -> None:
        apply_single_bucket_protocol()
        condition = build_conditions(ROOT / "configs" / "sensitivity.json")[0]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            run_dir = Path(temporary)
            (run_dir / "models" / "L=1440").mkdir(parents=True)
            (run_dir / "models" / "shared_preprocessing.npz").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "data").mkdir(parents=True)
            prediction_dir = run_dir / "predictions" / "L=1440" / "split=validation"
            prediction_dir.mkdir(parents=True)

            rng = np.random.default_rng(7)
            values = rng.uniform(0.0, 1.0, size=(2, 1442, 4)).astype(np.float32)
            masks = np.ones((2, 1442), dtype=np.uint8)
            times = np.arange(1442, dtype=np.int64) * 60
            np.savez_compressed(
                run_dir / "data" / "dataset.npz",
                workload_ids=np.asarray(["w0", "w1"]),
                resource_names=np.asarray(["cpu", "mem", "disk_i", "disk_o"]),
                input_values=values,
                observed_values=values,
                input_valid_mask=masks,
                target_observed_mask=masks,
                imputed_mask=np.zeros_like(masks),
            )
            # Cache time is required by post-evaluation alignment.
            with np.load(run_dir / "data" / "dataset.npz", allow_pickle=False) as source:
                payload = {name: source[name] for name in source.files}
            payload["time_seconds"] = times
            np.savez_compressed(run_dir / "data" / "dataset.npz", **payload)
            np.savez_compressed(
                run_dir / "models" / "shared_preprocessing.npz",
                mean=np.zeros(4, dtype=np.float32),
                std=np.ones(4, dtype=np.float32),
            )

            model = AlibabaOursModel(
                OursModelConfig(
                    resource_dim=4,
                    num_states=8,
                    history_window=1440,
                    prototype_length=30,
                    max_order=3,
                    time_period=1440,
                )
            )
            model.save(run_dir / "models" / "L=1440" / "model.pt")
            (run_dir / "models" / "L=1440" / "complete.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_identity.json").write_text(
                json.dumps({"identity_hash": "synthetic"}), encoding="utf-8"
            )
            truth = model.encode_membership(torch.from_numpy(values[:, 1441:1442])).detach().numpy()
            predicted = model.encode_membership(torch.from_numpy(values[:, 1440:1441])).detach().numpy()
            rows = []
            for workload, (truth_row, predicted_row) in enumerate(zip(truth, predicted)):
                row = {
                    "split": "validation",
                    "history_length": 1440,
                    "h": 1,
                    "origin_id": "validation-synthetic",
                    "origin_time": int(times[1440]),
                    "target_time": int(times[1441]),
                    "workload_id": f"w{workload}",
                    "true_class": int(np.argmax(truth_row)),
                    "pred_class": int(np.argmax(predicted_row)),
                    "confidence": float(np.max(predicted_row)),
                    "true_observed": True,
                }
                for state in range(8):
                    row[f"true_mu_{state}"] = float(truth_row[state])
                    row[f"pred_mu_{state}"] = float(predicted_row[state])
                rows.append(row)
            fragment = prediction_dir / "origin=validation-synthetic.parquet"
            atomic_write_parquet(fragment, pd.DataFrame(rows))
            (prediction_dir / "complete-validation-synthetic.json").write_text("{}", encoding="utf-8")

            manifest = enrich_run(
                SimpleNamespace(runtime={"progress_interval_seconds": 30}),
                run_dir,
                condition,
                resume=False,
            )
            self.assertEqual(manifest["per_sample_rows"], 2)
            for name in (
                "enriched_predictions.parquet",
                "baseline_metrics_by_origin.csv",
                "baseline_metrics_by_condition.csv",
                "transition_metrics.csv",
                "error_quantiles.csv",
                "enrichment.complete.json",
            ):
                self.assertTrue((run_dir / "metrics" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
