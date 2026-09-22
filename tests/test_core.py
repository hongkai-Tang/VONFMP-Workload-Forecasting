from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workload_fmm.backoff import VariableOrderBackoff
from workload_fmm.datasets.alibaba_v2018 import AlibabaPrepareConfig, prepare_container_usage
from workload_fmm.dtw import dtw_distance
from workload_fmm.fuzzy import centered_einstein_modulation
from workload_fmm.graph import deployment_from_pairs, hyperedge_intersections
from workload_fmm.message_passing import LinkQualityMessagePassing
from workload_fmm.model import ModelConfig, NonstationaryFuzzyMarkovModel
from workload_fmm.shape_set import ShapeSetConfig


class CoreTests(unittest.TestCase):
    def test_dtw_identity(self):
        x = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        self.assertEqual(dtw_distance(x, x), 0.0)

    def test_centered_einstein_bounds_and_neutral(self):
        mu = np.linspace(0.0, 1.0, 11)
        out = centered_einstein_modulation(mu, np.zeros_like(mu), strength=0.6)
        self.assertTrue(np.allclose(out, mu))
        high = centered_einstein_modulation(mu, np.ones_like(mu), strength=0.9)
        low = centered_einstein_modulation(mu, -np.ones_like(mu), strength=0.9)
        self.assertTrue(np.all((0.0 <= high) & (high <= 1.0)))
        self.assertTrue(np.all((0.0 <= low) & (low <= 1.0)))

    def test_hyperedge_intersections(self):
        h = deployment_from_pairs([(0, 0), (1, 0), (1, 1), (2, 2)], num_nodes=3, num_workloads=3)
        c = hyperedge_intersections(h)
        self.assertGreater(c[0, 1], 0.0)
        self.assertEqual(c[0, 2], 0.0)

    def test_backoff_distribution_is_simplex(self):
        model = VariableOrderBackoff(max_order=3, gamma=0.01).fit(
            [np.asarray([0, 1, 0, 2, 0, 1, 2, 2])],
            num_states=3,
        )
        pred = model.predict([0, 1, 2])
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.isclose(pred.sum(), 1.0))

    def test_link_quality_message_passing_shapes(self):
        deployment = deployment_from_pairs([(0, 0), (1, 1), (1, 2)], num_nodes=2, num_workloads=3)
        link_features = np.asarray(
            [
                [[10.0, 1.0, 0.01, 0.99], [8.0, 2.0, 0.02, 0.95]],
                [[7.0, 2.5, 0.03, 0.94], [9.0, 1.0, 0.01, 0.98]],
            ]
        )
        edge_mask = np.ones((2, 2), dtype=bool)
        workload_repr = np.ones((3, 4), dtype=float)
        mp = LinkQualityMessagePassing(steps=2, hidden_dim=5).fit_link_stats(link_features, edge_mask)
        context = mp.compute_context(workload_repr, deployment, link_features, edge_mask)
        self.assertEqual(context.shape, (3, 5))

    def test_end_to_end_model_predicts_simplex(self):
        rng = np.random.default_rng(3)
        data = rng.random((4, 20, 5))
        data[..., 3:] *= 20.0
        mask = np.ones((4, 20), dtype=bool)
        deployment = deployment_from_pairs(
            [(0, 0), (0, 1), (1, 2), (1, 3)],
            num_nodes=2,
            num_workloads=4,
        )
        cfg = ModelConfig(
            shape_set=ShapeSetConfig(
                min_length=2,
                max_length=5,
                max_shapes=3,
                coverage_threshold=0.7,
                max_candidates=40,
                dtw_window=2,
            ),
            max_order=2,
            backoff_blend=0.5,
        )
        model = NonstationaryFuzzyMarkovModel(cfg).fit(data, mask=mask, deployment=deployment)
        pred = model.predict_next(0, 10)
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.isclose(pred.sum(), 1.0))

    def test_end_to_end_model_accepts_four_resource_alibaba_shape(self):
        rng = np.random.default_rng(11)
        data = rng.random((5, 24, 4))
        data[..., 2] *= 100.0
        data[..., 3] *= 1.0
        mask = np.ones((5, 24), dtype=bool)
        deployment = deployment_from_pairs(
            [(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)],
            num_nodes=3,
            num_workloads=5,
        )
        cfg = ModelConfig(
            resource_names=("cpu", "mem", "disk_i", "disk_o"),
            log_resource_names=(),
            shape_set=ShapeSetConfig(
                min_length=2,
                max_length=5,
                max_shapes=4,
                coverage_threshold=0.7,
                max_candidates=50,
                dtw_window=2,
                    resource_weights=(1.0, 1.0, 1.0, 1.0),
            ),
            max_order=2,
        )
        model = NonstationaryFuzzyMarkovModel(cfg).fit(data, mask=mask, deployment=deployment)
        pred = model.predict_next(0, 12)
        self.assertTrue(np.all(pred >= 0.0))
        self.assertTrue(np.isclose(pred.sum(), 1.0))

    def test_alibaba_v2018_prepare_writes_relative_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            processed = root / "processed"
            raw.mkdir()
            usage = raw / "container_usage.csv"
            with usage.open("w", encoding="utf-8", newline="") as f:
                f.write(
                    "c1,m1,0,10,20,0,100,0,0,0,30\n"
                    "c1,m1,10,20,30,0,120,0,0,0,40\n"
                    "c1,m1,20,30,40,0,130,0,0,0,101\n"
                    "c2,m2,0,50,60,0,140,0,0,0,70\n"
                    "c2,m2,10,60,70,0,150,0,0,0,80\n"
                )
            manifest = prepare_container_usage(
                AlibabaPrepareConfig(
                    raw_dir=raw,
                    processed_dir=processed,
                    max_workloads=10,
                    min_points=1,
                    resample_seconds=10,
                )
            )
            self.assertEqual(manifest["invalid_rows"], 1)
            self.assertTrue((processed / "resources.csv").exists())
            self.assertTrue((processed / "deployment.csv").exists())
            rows = (processed / "resources.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "workload_id,time,cpu,mem,disk_i,disk_o")

    def test_alibaba_v2018_prepare_requires_full_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            processed = root / "processed"
            raw.mkdir()
            usage = raw / "container_usage.csv"
            with usage.open("w", encoding="utf-8", newline="") as f:
                f.write(
                    "c1,m1,0,10,20,0,100,0,0,0,30\n"
                    "c1,m1,10,20,30,0,120,0,0,0,40\n"
                    "c1,m1,20,30,40,0,130,0,0,0,50\n"
                    "c2,m2,0,50,60,0,140,0,0,0,70\n"
                    "c2,m2,10,60,70,0,150,0,0,0,80\n"
                )
            manifest = prepare_container_usage(
                AlibabaPrepareConfig(
                    raw_dir=raw,
                    processed_dir=processed,
                    max_workloads=10,
                    min_points=1,
                    resample_seconds=10,
                    require_full_span=True,
                    full_span_days=30 / 86400,
                    full_span_start_bucket=0,
                    full_span_start_seconds=None,
                )
            )
            self.assertEqual(manifest["full_span_expected_points"], 3)
            self.assertEqual(manifest["kept_workloads"], 1)
            rows = (processed / "resources.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row.startswith("c1,") for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
