from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from alibaba_ours_exp.checkpoint import (
    CheckpointMismatchError,
    load_recursive_checkpoint,
    save_recursive_checkpoint,
)
from alibaba_ours_exp.config import ResourceRule, SourceConfig, load_experiment_config
from alibaba_ours_exp.metrics import atomic_write_json, evaluate_predictions
from alibaba_ours_exp.model import AlibabaOursModel, OursModelConfig, PhysicalTimeBackoff
from alibaba_ours_exp.progress import ProgressTracker, format_console_progress, read_status
from alibaba_ours_exp.recursive_forecast import recursive_forecast
from alibaba_ours_exp.raw_prepare import (
    export_selected_raw_rows,
    extract_selected_resumable,
    scan_stats_resumable,
)
from alibaba_ours_exp.training import TrainingConfig, compute_membership_series, train_model


def synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    workloads, steps, resources_count = 4, 24, 4
    time = np.arange(steps, dtype=np.float32)
    values = np.empty((workloads, steps, resources_count), dtype=np.float32)
    for workload in range(workloads):
        for resource in range(resources_count):
            values[workload, :, resource] = np.clip(
                0.4
                + 0.25 * np.sin((time + workload) / (2.0 + resource))
                + 0.01 * rng.normal(size=steps),
                0.0,
                1.0,
            )
    mask = np.ones((workloads, steps), dtype=bool)
    deployment = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    links = np.asarray(
        [
            [[10.0, 1.0, 0.01, 0.99], [8.0, 2.0, 0.02, 0.96]],
            [[8.0, 2.0, 0.02, 0.96], [11.0, 1.0, 0.01, 0.99]],
        ],
        dtype=np.float32,
    )
    return values, mask, time, deployment, links


class ModelProtocolTests(unittest.TestCase):
    def test_decode_resources_aligns_input_with_decoder(self) -> None:
        model = AlibabaOursModel(
            OursModelConfig(
                resource_dim=4,
                num_states=8,
                history_window=5,
                prototype_length=5,
                max_order=3,
                random_state=7,
            )
        )
        memberships = torch.softmax(torch.randn(3, 8, dtype=torch.float64), dim=-1)
        decoded = model.shape_encoder.decode_resources(memberships)
        self.assertEqual(decoded.device, model.shape_encoder.prototypes.device)
        self.assertEqual(decoded.dtype, model.shape_encoder.prototypes.dtype)
        self.assertEqual(tuple(decoded.shape), (3, 4))

    def test_batched_membership_series_matches_stepwise_encoding(self) -> None:
        values, mask, _, _, _ = synthetic_case()
        mask[0, 3] = False
        mask[2, 9:11] = False
        model = AlibabaOursModel(
            OursModelConfig(
                resource_dim=4,
                num_states=8,
                history_window=5,
                prototype_length=5,
                max_order=3,
                random_state=7,
            )
        )
        expected = []
        for h in range(values.shape[1]):
            start = max(0, h - model.config.history_window + 1)
            expected.append(
                model.encode_membership(
                    torch.as_tensor(values[:, start : h + 1]),
                    torch.as_tensor(mask[:, start : h + 1]),
                )
            )
        expected_tensor = torch.stack(expected, dim=1)
        actual = compute_membership_series(
            model,
            torch.as_tensor(values),
            torch.as_tensor(mask),
        )
        self.assertTrue(torch.allclose(actual, expected_tensor, atol=1e-7, rtol=1e-7))

    def test_indexed_backoff_matches_direct_event_scan(self) -> None:
        rng = np.random.default_rng(17)
        memberships = rng.random((6, 80, 8))
        memberships /= memberships.sum(axis=-1, keepdims=True)
        mask = rng.random((6, 80)) > 0.08
        times = np.arange(80, dtype=np.float64)
        backoff = PhysicalTimeBackoff(
            num_states=8,
            max_order=3,
            decay=0.02,
            time_unit=1.0,
            smoothing=1.0,
            threshold=2.0,
            temperature=1.0,
            max_age=15,
        ).fit(memberships, times, mask)

        def direct(order: int, context: tuple[int, ...], forecast_time: float) -> tuple[np.ndarray, float]:
            counts = np.zeros(8, dtype=np.float64)
            for event_time, target in backoff._active_events(order, context):
                age = forecast_time - event_time
                if age <= 0.0 or age > 15.0:
                    continue
                counts += np.exp(-0.02 * age) * target
            support = float(counts.sum())
            smoothed = counts + 1.0
            return smoothed / smoothed.sum(), support

        for order, order_events in enumerate(backoff.events):
            for context in list(order_events)[:8]:
                for forecast_time in (17.5, 42.0, 80.0):
                    expected_distribution, expected_support = direct(
                        order, context, forecast_time
                    )
                    actual_distribution, actual_support = backoff._distribution(
                        order, context, forecast_time
                    )
                    np.testing.assert_allclose(
                        actual_distribution, expected_distribution, rtol=1e-11, atol=1e-11
                    )
                    self.assertAlmostEqual(actual_support, expected_support, places=9)

    def test_streaming_backoff_matches_reindexed_queries_and_resume(self) -> None:
        rng = np.random.default_rng(23)
        memberships = rng.random((6, 20, 8))
        memberships /= memberships.sum(axis=-1, keepdims=True)
        times = np.arange(20, dtype=np.float64)
        baseline = PhysicalTimeBackoff(8, 3, 0.02, 1.0, 1.0, 2.0, 1.0, max_age=5).fit(
            memberships, times
        )
        streaming = baseline.clone()
        streaming.enable_streaming()
        contexts = np.argmax(memberships[:, -3:], axis=-1)
        for forecast_time in range(20, 36):
            forecast_times = torch.full((6,), float(forecast_time))
            expected, expected_diagnostics = baseline.predict_batch(
                torch.as_tensor(contexts), forecast_times, device=torch.device("cpu"), dtype=torch.float32
            )
            actual, actual_diagnostics = streaming.predict_batch(
                torch.as_tensor(contexts), forecast_times, device=torch.device("cpu"), dtype=torch.float32
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
            self.assertTrue(
                torch.allclose(
                    actual_diagnostics["supports"],
                    expected_diagnostics["supports"],
                    atol=1e-5,
                    rtol=1e-5,
                )
            )
            targets = rng.random((6, 8))
            targets /= targets.sum(axis=-1, keepdims=True)
            baseline.append_batch(contexts, targets, np.full(6, forecast_time))
            streaming.append_batch(contexts, targets, np.full(6, forecast_time))
            contexts = np.column_stack([contexts[:, 1:], np.argmax(targets, axis=-1)])

        restored = PhysicalTimeBackoff(8, 3, 0.02, 1.0, 1.0, 2.0, 1.0, max_age=5)
        restored.import_state(streaming.export_state())
        restored.enable_streaming()
        expected, _ = baseline.predict_batch(
            torch.as_tensor(contexts),
            torch.full((6,), 36.0),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        actual, _ = restored.predict_batch(
            torch.as_tensor(contexts),
            torch.full((6,), 36.0),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def _trained_model(self) -> AlibabaOursModel:
        values, mask, time, deployment, links = synthetic_case()
        model = AlibabaOursModel(
            OursModelConfig(
                resource_dim=4,
                num_states=8,
                history_window=5,
                prototype_length=5,
                max_order=3,
                backoff_decay=0.02,
                backoff_max_age=5,
                message_passing_steps=2,
                message_hidden_dim=8,
                random_state=7,
            )
        )
        initial_theta = model.graph_block.theta.detach().clone()
        result = train_model(
            model,
            values[:, :18],
            time[:18],
            mask[:, :18],
            deployment,
            links,
            config=TrainingConfig(
                epochs=1,
                max_origins_per_epoch=2,
                prototype_candidates=12,
                early_stopping_patience=1,
                random_state=7,
            ),
        )
        self.assertFalse(torch.allclose(initial_theta, result.model.graph_block.theta))
        return result.model

    def test_recursive_forecast_is_synchronous_and_simplex(self) -> None:
        values, mask, time, deployment, links = synthetic_case()
        model = self._trained_model()
        result = recursive_forecast(
            model,
            values[:, 13:18],
            time[13:18],
            [1, 3, 6],
            mask[:, 13:18],
            deployment,
            links,
        )
        self.assertEqual(result.forecast_horizons, (1, 3, 6))
        self.assertEqual(tuple(result.predictions.shape), (4, 6, 8))
        self.assertTrue(torch.all(result.predictions >= 0.0))
        self.assertTrue(torch.allclose(result.predictions.sum(dim=-1), torch.ones(4, 6), atol=1e-5))
        self.assertTrue(result.metadata["origin_topology_frozen"])
        self.assertTrue(result.metadata["origin_links_frozen"])
        adjacency = result.diagnostics[0]["graph"]["adjacency"]
        self.assertGreater(float(adjacency[0, 2]), 0.0)
        self.assertGreater(
            float(result.diagnostics[0]["graph"]["modulation_l1"].mean()),
            0.0,
        )

    def test_topology_only_uses_unweighted_deployment_without_fake_link_metrics(self) -> None:
        values, mask, time, deployment, _ = synthetic_case()
        model = self._trained_model()
        result = recursive_forecast(
            model,
            values[:, 13:18],
            time[13:18],
            [1, 3],
            mask[:, 13:18],
            deployment,
            None,
        )
        graph = result.diagnostics[0]["graph"]
        self.assertEqual(result.metadata["link_mode"], "topology_only")
        self.assertFalse(result.metadata["real_link_features"])
        self.assertFalse(result.metadata["full_ours"])
        self.assertTrue(torch.all(graph["unweighted_topology"]))
        self.assertFalse(torch.any(graph["link_features_present"]))
        self.assertTrue(torch.isnan(graph["link_gate_mean"]).all())
        self.assertGreater(float(graph["neighbour_count"].sum()), 0.0)
        self.assertTrue(
            torch.allclose(result.predictions.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
        )

    def test_topology_only_training_updates_spatial_theta_but_not_link_gate(self) -> None:
        values, mask, time, deployment, _ = synthetic_case()
        adjacency = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        model = AlibabaOursModel(
            OursModelConfig(
                resource_dim=4,
                num_states=8,
                history_window=5,
                prototype_length=5,
                max_order=3,
                backoff_decay=0.02,
                backoff_max_age=5,
                message_passing_steps=2,
                message_hidden_dim=8,
                random_state=7,
            )
        )
        initial_theta = model.graph_block.theta.detach().clone()
        initial_link_gate = model.graph_block.link_gate.weight.detach().clone()
        result = train_model(
            model,
            values[:, :18],
            time[:18],
            mask[:, :18],
            adjacency,
            None,
            config=TrainingConfig(
                epochs=1,
                max_origins_per_epoch=2,
                prototype_candidates=12,
                early_stopping_patience=1,
                random_state=7,
            ),
        )
        self.assertFalse(torch.allclose(initial_theta, result.model.graph_block.theta))
        self.assertTrue(
            torch.equal(initial_link_gate, result.model.graph_block.link_gate.weight.detach())
        )

    def test_interrupted_recursive_forecast_resumes_identically(self) -> None:
        values, mask, time, deployment, links = synthetic_case()
        model = self._trained_model()
        full = recursive_forecast(
            model,
            values[:, 13:18],
            time[13:18],
            [1, 3, 6],
            mask[:, 13:18],
            deployment,
            links,
        )
        captured: dict[str, object] = {}

        def interrupt(h: int, state: dict[str, object]) -> None:
            if h == 3:
                captured.update(state)
                raise RuntimeError("intentional interruption")

        with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
            recursive_forecast(
                model,
                values[:, 13:18],
                time[13:18],
                [1, 3, 6],
                mask[:, 13:18],
                deployment,
                links,
                checkpoint_interval_h=3,
                checkpoint_callback=interrupt,
            )
        self.assertNotIn("backoff_state", captured)
        resumed = recursive_forecast(
            model,
            values[:, 13:18],
            time[13:18],
            [1, 3, 6],
            mask[:, 13:18],
            deployment,
            links,
            resume_state=captured,
        )
        self.assertTrue(torch.allclose(full.predictions, resumed.predictions, atol=1e-6))

    def test_metrics_cover_class_membership_calibration_and_resources(self) -> None:
        truth = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)
        prediction = np.asarray([[0.7, 0.3], [0.25, 0.75]], dtype=float)
        resources = np.asarray([[0.2, 0.4], [0.5, 0.7]], dtype=float)
        evaluated = evaluate_predictions(
            truth,
            prediction,
            labels=[0, 1],
            true_resources=resources,
            predicted_resources=resources + 0.01,
            resource_names=["cpu", "mem"],
        )
        self.assertIn("f1_weighted", evaluated["classification"])
        self.assertIn("js_divergence", evaluated["membership"])
        self.assertIn("ece", evaluated["calibration"])
        self.assertIn("overall", evaluated["resources"])


class RecoveryTests(unittest.TestCase):
    def test_json_writer_encodes_undefined_metrics_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            atomic_write_json(
                path,
                {
                    "rows": [
                        {
                            "r2": float("nan"),
                            "pearson_r": np.float64(float("inf")),
                            "mae": 0.25,
                        }
                    ]
                },
            )
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["rows"][0]["r2"])
            self.assertIsNone(payload["rows"][0]["pearson_r"])
            self.assertEqual(payload["rows"][0]["mae"], 0.25)

    def test_recursive_checkpoint_round_trip_and_hash_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recursive.npz"
            state = {
                "membership": np.asarray([[0.2, 0.8]], dtype=np.float32),
                "states": np.asarray([[1, 0, 1]], dtype=np.int64),
            }
            config = {"history_length": 5, "seed": 7}
            data = {"origin": "test-1"}
            save_recursive_checkpoint(
                path,
                recursive_state=state,
                forecast_horizon=60,
                config=config,
                data=data,
                run_id="test-run",
            )
            loaded = load_recursive_checkpoint(
                path,
                expected_config=config,
                expected_data=data,
            )
            np.testing.assert_allclose(loaded.state["recursive"]["membership"], state["membership"])
            self.assertEqual(loaded.metadata.forecast_horizon, 60)
            with self.assertRaises(CheckpointMismatchError):
                load_recursive_checkpoint(path, expected_config={"history_length": 15, "seed": 7})

    def test_progress_status_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ProgressTracker(tmp, run_id="run-7")
            tracker.start(total=2, stage="evaluate")
            tracker.update(increment=1, forecast_horizon=60, extra={"history_length": 5})
            tracker.complete(summary={"ok": True})
            status = read_status(Path(tmp) / "status.json")
            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed"], 2)
            self.assertEqual(status["details"]["history_length"], 5)

    def test_cmd_progress_line_reports_pass_bytes_rows_and_eta(self) -> None:
        line = format_console_progress(
            {
                "stage": "prepare-pass1",
                "completed": 50,
                "total": 200,
                "progress_fraction": 0.25,
                "elapsed_seconds": 10,
                "eta_seconds": 30,
                "rate_per_second": 5,
                "details": {
                    "byte_offset": 50,
                    "source_bytes": 100,
                    "rows_scanned": 123456,
                },
            }
        )
        self.assertIn("25.00%", line)
        self.assertIn("pass=1/2: 50.0%", line)
        self.assertIn("rows=123,456", line)
        self.assertIn("ETA=00:00:30", line)
        self.assertTrue(line.isascii())

    def test_two_pass_raw_scan_reuses_byte_offset_checkpoint(self) -> None:
        config = load_experiment_config(ROOT / "configs" / "experiment.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "usage.csv"
            lines: list[str] = []
            for workload, node in (("c1", "m1"), ("c2", "m2")):
                for minute in range(8):
                    seconds = 86400 + 60 * minute
                    lines.append(
                        f"{workload},{node},{seconds},10,20,0,0,0,0,0,30\n"
                    )
            raw.write_text("".join(lines), encoding="utf-8")
            source = SourceConfig(
                name="raw-test",
                format="csv",
                path=raw,
                has_header=False,
                columns=(
                    "container_id", "machine_id", "time_stamp", "cpu_util_percent",
                    "mem_util_percent", "cpi", "mem_gps", "mpki", "net_in",
                    "net_out", "disk_io_percent",
                ),
                workload_column="container_id",
                node_column="machine_id",
                time_column="time_stamp",
                time_unit="seconds",
                delimiter=",",
                resources=(
                    ResourceRule("cpu", "cpu_util_percent", 0.01, 0.0, None),
                    ResourceRule("mem", "mem_util_percent", 0.01, 0.0, None),
                    ResourceRule("disk_i", "disk_io_percent", 0.005, 0.0, 100.0),
                    ResourceRule("disk_o", "disk_io_percent", 0.005, 0.0, 100.0),
                ),
            )
            checkpoint = root / "checkpoints"
            progress_events: list[tuple[str, int, int, int]] = []
            first = scan_stats_resumable(
                source,
                config,
                checkpoint,
                resume=False,
                checkpoint_every_rows=2,
                progress_every_rows=2,
                progress_every_seconds=0.0,
                progress_callback=lambda *values: progress_events.append(values),
            )
            self.assertEqual(progress_events[0][2], 0)
            self.assertEqual(progress_events[-1][2], raw.stat().st_size)
            resumed = scan_stats_resumable(
                source,
                config,
                checkpoint,
                resume=True,
                checkpoint_every_rows=2,
                progress_every_rows=2,
            )
            self.assertEqual(first["c1"].valid_rows, resumed["c1"].valid_rows)
            values, observed, nodes = extract_selected_resumable(
                source,
                config,
                ("c1", "c2"),
                checkpoint,
                resume=False,
                checkpoint_every_rows=2,
                progress_every_rows=2,
                raw_capture_path=root / "candidate-raw.csv",
            )
            values_resumed, observed_resumed, nodes_resumed = extract_selected_resumable(
                source,
                config,
                ("c1", "c2"),
                checkpoint,
                resume=True,
                checkpoint_every_rows=2,
                progress_every_rows=2,
                raw_capture_path=root / "candidate-raw.csv",
            )
            np.testing.assert_allclose(values, values_resumed, equal_nan=True)
            np.testing.assert_array_equal(observed, observed_resumed)
            self.assertEqual(nodes, nodes_resumed)
            raw_output = root / "portable" / "container_usage_selected_200.csv"
            counts = export_selected_raw_rows(
                root / "candidate-raw.csv",
                raw_output,
                ("c1",),
                per_workload_dir=root / "portable" / "raw_by_workload",
            )
            expected = "".join(line for line in lines if line.startswith("c1,"))
            self.assertEqual(raw_output.read_text(encoding="utf-8"), expected)
            self.assertEqual(
                (root / "portable" / "raw_by_workload" / "c1.csv").read_text(
                    encoding="utf-8"
                ),
                expected,
            )
            self.assertEqual(counts, {"c1": 8})

    def test_training_resumes_from_saved_epoch(self) -> None:
        values, mask, time_values, deployment, links = synthetic_case()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch.npz"
            model_config = OursModelConfig(
                resource_dim=4,
                num_states=8,
                history_window=5,
                prototype_length=5,
                max_order=3,
                backoff_max_age=5,
                message_passing_steps=2,
                random_state=7,
            )

            def interrupt(epoch: int, metrics: dict[str, float], model: AlibabaOursModel) -> None:
                del metrics, model
                if epoch == 1:
                    raise RuntimeError("stop after checkpoint")

            with self.assertRaisesRegex(RuntimeError, "stop after checkpoint"):
                train_model(
                    AlibabaOursModel(model_config),
                    values[:, :18],
                    time_values[:18],
                    mask[:, :18],
                    deployment,
                    links,
                    config=TrainingConfig(
                        epochs=2,
                        max_origins_per_epoch=2,
                        prototype_candidates=12,
                        resume_checkpoint_path=checkpoint,
                        resume_existing=False,
                        run_identity_digest="identity-a",
                        relevant_data_digest="data-a",
                        random_state=7,
                    ),
                    epoch_callback=interrupt,
                )
            self.assertTrue(checkpoint.exists())
            with self.assertRaises(CheckpointMismatchError):
                train_model(
                    AlibabaOursModel(model_config),
                    values[:, :18],
                    time_values[:18],
                    mask[:, :18],
                    deployment,
                    links,
                    config=TrainingConfig(
                        epochs=2,
                        max_origins_per_epoch=2,
                        prototype_candidates=12,
                        resume_checkpoint_path=checkpoint,
                        resume_existing=True,
                        run_identity_digest="identity-code-changed",
                        relevant_data_digest="data-a",
                        random_state=7,
                    ),
                )
            with self.assertRaises(CheckpointMismatchError):
                train_model(
                    AlibabaOursModel(model_config),
                    values[:, :18],
                    time_values[:18],
                    mask[:, :18],
                    deployment,
                    links,
                    config=TrainingConfig(
                        epochs=2,
                        max_origins_per_epoch=2,
                        prototype_candidates=12,
                        resume_checkpoint_path=checkpoint,
                        resume_existing=True,
                        run_identity_digest="identity-a",
                        relevant_data_digest="data-changed",
                        random_state=7,
                    ),
                )
            resumed = train_model(
                AlibabaOursModel(model_config),
                values[:, :18],
                time_values[:18],
                mask[:, :18],
                deployment,
                links,
                config=TrainingConfig(
                    epochs=2,
                    max_origins_per_epoch=2,
                    prototype_candidates=12,
                    resume_checkpoint_path=checkpoint,
                    resume_existing=True,
                    run_identity_digest="identity-a",
                    relevant_data_digest="data-a",
                    random_state=7,
                ),
            )
            self.assertEqual([int(row["epoch"]) for row in resumed.history], [1, 2])


class NamingTests(unittest.TestCase):
    def test_source_uses_h_and_not_deprecated_recursive_symbols(self) -> None:
        forbidden = ("rollout_step", "evaluation_horizon")
        for path in (ROOT / "src" / "alibaba_ours_exp").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token} found in {path.name}")


if __name__ == "__main__":
    unittest.main()
