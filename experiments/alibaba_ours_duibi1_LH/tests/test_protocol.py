from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alibaba_ours_duibi1.config import ExperimentConfig
from alibaba_ours_duibi1.data import valid_origins
from alibaba_ours_duibi1.utils import atomic_write_json


class ProtocolTests(unittest.TestCase):
    def test_exact_24_task_order(self) -> None:
        config = ExperimentConfig.load(ROOT / "configs" / "portable.json")
        tasks = config.task_queue()
        self.assertEqual(len(tasks), 24)
        self.assertEqual(
            [task.task_id for task in tasks[:3]],
            ["L1440-long_1p2d", "L1440-medium_8h", "L1440-short_60m"],
        )
        self.assertEqual(
            [task.task_id for task in tasks[3:10]],
            [f"L{length}-long_1p2d" for length in [5, 15, 30, 60, 180, 360, 720]],
        )
        self.assertEqual(
            [task.task_id for task in tasks[10:17]],
            [f"L{length}-medium_8h" for length in [5, 15, 30, 60, 180, 360, 720]],
        )
        self.assertEqual(
            [task.task_id for task in tasks[17:]],
            [f"L{length}-short_60m" for length in [5, 15, 30, 60, 180, 360, 720]],
        )

    def test_original_ours_fixed_hyperparameters_are_present(self) -> None:
        config = ExperimentConfig.load(ROOT / "configs" / "portable.json")
        self.assertEqual(config.model["max_order"], 3)
        self.assertEqual(config.model["time_decay"], 0.02)
        self.assertEqual(config.model["message_passing_steps"], 2)
        self.assertEqual(config.model["einstein_strength"], 0.5)
        self.assertEqual(config.model["backoff_blend"], 0.5)

    def test_origin_semantics_start_at_t_plus_one(self) -> None:
        origins = valid_origins(
            split_start=100,
            split_end=200,
            history_length=5,
            horizon_steps=60,
            stride=10,
            training=False,
        )
        self.assertEqual(origins, [100, 110, 120, 130])
        self.assertLessEqual(origins[-1] + 60, 199)

    def test_json_is_strict_when_metrics_contain_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"finite": 1.0, "nan": float("nan")})
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertIsNone(json.loads(raw)["nan"])


if __name__ == "__main__":
    unittest.main()
