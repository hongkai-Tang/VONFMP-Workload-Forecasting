from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from alibaba_ours_exp.config import ConfigError, load_experiment_config
from alibaba_ours_exp.checkpoint import config_hash
from alibaba_ours_exp.data_index import build_data_index


class ConfigAndIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "experiment.json"

    def test_protocol_grid_and_split_are_exact(self) -> None:
        config = load_experiment_config(self.config_path)
        self.assertEqual(config.history_lengths, (5, 15, 30, 60, 180, 360, 720, 1440))
        self.assertEqual(
            config.forecast_horizons,
            (1, 5, 15, 30, 60, 180, 360, 600, 720, 1440, 1728),
        )
        self.assertEqual(config.link_mode, "topology_only")
        self.assertFalse(config.requires_real_link_features)
        self.assertFalse(config.link_feature_priority)
        self.assertFalse(config.link_feature_sources)
        self.assertEqual(config.protocol["message_passing_steps"], 2)
        index = build_data_index(config)
        self.assertEqual(len(index.time_seconds), 11520)
        self.assertEqual(
            [(item.name, item.size) for item in index.splits],
            [("train", 6912), ("validation", 2304), ("test", 2304)],
        )
        self.assertEqual(len(index.origins_for_split("validation")), 10)
        self.assertEqual(len(index.origins_for_split("test")), 10)

    def test_h_is_a_physical_sixty_second_offset(self) -> None:
        config = load_experiment_config(self.config_path)
        index = build_data_index(config)
        origin = index.origins_for_split("test")[0]
        for h in config.forecast_horizons:
            target = origin.target_index(h)
            delta = int(index.time_seconds[target] - index.time_seconds[origin.origin_index])
            self.assertEqual(delta, h * 60)

    def test_absolute_paths_are_rejected(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["sources"]["raw_csv"]["path"] = str(Path.cwd().resolve() / "raw.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_experiment_config(path)

    def test_link_mode_is_validated_and_changes_the_config_hash(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        topology_hash = config_hash(raw)
        raw["spatial"]["link_mode"] = "real_link_quality"
        self.assertNotEqual(topology_hash, config_hash(raw))

        raw["spatial"]["link_mode"] = "unsupported"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-link-mode.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_experiment_config(path)

    def test_topology_only_does_not_require_a_link_source_contract(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.pop("link_feature_priority", None)
        raw.pop("link_feature_sources", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology-only.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_experiment_config(path)
        self.assertEqual(config.link_mode, "topology_only")
        self.assertFalse(config.link_feature_sources)

    def test_real_link_quality_requires_a_link_source_contract(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["spatial"]["link_mode"] = "real_link_quality"
        raw.pop("link_feature_priority", None)
        raw.pop("link_feature_sources", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "real-link-quality.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
