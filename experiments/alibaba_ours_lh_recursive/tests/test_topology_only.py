from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from alibaba_ours_exp.cli import (
    _mode_metadata,
    _preflight_report,
    _unweighted_workload_adjacency,
    _write_run_metadata,
)
from alibaba_ours_exp.config import load_experiment_config


class TopologyOnlyAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "experiment.json"
        self.config = load_experiment_config(self.config_path)

    def test_formal_config_is_topology_only_without_link_sources(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["spatial"]["link_mode"], "topology_only")
        self.assertNotIn("link_feature_priority", raw)
        self.assertNotIn("link_feature_sources", raw)
        self.assertFalse(self.config.requires_real_link_features)

    def test_metadata_cannot_be_mistaken_for_full_ours(self) -> None:
        self.assertEqual(
            _mode_metadata(self.config),
            {
                "link_mode": "topology_only",
                "link_features_required": False,
                "real_link_features": False,
                "full_ours": False,
                "experiment_variant": "Ours-TopologyOnly",
            },
        )

    def test_missing_link_features_do_not_block_topology_preflight(self) -> None:
        def report(name: str) -> SimpleNamespace:
            return SimpleNamespace(
                path=Path(name),
                exists=True,
                readable=True,
                missing_columns=(),
                error=None,
                ok=True,
            )

        reports = {
            self.config.source_priority[0]: report("resources"),
            f"deployment_{self.config.deployment_priority[0]}": report("deployment"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            candidate_metadata = Path(temporary) / "container_meta.tar.gz"
            candidate_metadata.write_bytes(b"test metadata")
            config = replace(self.config, candidate_metadata_path=candidate_metadata)
            with patch("alibaba_ours_exp.cli.preflight_configured_sources", return_value=reports):
                payload, ok = _preflight_report(config)

        self.assertTrue(ok)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["link_features_required"])
        self.assertFalse(payload["real_link_features_ok"])
        self.assertNotIn("blocking_reason", payload)

    def test_synthetic_link_generator_is_confined_to_smoke_test(self) -> None:
        """Formal prepare/train/evaluate paths must never fabricate link quality."""

        source = (ROOT / "src" / "alibaba_ours_exp" / "cli.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        callers: list[str] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_synthetic_links"
            ):
                continue
            current: ast.AST | None = node
            while current is not None and not isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                current = parents.get(current)
            callers.append(current.name if isinstance(current, ast.FunctionDef) else "<module>")

        self.assertTrue(callers, "smoke-test should retain an explicit synthetic fixture")
        self.assertEqual(set(callers), {"command_smoke_test"})

    def test_formal_stage_overrides_an_earlier_smoke_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run_metadata(
                self.config,
                run_dir,
                stage="smoke_test_complete",
                formal_result=False,
            )
            _write_run_metadata(
                self.config,
                run_dir,
                stage="prepare",
                formal_result=True,
            )
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["formal_result"])
        self.assertEqual(metadata["stage"], "prepare")

    def test_square_deployment_incidence_is_not_mistaken_for_adjacency(self) -> None:
        deployment = np.asarray(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        adjacency = _unweighted_workload_adjacency(deployment)
        expected = np.asarray(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(adjacency, expected)


if __name__ == "__main__":
    unittest.main()
