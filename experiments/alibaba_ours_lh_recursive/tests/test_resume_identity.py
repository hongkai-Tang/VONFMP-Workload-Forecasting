from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from alibaba_ours_exp.checkpoint import (
    CheckpointMismatchError,
    code_fingerprint,
    ensure_run_identity,
    make_run_identity,
    validate_artifact_guard,
    write_artifact_guard,
)


class RunIdentityTests(unittest.TestCase):
    def _layout(self, temporary: str) -> tuple[Path, Path]:
        our_root = Path(temporary) / "code" / "our"
        experiment = our_root / "experiments" / "example"
        (experiment / "src" / "package").mkdir(parents=True)
        (our_root / "src" / "workload_fmm").mkdir(parents=True)
        (experiment / "run_experiment.py").write_text("print('entry')\n", encoding="utf-8")
        (experiment / "src" / "package" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        (our_root / "src" / "workload_fmm" / "dtw.py").write_text(
            "def distance(): return 1\n", encoding="utf-8"
        )
        return our_root, experiment

    def test_code_hash_covers_only_entry_experiment_src_and_core_src(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, experiment = self._layout(temporary)
            before = code_fingerprint(experiment)["sha256"]
            (experiment / "tests").mkdir()
            (experiment / "tests" / "test_generated.py").write_text("noise\n", encoding="utf-8")
            (experiment / "runs").mkdir()
            (experiment / "runs" / "result.json").write_text("{}\n", encoding="utf-8")
            (experiment / "src" / "package" / "__pycache__").mkdir()
            (experiment / "src" / "package" / "__pycache__" / "model.pyc").write_bytes(b"noise")
            self.assertEqual(before, code_fingerprint(experiment)["sha256"])
            (experiment / "src" / "package" / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, code_fingerprint(experiment)["sha256"])

    def test_custom_run_id_cannot_bypass_input_or_code_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            our_root, experiment = self._layout(temporary)
            source = our_root / "input.csv"
            source.write_text("abc\n", encoding="utf-8")
            raw_config = {"seed": 7, "link_mode": "topology_only"}
            first = make_run_identity(
                raw_config=raw_config,
                experiment_root=experiment,
                project_root=our_root,
                input_paths={"resource:main": source},
            )
            run_dir = experiment / "runs" / "custom-id"
            ensure_run_identity(run_dir, first, resume=False)

            stat = source.stat()
            source.write_text("xyz\n", encoding="utf-8")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            changed_data = make_run_identity(
                raw_config=raw_config,
                experiment_root=experiment,
                project_root=our_root,
                input_paths={"resource:main": source},
            )
            with self.assertRaisesRegex(CheckpointMismatchError, "run identity mismatch"):
                ensure_run_identity(run_dir, changed_data, resume=True)

            source.write_text("abc\n", encoding="utf-8")
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            (experiment / "src" / "package" / "model.py").write_text("VALUE = 9\n", encoding="utf-8")
            changed_code = make_run_identity(
                raw_config=raw_config,
                experiment_root=experiment,
                project_root=our_root,
                input_paths={"resource:main": source},
            )
            with self.assertRaisesRegex(CheckpointMismatchError, "run identity mismatch"):
                ensure_run_identity(run_dir, changed_code, resume=True)

    def test_legacy_nonempty_directory_is_never_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            our_root, experiment = self._layout(temporary)
            identity = make_run_identity(
                raw_config={"seed": 7},
                experiment_root=experiment,
                project_root=our_root,
                input_paths={},
            )
            run_dir = experiment / "runs" / "legacy"
            run_dir.mkdir(parents=True)
            (run_dir / "old-result.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointMismatchError, "legacy run directory"):
                ensure_run_identity(run_dir, identity, resume=False)

    def test_artifact_guard_rejects_data_identity_and_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "dataset.bin"
            artifact.write_bytes(b"prepared-v1")
            guard = root / "complete.json"
            artifacts = {"data/dataset.bin": artifact}
            write_artifact_guard(
                guard,
                stage="prepare",
                run_identity_digest="identity-a",
                relevant_data_digest="data-a",
                artifacts=artifacts,
            )
            validate_artifact_guard(
                guard,
                stage="prepare",
                run_identity_digest="identity-a",
                relevant_data_digest="data-a",
                artifacts=artifacts,
            )
            with self.assertRaises(CheckpointMismatchError):
                validate_artifact_guard(
                    guard,
                    stage="prepare",
                    run_identity_digest="identity-b",
                    relevant_data_digest="data-a",
                    artifacts=artifacts,
                )
            with self.assertRaises(CheckpointMismatchError):
                validate_artifact_guard(
                    guard,
                    stage="prepare",
                    run_identity_digest="identity-a",
                    relevant_data_digest="data-b",
                    artifacts=artifacts,
                )
            artifact.write_bytes(b"prepared-v2")
            with self.assertRaisesRegex(CheckpointMismatchError, "contents changed"):
                validate_artifact_guard(
                    guard,
                    stage="prepare",
                    run_identity_digest="identity-a",
                    relevant_data_digest="data-a",
                    artifacts=artifacts,
                )


if __name__ == "__main__":
    unittest.main()
