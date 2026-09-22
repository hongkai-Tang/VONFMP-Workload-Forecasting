from __future__ import annotations

"""Cross-machine collection utilities for ablation outputs."""

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _publish_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _strict_json_value(value: Any) -> Any:
    """Represent undefined numeric metrics as JSON null, never NaN/Infinity."""

    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_strict_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: Any) -> Path:
    return _publish_text(
        path,
        json.dumps(
            _strict_json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _machine_for_variant(runs_root: Path, variant: str) -> str | None:
    for path in (runs_root / "machine_manifests").glob("*-machine.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if variant in value.get("assigned_variants", []):
            return str(value["machine_label"])
    return None


def collect_suite(runs_root: Path, output: Path) -> dict[str, Any]:
    """Merge all CSV metric families without discarding future columns."""

    run_records: list[dict[str, Any]] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    seen_variants: set[str] = set()
    completed_variants: set[str] = set()
    for metadata_path in sorted(runs_root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        variant = metadata.get("ablation_variant")
        if not variant:
            continue
        if (
            metadata.get("forecast_strategy") != "direct_multi_step"
            or metadata.get("membership_definition") != "per_slot"
        ):
            # Do not mix legacy recursive/window-labelled runs into the new
            # direct per-slot ablation tables, even when they share a variant.
            continue
        variant = str(variant)
        seen_variants.add(variant)
        if metadata.get("formal_result") and metadata.get("stage") == "aggregate_complete":
            completed_variants.add(variant)
        common = {
            "variant": variant,
            "display_name": metadata.get("ablation_display_name", metadata.get("experiment_variant")),
            "run_id": run_dir.name,
            "machine_label": _machine_for_variant(runs_root, variant),
            "stage": metadata.get("stage"),
            "formal_result": metadata.get("formal_result"),
            "forecast_strategy": metadata.get("forecast_strategy"),
            "membership_definition": metadata.get("membership_definition"),
            "target_window_steps": metadata.get("target_window_steps"),
            "prediction_feedback": metadata.get("prediction_feedback"),
            "run_dir": str(run_dir),
        }
        run_records.append(common)
        metrics_dir = run_dir / "metrics"
        if metrics_dir.is_dir():
            for csv_path in sorted(metrics_dir.glob("*.csv")):
                stem = csv_path.stem
                for row in _read_csv(csv_path):
                    tables.setdefault(stem, []).append({**common, **row})
        for history_path in sorted((run_dir / "models").glob("L=*/training_history.csv")):
            history_length = history_path.parent.name.split("=", 1)[-1]
            for row in _read_csv(history_path):
                tables.setdefault("training_history", []).append(
                    {**common, "history_length": history_length, **row}
                )

    output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output / "runs.csv", run_records)
    for stem, rows in tables.items():
        write_csv_atomic(output / f"{stem}.csv", rows)
    third_manifest_path = runs_root / "machine_manifests" / "third-machine.json"
    third_manifest = (
        json.loads(third_manifest_path.read_text(encoding="utf-8"))
        if third_manifest_path.is_file()
        else None
    )
    environment_check = {
        "target_machine": "third",
        "reference_machine": "second",
        "manifest_present": third_manifest is not None,
        "reports_p2200": bool(
            third_manifest and "P2200" in str(third_manifest.get("gpu_name", "")).upper()
        ),
        "manifest": third_manifest,
        "note": (
            "Install the dependency lock exported by the second machine on the third machine; "
            "this report validates the third machine only and does not imply three-machine sharding."
        ),
    }
    environment_check["ready"] = bool(
        environment_check["manifest_present"] and environment_check["reports_p2200"]
    )
    write_json_atomic(output / "third_machine_environment.json", environment_check)

    completeness = {
        "runs_found": len(run_records),
        "variants_found": sorted(seen_variants),
        "formal_variants_complete": sorted(completed_variants),
        "variants_missing": sorted(
            {"ns_mem", "hypergraph", "dyn_trans", "vo_markov", "multi_hop"}
            - completed_variants
        ),
        "tables": {name: len(rows) for name, rows in sorted(tables.items())},
        "third_machine_p2200_ready": environment_check["ready"],
        "output": str(output),
    }
    write_json_atomic(output / "collection_manifest.json", completeness)
    return completeness


__all__ = ["collect_suite", "write_csv_atomic", "write_json_atomic"]
