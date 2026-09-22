from __future__ import annotations

"""Lossless suite-level collection for sensitivity outputs."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .design import STUDIES, build_conditions


CONDITION_COLUMNS = (
    "run_id",
    "sensitivity_condition_id",
    "sensitivity_study",
    "sensitivity_parameter",
    "sensitivity_value",
    "sensitivity_shared_baseline",
    "sensitivity_num_states",
    "sensitivity_max_order",
    "sensitivity_message_passing_steps",
    "sensitivity_time_decay_per_minute",
    "sensitivity_effective_time_decay_per_step",
    "sensitivity_granularity_seconds",
    "sensitivity_history_duration_seconds",
    "sensitivity_history_length",
    "sensitivity_forecast_horizon_steps",
    "sensitivity_forecast_duration_seconds",
    "sensitivity_origin_stride_steps",
    "sensitivity_origin_stride_seconds",
    "sensitivity_seed",
    "membership_definition",
    "prediction_strategy",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for key in CONDITION_COLUMNS:
        if key not in seen:
            seen.add(key)
            result.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = _fieldnames(rows)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_table(output: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_csv_atomic(output / f"{stem}.csv", rows)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        temporary = output / f"{stem}.parquet.tmp"
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output / f"{stem}.parquet")
    except (ImportError, OSError, ValueError):
        pass


def _condition_metadata(metadata: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    result = {"run_id": run_id}
    for key in CONDITION_COLUMNS:
        if key == "run_id":
            continue
        result[key] = metadata.get(key)
    return result


def _attach(
    rows: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any], run_id: str
) -> list[dict[str, Any]]:
    prefix = _condition_metadata(metadata, run_id)
    return [{**prefix, **dict(row)} for row in rows]


def _plot_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("sensitivity_shared_baseline", "")).lower() in {"true", "1"}:
            for study in STUDIES:
                output.append({**dict(row), "plot_study": study, "plot_is_baseline": True})
        else:
            output.append(
                {
                    **dict(row),
                    "plot_study": row.get("sensitivity_study"),
                    "plot_is_baseline": False,
                }
            )
    return output


def collect_suite(
    config_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = Path(config_path).resolve()
    runs = Path(runs_root).resolve()
    output = Path(output_dir).resolve()
    conditions = build_conditions(config)
    expected = {condition.condition_id for condition in conditions}
    expected_seed = conditions[0].seed
    found: set[str] = set()
    complete: set[str] = set()

    tables: dict[str, list[dict[str, Any]]] = {
        "metrics_by_h": [],
        "metrics_by_origin": [],
        "metrics_all_h": [],
        "per_class": [],
        "calibration_bins": [],
        "recursive_diagnostics": [],
        "baseline_metrics_by_origin": [],
        "baseline_metrics_by_condition": [],
        "error_quantiles": [],
        "transition_metrics": [],
        "training_history": [],
        "efficiency": [],
    }

    if runs.is_dir():
        for run_dir in sorted(path for path in runs.iterdir() if path.is_dir()):
            metadata_path = run_dir / "run_metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = _read_json(metadata_path)
            run_seed = metadata.get("sensitivity_seed")
            if run_seed is None or int(run_seed) != expected_seed:
                continue
            condition_id = metadata.get("sensitivity_condition_id")
            if not condition_id:
                continue
            condition_id = str(condition_id)
            found.add(condition_id)
            enrichment = run_dir / "metrics" / "enrichment.complete.json"
            if metadata.get("stage") == "aggregate_complete" and enrichment.is_file():
                enriched = _read_json(enrichment)
                if enriched.get("status") == "complete":
                    complete.add(condition_id)

            for stem in (
                "metrics_by_h",
                "metrics_by_origin",
                "metrics_all_h",
                "per_class",
                "calibration_bins",
                "recursive_diagnostics",
                "baseline_metrics_by_origin",
                "baseline_metrics_by_condition",
                "error_quantiles",
                "transition_metrics",
            ):
                path = run_dir / "metrics" / f"{stem}.csv"
                if path.is_file():
                    tables[stem].extend(_attach(_read_csv(path), metadata, run_dir.name))

            for path in sorted((run_dir / "models").glob("L=*/training_history.csv")):
                history_length = int(path.parent.name.split("=", 1)[1])
                rows = _read_csv(path)
                for row in rows:
                    row.setdefault("history_length", history_length)
                tables["training_history"].extend(_attach(rows, metadata, run_dir.name))

            runtime_path = run_dir / "condition_runtime.json"
            if runtime_path.is_file():
                tables["efficiency"].append(
                    {**_condition_metadata(metadata, run_dir.name), **_read_json(runtime_path)}
                )

    output.mkdir(parents=True, exist_ok=True)
    for stem, rows in tables.items():
        _write_table(output, stem, rows)
    _write_table(output, "plotting_metrics_by_h", _plot_rows(tables["metrics_by_h"]))
    _write_table(output, "plotting_efficiency", _plot_rows(tables["efficiency"]))

    manifest = {
        "config": str(config),
        "seed": expected_seed,
        "runs_root": str(runs),
        "expected_conditions": len(expected),
        "conditions_found": len(found),
        "conditions_complete": len(complete),
        "missing_conditions": sorted(expected - complete),
        "unexpected_conditions": sorted(found - expected),
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "per_sample_prediction_files": sum(
            len(list((path / "metrics").glob("enriched_predictions*.parquet")))
            for path in runs.iterdir()
            if path.is_dir()
        ) if runs.is_dir() else 0,
        "all_complete": expected == complete,
    }
    write_json_atomic(output / "collection_manifest.json", manifest)
    return manifest


__all__ = ["collect_suite", "write_csv_atomic", "write_json_atomic"]
