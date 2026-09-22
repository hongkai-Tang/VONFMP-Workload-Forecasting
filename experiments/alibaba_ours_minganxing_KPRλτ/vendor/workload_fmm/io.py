from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .graph import deployment_from_pairs
from .types import DEFAULT_RESOURCE_NAMES


def load_resource_csv(
    path: str | Path,
    resource_names: tuple[str, ...] = DEFAULT_RESOURCE_NAMES,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load CSV with columns workload_id,time and configurable resource names."""

    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"workload_id", "time", *resource_names}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    workload_ids = sorted({row["workload_id"] for row in rows})
    time_ids = sorted({row["time"] for row in rows}, key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else x)
    w_index = {v: i for i, v in enumerate(workload_ids)}
    t_index = {v: i for i, v in enumerate(time_ids)}
    values = np.full((len(workload_ids), len(time_ids), len(resource_names)), np.nan, dtype=float)
    mask = np.zeros((len(workload_ids), len(time_ids)), dtype=bool)
    for row in rows:
        n = w_index[row["workload_id"]]
        t = t_index[row["time"]]
        values[n, t] = [float(row[name]) for name in resource_names]
        mask[n, t] = True
    return values, mask, workload_ids, time_ids


def load_deployment_csv(
    path: str | Path,
    workload_ids: list[str] | None = None,
    node_ids: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load CSV with columns workload_id,node_id."""

    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"workload_id", "node_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        rows.extend(reader)
    workload_ids = workload_ids or sorted({row["workload_id"] for row in rows})
    node_ids = node_ids or sorted({row["node_id"] for row in rows})
    w_index = {v: i for i, v in enumerate(workload_ids)}
    node_index = {v: i for i, v in enumerate(node_ids)}
    pairs = [(node_index[row["node_id"]], w_index[row["workload_id"]]) for row in rows]
    return deployment_from_pairs(pairs, len(node_ids), len(workload_ids)), workload_ids, node_ids


def load_link_csv(path: str | Path, node_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV with columns src,dst,bw,delay,loss,avail."""

    node_index = {v: i for i, v in enumerate(node_ids)}
    features = np.full((len(node_ids), len(node_ids), 4), np.nan, dtype=float)
    mask = np.zeros((len(node_ids), len(node_ids)), dtype=bool)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src", "dst", "bw", "delay", "loss", "avail"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for row in reader:
            src = node_index[row["src"]]
            dst = node_index[row["dst"]]
            features[src, dst] = [
                float(row["bw"]),
                float(row["delay"]),
                float(row["loss"]),
                float(row["avail"]),
            ]
            mask[src, dst] = True
    return features, mask


def save_prediction_json(path: str | Path, prediction: dict) -> None:
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in prediction.items()
    }
    Path(path).write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
