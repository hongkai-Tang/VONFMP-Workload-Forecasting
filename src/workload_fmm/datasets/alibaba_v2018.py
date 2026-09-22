from __future__ import annotations

import csv
import itertools
import json
import math
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import load_deployment_csv, load_resource_csv
from ..model import ModelConfig
from ..shape_set import ShapeSetConfig

ALIBABA_V2018_URL_BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
ALIBABA_V2018_FILES = {
    "machine_meta": "machine_meta.tar.gz",
    "machine_usage": "machine_usage.tar.gz",
    "container_meta": "container_meta.tar.gz",
    "container_usage": "container_usage.tar.gz",
    "batch_task": "batch_task.tar.gz",
    "batch_instance": "batch_instance.tar.gz",
}

ALIBABA_V2018_RESOURCE_NAMES = ("cpu", "mem", "disk_i", "disk_o")

# Public v2018 container_usage rows are usually headerless. These names follow
# the official schema and are also accepted when a header row is present.
CONTAINER_USAGE_COLUMNS = (
    "container_id",
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io_percent",
)


@dataclass
class AlibabaPrepareConfig:
    raw_dir: Path = Path("data/alibaba_v2018/raw")
    processed_dir: Path = Path("data/alibaba_v2018/processed")
    usage_name: str = "container_usage"
    max_workloads: int = 2000
    min_points: int = 30
    resample_seconds: int = 60
    resource_names: tuple[str, ...] = ALIBABA_V2018_RESOURCE_NAMES
    progress_every: int = 0
    stop_when_enough: bool = False
    candidate_multiplier: int = 10
    require_full_span: bool = False
    full_span_days: float = 8.0
    full_span_start_bucket: int | None = None
    full_span_start_seconds: int | None = 86400
    allowed_missing_fraction: float = 0.0


def alibaba_v2018_model_config() -> ModelConfig:
    """Return a model config matching Alibaba v2018's four resource metrics."""

    return ModelConfig(
        resource_names=ALIBABA_V2018_RESOURCE_NAMES,
        log_resource_names=(),
        shape_set=ShapeSetConfig(resource_weights=(1.0, 1.0, 1.0, 1.0)),
    )


def _csv_path(raw_dir: Path, usage_name: str) -> Path:
    candidates = [
        raw_dir / f"{usage_name}.csv",
        raw_dir / usage_name / f"{usage_name}.csv",
        raw_dir / f"{usage_name}.tar.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"could not find {usage_name}.csv or {usage_name}.tar.gz under {raw_dir}"
    )


def _open_usage_rows(path: Path):
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from _dict_rows(f)
        return
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(".csv")), None)
            if member is None:
                raise FileNotFoundError(f"no csv file found inside {path}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"could not extract {member.name} from {path}")
            import io

            text = io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="")
            yield from _dict_rows(text)
        return
    raise ValueError(f"unsupported usage file: {path}")


def _dict_rows(text_file):
    sample = text_file.readline()
    if not sample:
        return
    fields = [part.strip() for part in sample.rstrip("\n").split(",")]
    has_header = "container_id" in fields and "time_stamp" in fields
    if has_header:
        reader = csv.DictReader(text_file, fieldnames=fields)
    else:
        reader = csv.DictReader(
            itertools.chain([sample], text_file),
            fieldnames=CONTAINER_USAGE_COLUMNS,
        )
    for row in reader:
        yield row


def _float_or_nan(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _valid_metrics(cpu: float, mem: float, disk_io: float) -> bool:
    if not all(np.isfinite([cpu, mem, disk_io])):
        return False
    if cpu < 0.0 or mem < 0.0:
        return False
    # The Alibaba schema notes disk_io_percent can contain invalid -1/101 style values.
    if disk_io < 0.0 or disk_io > 100.0:
        return False
    return True


def _expected_full_span_points(full_span_days: float, bucket_size: int) -> int:
    return int(math.ceil(max(float(full_span_days), 0.0) * 86400.0 / float(bucket_size)))


def _resolve_full_span_start_bucket(
    cfg: AlibabaPrepareConfig,
    bucket_size: int,
    observed_min_bucket: int | None = None,
) -> int | None:
    if cfg.full_span_start_bucket is not None:
        return int(cfg.full_span_start_bucket)
    if cfg.full_span_start_seconds is not None:
        return int(math.floor(float(cfg.full_span_start_seconds) / float(bucket_size)))
    return observed_min_bucket


def _full_span_min_points(expected_points: int, allowed_missing_fraction: float) -> int:
    fraction = min(max(float(allowed_missing_fraction), 0.0), 1.0)
    allowed_missing = int(math.floor(expected_points * fraction))
    return max(expected_points - allowed_missing, 0)


def prepare_container_usage(config: AlibabaPrepareConfig | None = None) -> dict[str, object]:
    cfg = config or AlibabaPrepareConfig()
    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    usage_path = _csv_path(raw_dir, cfg.usage_name)

    selected: set[str] = set()
    selected_order: list[str] = []
    node_by_workload: dict[str, str] = {}
    sums: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=float))
    counts: dict[tuple[str, int], int] = defaultdict(int)
    buckets_by_workload: dict[str, set[int]] = defaultdict(set)
    full_span_buckets_by_workload: dict[str, set[int]] = defaultdict(set)
    invalid_rows = 0
    seen_rows = 0
    enough_workloads: set[str] = set()

    bucket_size = max(int(cfg.resample_seconds), 10)
    candidate_limit = (
        None
        if cfg.candidate_multiplier <= 0
        else max(cfg.max_workloads, cfg.max_workloads * max(cfg.candidate_multiplier, 1))
    )
    full_span_expected_points = _expected_full_span_points(cfg.full_span_days, bucket_size)
    full_span_min_required_points = _full_span_min_points(
        full_span_expected_points,
        cfg.allowed_missing_fraction,
    )
    full_span_start_bucket = _resolve_full_span_start_bucket(cfg, bucket_size)
    full_span_end_bucket = (
        None
        if full_span_start_bucket is None
        else full_span_start_bucket + full_span_expected_points
    )
    observed_min_bucket: int | None = None
    observed_max_bucket: int | None = None

    def has_enough_points(workload_id: str) -> bool:
        if cfg.require_full_span:
            if full_span_start_bucket is not None:
                return len(full_span_buckets_by_workload[workload_id]) >= full_span_min_required_points
            return len(buckets_by_workload[workload_id]) >= full_span_min_required_points
        return len(buckets_by_workload[workload_id]) >= cfg.min_points

    for row in _open_usage_rows(usage_path):
        seen_rows += 1
        if cfg.progress_every > 0 and seen_rows % cfg.progress_every == 0:
            print(
                "progress "
                f"rows={seen_rows} candidates={len(selected)} "
                f"enough={len(enough_workloads)} invalid={invalid_rows}",
                flush=True,
            )

        workload_id = str(row.get("container_id", "")).strip()
        machine_id = str(row.get("machine_id", "")).strip()
        if not workload_id or not machine_id:
            invalid_rows += 1
            continue
        if workload_id not in selected:
            if len(enough_workloads) >= cfg.max_workloads or (
                candidate_limit is not None and len(selected) >= candidate_limit
            ):
                continue
            selected.add(workload_id)
            selected_order.append(workload_id)
            node_by_workload[workload_id] = machine_id

        cpu = _float_or_nan(row.get("cpu_util_percent")) / 100.0
        mem = _float_or_nan(row.get("mem_util_percent")) / 100.0
        disk_io = _float_or_nan(row.get("disk_io_percent")) / 100.0
        if not _valid_metrics(cpu, mem, disk_io * 100.0):
            invalid_rows += 1
            continue
        time_stamp = int(_float_or_nan(row.get("time_stamp")))
        bucket = time_stamp // bucket_size
        observed_min_bucket = bucket if observed_min_bucket is None else min(observed_min_bucket, bucket)
        observed_max_bucket = bucket if observed_max_bucket is None else max(observed_max_bucket, bucket)
        key = (workload_id, bucket)
        disk_i = disk_io / 2.0
        disk_o = disk_io / 2.0
        sums[key] += np.asarray([cpu, mem, disk_i, disk_o], dtype=float)
        counts[key] += 1
        buckets_by_workload[workload_id].add(bucket)
        if (
            cfg.require_full_span
            and full_span_start_bucket is not None
            and full_span_end_bucket is not None
            and full_span_start_bucket <= bucket < full_span_end_bucket
        ):
            full_span_buckets_by_workload[workload_id].add(bucket)
        if has_enough_points(workload_id):
            enough_workloads.add(workload_id)

        if (
            cfg.stop_when_enough
            and len(enough_workloads) >= cfg.max_workloads
        ):
            print(
                "early_stop "
                f"rows={seen_rows} candidates={len(selected)} "
                f"enough={len(enough_workloads)}",
                flush=True,
            )
            break

    by_workload: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (workload_id, bucket), total in sums.items():
        by_workload[workload_id].append((bucket, total / max(counts[(workload_id, bucket)], 1)))
    if cfg.require_full_span and full_span_start_bucket is None:
        full_span_start_bucket = _resolve_full_span_start_bucket(cfg, bucket_size, observed_min_bucket)
        full_span_end_bucket = (
            None
            if full_span_start_bucket is None
            else full_span_start_bucket + full_span_expected_points
        )

    def kept_workload_ok(workload_id: str) -> bool:
        if workload_id not in by_workload:
            return False
        if not cfg.require_full_span:
            return len(by_workload[workload_id]) >= cfg.min_points
        if full_span_start_bucket is None or full_span_end_bucket is None:
            return False
        covered = {
            bucket
            for bucket, _ in by_workload[workload_id]
            if full_span_start_bucket <= bucket < full_span_end_bucket
        }
        return len(covered) >= full_span_min_required_points

    kept_ids = [
        workload_id
        for workload_id in selected_order
        if kept_workload_ok(workload_id)
    ][: cfg.max_workloads]
    kept: dict[str, list[tuple[int, np.ndarray]]] = {}
    for workload_id in kept_ids:
        rows = sorted(by_workload[workload_id])
        if cfg.require_full_span and full_span_start_bucket is not None and full_span_end_bucket is not None:
            rows = [
                (bucket, values)
                for bucket, values in rows
                if full_span_start_bucket <= bucket < full_span_end_bucket
            ]
        kept[workload_id] = rows

    resources_path = processed_dir / "resources.csv"
    deployment_path = processed_dir / "deployment.csv"
    with resources_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload_id", "time", *cfg.resource_names])
        for workload_id in sorted(kept):
            for bucket, values in kept[workload_id]:
                writer.writerow([workload_id, bucket, *[f"{v:.10g}" for v in values]])

    with deployment_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload_id", "node_id"])
        for workload_id in sorted(kept):
            writer.writerow([workload_id, node_by_workload.get(workload_id, "")])

    manifest = {
        "source": str(usage_path),
        "resources": str(resources_path),
        "deployment": str(deployment_path),
        "resource_names": list(cfg.resource_names),
        "resample_seconds": cfg.resample_seconds,
        "max_workloads": cfg.max_workloads,
        "min_points": cfg.min_points,
        "seen_rows": seen_rows,
        "invalid_rows": invalid_rows,
        "selected_workloads": len(selected),
        "kept_workloads": len(kept),
        "candidate_limit": candidate_limit if candidate_limit is not None else "unlimited",
        "observed_min_bucket": observed_min_bucket,
        "observed_max_bucket": observed_max_bucket,
        "require_full_span": cfg.require_full_span,
        "full_span_days": cfg.full_span_days if cfg.require_full_span else None,
        "full_span_start_bucket": full_span_start_bucket if cfg.require_full_span else None,
        "full_span_expected_points": full_span_expected_points if cfg.require_full_span else None,
        "full_span_min_required_points": full_span_min_required_points if cfg.require_full_span else None,
        "allowed_missing_fraction": cfg.allowed_missing_fraction if cfg.require_full_span else None,
        "note": (
            "Alibaba v2018 has no GPU metric and no separate disk-read/disk-write metrics. "
            "disk_i and disk_o are derived as disk_io_percent/2 each."
        ),
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_prepared(processed_dir: str | Path = "data/alibaba_v2018/processed"):
    root = Path(processed_dir)
    resources, mask, workload_ids, time_ids = load_resource_csv(
        root / "resources.csv",
        resource_names=ALIBABA_V2018_RESOURCE_NAMES,
    )
    deployment, _, node_ids = load_deployment_csv(
        root / "deployment.csv",
        workload_ids=workload_ids,
    )
    return resources, mask, deployment, workload_ids, time_ids, node_ids
