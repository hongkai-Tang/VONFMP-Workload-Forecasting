from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .checkpoint import (
    config_hash,
    data_hash,
    ensure_run_identity,
    load_recursive_checkpoint,
    make_run_identity,
    save_recursive_checkpoint,
    validate_artifact_guard,
    write_artifact_guard,
)
from .config import ExperimentConfig, load_experiment_config
from .data_index import OriginRecord, build_data_index
from .dataset import (
    DatasetError,
    DenseDataset,
    build_dataset,
    preflight_configured_sources,
    resolve_link_feature_source,
)
from .metrics import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_parquet,
    evaluate_predictions,
    horizon_degradation,
)
from .model import AlibabaOursModel, OursModelConfig
from .progress import ProgressStateError, ProgressTracker, read_status
from .portable import atomic_copy_file, export_portable_cache
from .recursive_forecast import recursive_forecast
from .raw_prepare import (
    build_raw_dataset_resumable,
    build_seeded_candidate_dataset_resumable,
)
from .training import TrainingConfig, initialize_fixed_k_prototypes, train_model


EXIT_PREFLIGHT_FAILED = 2

LINK_MODE_REAL = "real_link_quality"
LINK_MODE_TOPOLOGY_ONLY = "topology_only"


def _raw_config(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.config_path.read_text(encoding="utf-8"))


def _identity_input_paths(config: ExperimentConfig) -> dict[str, Path]:
    paths = {
        f"resource:{name}": config.sources[name].path
        for name in config.source_priority
    }
    paths.update(
        {
            f"deployment:{name}": config.deployment_sources[name].path
            for name in config.deployment_priority
        }
    )
    if config.requires_real_link_features:
        paths.update(
            {
                f"link:{name}": config.link_feature_sources[name].path
                for name in config.link_feature_priority
            }
        )
    if config.candidate_metadata_path is not None:
        paths["selection:candidate_metadata"] = config.candidate_metadata_path
    if config.portable_dataset_path is not None:
        paths["portable:dataset"] = config.portable_dataset_path
        portable_manifest = config.portable_dataset_path.with_name("manifest.json")
        if portable_manifest.is_file():
            paths["portable:manifest"] = portable_manifest
    return paths


def _current_run_identity(config: ExperimentConfig) -> dict[str, Any]:
    return make_run_identity(
        raw_config=_raw_config(config),
        experiment_root=Path(__file__).resolve().parents[2],
        project_root=config.project_root,
        input_paths=_identity_input_paths(config),
    )


def _run_id(config: ExperimentConfig, identity: Mapping[str, Any] | None = None) -> str:
    current = dict(identity or _current_run_identity(config))
    digest = str(current["identity_hash"])[:12]
    variant = "topology-only" if config.link_mode == LINK_MODE_TOPOLOGY_ONLY else "real-link-quality"
    return f"alibaba-ours-{variant}-lh-seed7-{digest}"


def _run_dir(
    config: ExperimentConfig,
    run_id: str | None = None,
    identity: Mapping[str, Any] | None = None,
) -> Path:
    return config.output_dir / (run_id or _run_id(config, identity))


def _bind_run(config: ExperimentConfig, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    identity = _current_run_identity(config)
    run_dir = _run_dir(config, getattr(args, "run_id", None), identity)
    bound = ensure_run_identity(run_dir, identity, resume=bool(getattr(args, "resume", False)))
    return run_dir, bound


def _tracker(config: ExperimentConfig, run_dir: Path) -> ProgressTracker:
    return ProgressTracker(run_dir / "progress", run_id=run_dir.name)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str), flush=True)


def _link_mode(config: ExperimentConfig) -> str:
    """Return the configured spatial mode while remaining backward compatible."""
    spatial = getattr(config, "spatial", None)
    if not isinstance(spatial, Mapping):
        spatial = _raw_config(config).get("spatial", {})
    mode = str(spatial.get("link_mode", LINK_MODE_REAL)).strip().lower()
    if mode not in {LINK_MODE_REAL, LINK_MODE_TOPOLOGY_ONLY}:
        raise DatasetError(
            "spatial.link_mode must be 'real_link_quality' or 'topology_only'"
        )
    return mode


def _mode_metadata(config: ExperimentConfig) -> dict[str, Any]:
    mode = _link_mode(config)
    has_real_links = mode == LINK_MODE_REAL
    return {
        "link_mode": mode,
        "link_features_required": has_real_links,
        "real_link_features": has_real_links,
        "full_ours": has_real_links,
        "experiment_variant": "Ours" if has_real_links else "Ours-TopologyOnly",
    }


def _write_run_metadata(
    config: ExperimentConfig,
    run_dir: Path,
    **extra: Any,
) -> None:
    path = run_dir / "run_metadata.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                existing.update(value)
        except (OSError, json.JSONDecodeError):
            pass
    identity_path = run_dir / "run_identity.json"
    identity_digest = None
    if identity_path.is_file():
        try:
            identity_digest = json.loads(identity_path.read_text(encoding="utf-8")).get(
                "identity_hash"
            )
        except (OSError, json.JSONDecodeError):
            identity_digest = None
    existing.update(
        {
            "run_id": run_dir.name,
            "seed": int(config.protocol["seed"]),
            "forecast_strategy": config.protocol["forecast_strategy"],
            "config_hash": config_hash(_raw_config(config)),
            "run_identity_hash": identity_digest,
            **_mode_metadata(config),
            **extra,
        }
    )
    atomic_write_json(path, existing)


def _prepare_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "data/dataset.npz": run_dir / "data" / "dataset.npz",
        "data/manifest.json": run_dir / "data" / "manifest.json",
        "data/origins.csv": run_dir / "data" / "origins.csv",
        "config_resolved.json": run_dir / "config_resolved.json",
    }


def _prepared_data_digest(run_dir: Path, identity_digest: str) -> str:
    guard = validate_artifact_guard(
        run_dir / "data" / "prepare.complete.json",
        stage="prepare",
        run_identity_digest=identity_digest,
        relevant_data_digest=identity_digest,
        artifacts=_prepare_artifacts(run_dir),
    )
    return str(guard["artifact_hash"])


def _preflight_report(config: ExperimentConfig) -> tuple[dict[str, Any], bool]:
    reports = preflight_configured_sources(config)
    mode_metadata = _mode_metadata(config)
    selected_resource = config.source_priority[0]
    resource_ok = bool(reports.get(selected_resource) and reports[selected_resource].ok)
    deployment_ok = any(
        reports.get(f"deployment_{name}") is not None
        and reports[f"deployment_{name}"].ok
        for name in config.deployment_priority
    )
    link_ok = any(
        reports.get(f"link_features_{name}") is not None
        and reports[f"link_features_{name}"].ok
        for name in config.link_feature_priority
    )
    link_required = bool(mode_metadata["link_features_required"])
    candidate_metadata_required = (
        config.selection_strategy == "seeded_metadata_candidates"
    )
    candidate_metadata_ok = bool(
        not candidate_metadata_required
        or (
            config.candidate_metadata_path is not None
            and config.candidate_metadata_path.is_file()
        )
    )
    portable_dataset_ok = bool(
        config.portable_dataset_path is None or config.portable_dataset_path.is_file()
    )
    requested_device = str(config.training.get("device", "cpu")).strip().lower()
    cuda_required = requested_device.startswith("cuda")
    cuda_available = bool(torch.cuda.is_available())
    compute_device_ok = bool(not cuda_required or cuda_available)
    cuda_device_name = None
    cuda_capability = None
    if cuda_available:
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_capability = list(torch.cuda.get_device_capability(0))
    payload = {
        "ok": (
            resource_ok
            and deployment_ok
            and candidate_metadata_ok
            and portable_dataset_ok
            and compute_device_ok
            and (link_ok or not link_required)
        ),
        "config": str(config.config_path),
        "forecast_strategy": config.protocol["forecast_strategy"],
        "seed": config.protocol["seed"],
        "history_lengths": list(config.history_lengths),
        "forecast_horizons": list(config.forecast_horizons),
        "max_forecast_horizon": config.max_forecast_horizon,
        "resource_source_ok": resource_ok,
        "deployment_source_ok": deployment_ok,
        "candidate_metadata_ok": candidate_metadata_ok,
        "candidate_metadata_path": (
            None
            if config.candidate_metadata_path is None
            else str(config.candidate_metadata_path)
        ),
        "selection_strategy": config.selection_strategy,
        "candidate_pool_size": config.candidate_pool_size,
        "portable_dataset_ok": portable_dataset_ok,
        "portable_dataset_path": (
            None if config.portable_dataset_path is None else str(config.portable_dataset_path)
        ),
        "requested_device": requested_device,
        "compute_device_ok": compute_device_ok,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "cuda_compute_capability": cuda_capability,
        "real_link_features_ok": link_ok,
        **mode_metadata,
        "reports": {
            name: {
                "path": str(report.path),
                "exists": report.exists,
                "readable": report.readable,
                "missing_columns": list(report.missing_columns),
                "error": report.error,
                "ok": report.ok,
            }
            for name, report in reports.items()
        },
    }
    if not compute_device_ok:
        payload["blocking_reason"] = (
            "配置要求CUDA GPU，但当前PyTorch无法使用CUDA；请安装支持该显卡的CUDA版PyTorch。"
        )
    elif link_required and not link_ok:
        payload["blocking_reason"] = (
            "严格完整 Ours 需要真实 link_features；请按 inputs/README.md 提供文件。"
        )
    elif not resource_ok or not deployment_ok or not candidate_metadata_ok or not portable_dataset_ok:
        payload["blocking_reason"] = (
            "资源数据、真实部署关系和候选任务元数据必须通过预检。"
        )
    elif not link_required:
        payload["link_features_note"] = (
            "topology_only 模式仅使用真实部署拓扑；不会读取或伪造链路质量特征。"
        )
    return payload, bool(payload["ok"])


def command_preflight(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_dir, _ = _bind_run(config, args)
    report, ok = _preflight_report(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "preflight.json", report)
    _write_run_metadata(config, run_dir, preflight_ok=ok)
    _print_json(report)
    return 0 if ok else EXIT_PREFLIGHT_FAILED


def _dataset_arrays(dataset: DenseDataset) -> dict[str, np.ndarray]:
    return {
        "workload_ids": np.asarray(dataset.workload_ids, dtype=str),
        "workload_node_ids": np.asarray(dataset.workload_node_ids, dtype=str),
        "node_ids": np.asarray(dataset.node_ids, dtype=str),
        "resource_names": np.asarray(dataset.resource_names, dtype=str),
        "deployment": dataset.deployment.astype(np.float32),
        "observed_values": dataset.observed_values.astype(np.float32),
        "input_values": dataset.input_values.astype(np.float32),
        "target_observed_mask": dataset.target_observed_mask.astype(np.uint8),
        "input_valid_mask": dataset.input_valid_mask.astype(np.uint8),
        "imputed_mask": dataset.imputed_mask.astype(np.uint8),
        "time_seconds": dataset.data_index.time_seconds.astype(np.int64),
        "cohort_eligible_mask": dataset.cohort.eligible_mask.astype(np.uint8),
        "history_coverage": dataset.cohort.history_coverage.astype(np.float32),
        "cohort_target_observed_mask": dataset.cohort.target_observed_mask.astype(np.uint8),
    }


def _dataset_manifest(dataset: DenseDataset, config: ExperimentConfig) -> dict[str, Any]:
    point_counts = [item.valid_time_points for item in dataset.workload_stats.values()]
    return {
        **_mode_metadata(config),
        "source_name": dataset.source_name,
        "source_path": str(dataset.source_report.path),
        "workload_count": len(dataset.workload_ids),
        "node_count": len(dataset.node_ids),
        "resource_names": list(dataset.resource_names),
        "time_steps": int(dataset.input_values.shape[1]),
        "node_assignment_source": dataset.node_assignment_source,
        "valid_time_points_min": int(min(point_counts)) if point_counts else 0,
        "valid_time_points_max": int(max(point_counts)) if point_counts else 0,
        "eligible_pairs": int(dataset.cohort.eligible_mask.sum()),
        "target_observed_pairs": int(dataset.cohort.target_observed_mask.sum()),
        "config_hash": config_hash(_raw_config(config)),
        "data_hash": data_hash(
            {
                "workload_ids": dataset.workload_ids,
                "time_seconds": dataset.data_index.time_seconds,
                "observed_mask": dataset.target_observed_mask,
            }
        ),
    }


def _origin_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in build_data_index(config).origins:
        rows.append(
            {
                "origin_id": origin.origin_id,
                "split": origin.split,
                "origin_index": origin.origin_index,
                "origin_seconds": origin.origin_seconds,
                "history_start_index": origin.history_start_index,
                "forecast_end_index": origin.forecast_end_index,
                "forecast_end_seconds": origin.forecast_end_seconds,
            }
        )
    return rows


def command_prepare(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_dir, identity = _bind_run(config, args)
    identity_digest = str(identity["identity_hash"])
    report, ok = _preflight_report(config)
    if not ok:
        _print_json(report)
        print("prepare 已停止：正式输入未通过 preflight。", file=sys.stderr)
        return EXIT_PREFLIGHT_FAILED
    _write_run_metadata(config, run_dir, stage="prepare", formal_result=True)
    cache_path = run_dir / "data" / "dataset.npz"
    manifest_path = run_dir / "data" / "manifest.json"
    portable_dir = config.config_path.parent.parent / "inputs" / "alibaba_selected_200"
    if args.resume and cache_path.exists() and manifest_path.exists():
        _prepared_data_digest(run_dir, identity_digest)
        portable_manifest = portable_dir / "manifest.json"
        if not portable_manifest.exists():
            print("正在从已准备的数据导出便携式200任务CSV……", flush=True)
            export_portable_cache(
                _load_cache(run_dir),
                portable_dir,
                split_ratios=config.split_ratios,
                seed=int(config.protocol["seed"]),
            )
        print(f"prepare 已完成，复用 {cache_path}")
        return 0
    tracker = _tracker(config, run_dir)
    if config.portable_dataset_path is not None:
        try:
            tracker.start(total=1, stage="prepare-portable", message="载入便携式200任务缓存")
        except ProgressStateError:
            tracker.start(
                total=1,
                stage="prepare-portable",
                message="载入便携式200任务缓存",
                reset=True,
            )
        try:
            with np.load(config.portable_dataset_path, allow_pickle=False) as archive:
                portable_cache = {name: archive[name] for name in archive.files}
            required = {
                "workload_ids", "workload_node_ids", "resource_names", "observed_values",
                "input_values", "target_observed_mask", "input_valid_mask", "imputed_mask",
                "time_seconds", "deployment", "node_ids", "cohort_eligible_mask",
                "history_coverage", "cohort_target_observed_mask",
            }
            missing = sorted(required.difference(portable_cache))
            if missing:
                raise DatasetError(f"便携式dataset.npz缺少字段: {missing}")
            workload_count = len(portable_cache["workload_ids"])
            if workload_count != config.max_workloads:
                raise DatasetError(
                    f"便携式缓存包含{workload_count}个任务，预期{config.max_workloads}个"
                )
            if portable_cache["input_values"].shape[:2] != (
                config.max_workloads,
                config.total_steps,
            ):
                raise DatasetError("便携式缓存的任务数或时间轴长度与配置不一致")
            if tuple(portable_cache["resource_names"].astype(str)) != config.resource_names:
                raise DatasetError("便携式缓存的资源字段与配置不一致")
            atomic_write_npz(cache_path, portable_cache)
            portable_sidecar = config.portable_dataset_path.with_name("dataset_manifest.json")
            if portable_sidecar.is_file():
                prepared_manifest = json.loads(portable_sidecar.read_text(encoding="utf-8"))
            else:
                prepared_manifest = {
                    "source_name": "portable_prepared_cache",
                    "workload_count": workload_count,
                    "time_steps": config.total_steps,
                    "resource_names": list(config.resource_names),
                }
            prepared_manifest.update(
                {
                    "source_name": "portable_prepared_cache",
                    "portable_dataset_path": str(config.portable_dataset_path),
                    "config_hash": config_hash(_raw_config(config)),
                }
            )
            atomic_write_json(manifest_path, prepared_manifest)
            atomic_write_csv(run_dir / "data" / "origins.csv", _origin_records(config))
            atomic_write_json(run_dir / "config_resolved.json", _raw_config(config))
            write_artifact_guard(
                run_dir / "data" / "prepare.complete.json",
                stage="prepare",
                run_identity_digest=identity_digest,
                relevant_data_digest=identity_digest,
                artifacts=_prepare_artifacts(run_dir),
                extra={"link_mode": config.link_mode, "portable_prepared": True},
            )
            tracker.update(completed=1, stage="prepare-portable", message="便携式数据已就绪")
            tracker.complete(summary={"dataset": str(cache_path), "portable_prepared": True})
            _write_run_metadata(
                config,
                run_dir,
                stage="prepare_complete",
                formal_result=True,
                portable_prepared=True,
            )
        except Exception as exc:
            tracker.fail(exc, context={"stage": "prepare-portable"})
            raise
        print(f"便携式数据准备完成：{cache_path}")
        return 0
    formal_source = config.sources[config.source_priority[0]]
    resumable_raw = formal_source.format == "csv" and not formal_source.has_header
    one_pass_candidates = (
        resumable_raw and config.selection_strategy == "seeded_metadata_candidates"
    )
    source_bytes = formal_source.path.stat().st_size if resumable_raw else 0
    progress_total = (
        source_bytes if one_pass_candidates else (source_bytes * 2 if resumable_raw else 2)
    )
    try:
        tracker.start(
            total=progress_total,
            stage="prepare",
            message="扫描正式资源数据",
            resume=args.resume,
        )
    except ProgressStateError:
        tracker.start(
            total=progress_total,
            stage="prepare",
            message="扫描正式资源数据",
            reset=True,
        )
    try:
        if resumable_raw:
            first_scan_progress = True

            def scan_progress(stage: str, rows: int, byte_offset: int, total_bytes: int) -> None:
                nonlocal first_scan_progress
                completed_bytes = byte_offset + (total_bytes if stage == "prepare-pass2" else 0)
                total_progress_bytes = (
                    total_bytes if stage == "prepare-one-pass" else total_bytes * 2
                )
                tracker.update(
                    completed=completed_bytes,
                    total=total_progress_bytes,
                    stage=stage,
                    message=f"rows={rows}, bytes={byte_offset}/{total_bytes}",
                    extra={
                        "rows_scanned": rows,
                        "byte_offset": byte_offset,
                        "source_bytes": total_bytes,
                        "rebase_rate": first_scan_progress and byte_offset > 0,
                    },
                )
                first_scan_progress = False

            builder = (
                build_seeded_candidate_dataset_resumable
                if one_pass_candidates
                else build_raw_dataset_resumable
            )
            if one_pass_candidates:
                dataset = builder(
                    config,
                    formal_source,
                    run_dir / "checkpoints" / "prepare",
                    resume=args.resume,
                    progress_callback=scan_progress,
                    raw_export_path=portable_dir / "container_usage_selected_200.csv",
                )
            else:
                dataset = builder(
                    config,
                    formal_source,
                    run_dir / "checkpoints" / "prepare",
                    resume=args.resume,
                    progress_callback=scan_progress,
                )
            tracker.update(stage="prepare", message="写入连续时间数据缓存")
        else:
            dataset = build_dataset(config)
            tracker.update(completed=1, stage="prepare", message="写入连续时间数据缓存")
        atomic_write_npz(cache_path, _dataset_arrays(dataset))
        atomic_write_json(manifest_path, _dataset_manifest(dataset, config))
        atomic_write_csv(run_dir / "data" / "origins.csv", _origin_records(config))
        atomic_write_json(run_dir / "config_resolved.json", _raw_config(config))
        atomic_copy_file(cache_path, portable_dir / "dataset.npz")
        atomic_copy_file(manifest_path, portable_dir / "dataset_manifest.json")
        tracker.update(stage="portable-export", message="导出200任务便携式CSV")
        portable_manifest = export_portable_cache(
            _dataset_arrays(dataset),
            portable_dir,
            split_ratios=config.split_ratios,
            seed=int(config.protocol["seed"]),
        )
        write_artifact_guard(
            run_dir / "data" / "prepare.complete.json",
            stage="prepare",
            run_identity_digest=identity_digest,
            relevant_data_digest=identity_digest,
            artifacts=_prepare_artifacts(run_dir),
            extra={"link_mode": config.link_mode},
        )
        if not resumable_raw:
            tracker.update(completed=2, stage="prepare", message="数据准备完成")
        tracker.complete(
            summary={
                "dataset": str(cache_path),
                "portable_csv": str(portable_dir / portable_manifest["csv_file"]),
            }
        )
        _write_run_metadata(config, run_dir, stage="prepare_complete", formal_result=True)
    except Exception as exc:
        tracker.fail(exc, context={"stage": "prepare"})
        raise
    print(f"数据准备完成：{cache_path}")
    return 0


def _load_cache(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "data" / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}；请先执行 prepare")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _normalization(cache: Mapping[str, np.ndarray], train_stop: int) -> tuple[np.ndarray, np.ndarray]:
    values = cache["input_values"][:, :train_stop].astype(np.float64)
    mask = cache["input_valid_mask"][:, :train_stop].astype(bool)
    flat = values[mask]
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalized_resources(cache: Mapping[str, np.ndarray], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = (cache["input_values"].astype(np.float32) - mean) / std
    return np.where(np.isfinite(values), values, 0.0).astype(np.float32)


def _eligible_workloads_for_origin(
    config: ExperimentConfig,
    input_valid_mask: np.ndarray,
    target_observed_mask: np.ndarray,
    origin: OriginRecord,
) -> np.ndarray:
    """Rebuild cohort eligibility for the active L/h origin grid.

    Portable NPZ files can be reused with a different forecast horizon.  Their
    cached cohort arrays belong to the origin grid used during export, so they
    must not be indexed positionally after L or h changes.
    """

    history = input_valid_mask[
        :,
        origin.history_start_index : origin.origin_index + 1,
    ]
    if history.shape[1] != config.max_history:
        raise DatasetError(
            f"origin {origin.origin_id} history width {history.shape[1]} "
            f"does not match max_history={config.max_history}"
        )
    coverage = history.mean(axis=1, dtype=np.float64)
    keep = coverage >= config.min_history_coverage
    keep &= input_valid_mask[:, origin.origin_index]
    if config.require_all_forecast_targets_observed:
        target_indices = np.asarray(origin.target_indices, dtype=int)
        keep &= target_observed_mask[:, target_indices].all(axis=1)
    return keep


def _model_config(config: ExperimentConfig, history_length: int, resource_dim: int) -> OursModelConfig:
    p = config.protocol
    return OursModelConfig(
        resource_dim=resource_dim,
        num_states=int(p["num_states"]),
        history_window=int(history_length),
        prototype_length=int(p["prototype_length"]),
        einstein_strength=float(p["einstein_strength"]),
        max_order=int(p["max_order"]),
        backoff_decay=float(p["time_decay"]),
        backoff_time_unit=1.0,
        backoff_max_age=float(history_length),
        backoff_blend=float(p["backoff_blend"]),
        time_period=float(p["time_period_steps"]),
        transition_hidden_dim=int(p.get("transition_hidden_dim", 32)),
        message_passing_steps=int(p["message_passing_steps"]),
        message_hidden_dim=int(p.get("message_hidden_dim", 8)),
        random_state=int(p["seed"]),
    )


def _training_config(
    config: ExperimentConfig,
    checkpoint_path: Path,
    *,
    resume: bool,
    run_identity_digest: str | None = None,
    relevant_data_digest: str | None = None,
    run_id: str | None = None,
) -> TrainingConfig:
    t = config.training
    return TrainingConfig(
        epochs=int(t["epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        gradient_clip=float(t["gradient_clip"]),
        max_origins_per_epoch=int(t["max_origins_per_epoch"]),
        dynamic_loss_weight=float(t["dynamic_loss_weight"]),
        adaptive_order_loss_weight=float(t["adaptive_order_loss_weight"]),
        early_stopping_patience=int(t["early_stopping_patience"]),
        device=str(t.get("device", "cpu")),
        random_state=int(config.protocol["seed"]),
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=checkpoint_path.with_name("epoch-resume.npz"),
        resume_existing=resume,
        run_identity_digest=run_identity_digest,
        relevant_data_digest=relevant_data_digest,
        run_id=run_id,
    )


def _load_link_rows(config: ExperimentConfig):
    resolved = resolve_link_feature_source(config)
    if resolved is None:
        raise DatasetError("真实 link_features 未通过预检")
    source, _ = resolved
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("读取链路文件需要 pandas") from exc
    if source.format == "parquet":
        return pd.read_parquet(source.path)
    return pd.read_csv(source.path)


def _link_matrix(
    rows: Any,
    node_ids: Iterable[str],
    *,
    time_bucket: int | None = None,
    latest_not_after: bool = False,
) -> np.ndarray:
    nodes = tuple(str(value) for value in node_ids)
    positions = {node: index for index, node in enumerate(nodes)}
    selected = rows
    if time_bucket is not None:
        if latest_not_after:
            eligible = rows[rows["time_bucket"] <= int(time_bucket)]
            if eligible.empty:
                raise DatasetError(f"h={time_bucket} 之前没有真实链路快照")
            selected_time = int(eligible["time_bucket"].max())
            selected = eligible[eligible["time_bucket"] == selected_time]
        else:
            selected = rows[rows["time_bucket"] == int(time_bucket)]
    else:
        selected = rows
    matrix = np.full((len(nodes), len(nodes), 4), np.nan, dtype=np.float32)
    counts = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    for row in selected.itertuples(index=False):
        src = positions.get(str(row.src_node))
        dst = positions.get(str(row.dst_node))
        if src is None or dst is None:
            continue
        values = np.asarray([row.bandwidth, row.delay, row.loss, row.availability], dtype=np.float32)
        if not np.isfinite(values).all() or not 0.0 <= values[2] <= 1.0 or not 0.0 <= values[3] <= 1.0:
            raise DatasetError("link_features 含非法数值")
        if counts[src, dst] == 0:
            matrix[src, dst] = values
        else:
            matrix[src, dst] += values
        counts[src, dst] += 1
    valid = counts > 0
    matrix[valid] /= counts[valid, None]
    if not valid.any():
        raise DatasetError("所选节点在真实链路文件中没有可用边")
    # 缺边保持中性零值并由图邻接掩码排除；绝不估计真实链路数值。
    return np.nan_to_num(matrix, nan=0.0)


def _topology_coverage(topology: np.ndarray, hops: int) -> dict[str, Any]:
    """Summarise the unweighted workload graph induced by deployment."""
    values = np.asarray(topology, dtype=np.float64)
    if values.ndim != 2:
        raise DatasetError("deployment topology must be a 2-D matrix")
    if values.shape[0] == values.shape[1]:
        adjacency = values > 0
    else:
        occupied = values > 0
        adjacency = np.any(occupied[:, :, None] & occupied[:, None, :], axis=0)
    adjacency = np.asarray(adjacency | adjacency.T, dtype=bool)
    np.fill_diagonal(adjacency, False)
    workload_count = int(adjacency.shape[0])
    degrees = adjacency.sum(axis=1).astype(np.int64)

    components: list[int] = []
    unseen = set(range(workload_count))
    while unseen:
        root = unseen.pop()
        stack = [root]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbours = set(np.flatnonzero(adjacency[current]).tolist()).intersection(unseen)
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        components.append(size)

    reachable = adjacency.copy()
    frontier = adjacency.copy()
    for _ in range(1, max(1, int(hops))):
        next_frontier = np.zeros_like(frontier)
        for source in range(workload_count):
            through = np.flatnonzero(frontier[source])
            if through.size:
                next_frontier[source] = np.any(adjacency[through], axis=0)
        frontier = next_frontier
        reachable |= frontier
    np.fill_diagonal(reachable, False)
    reachable_counts = reachable.sum(axis=1) if workload_count else np.asarray([], dtype=int)
    nonisolated = degrees > 0
    return {
        "graph_workload_count": workload_count,
        "graph_edge_count": int(adjacency.sum() // 2),
        "isolated_workload_ratio": (
            float((~nonisolated).mean()) if workload_count else float("nan")
        ),
        "workloads_with_neighbors_ratio": (
            float(nonisolated.mean()) if workload_count else float("nan")
        ),
        "mean_degree": float(degrees.mean()) if workload_count else float("nan"),
        "degree_p50": float(np.quantile(degrees, 0.50)) if workload_count else float("nan"),
        "degree_p95": float(np.quantile(degrees, 0.95)) if workload_count else float("nan"),
        "connected_component_count": len(components),
        "largest_component_ratio": (
            float(max(components) / workload_count) if workload_count else float("nan")
        ),
        "propagation_hops": int(hops),
        "mean_r_hop_reachable_nodes": (
            float(reachable_counts.mean()) if workload_count else float("nan")
        ),
    }


def _unweighted_workload_adjacency(deployment: np.ndarray) -> np.ndarray:
    """Resolve node-workload incidence explicitly, including square incidence."""
    incidence = np.asarray(deployment, dtype=np.float32)
    if incidence.ndim != 2:
        raise DatasetError("deployment must be a node-workload incidence matrix")
    occupied = incidence > 0
    # Boolean co-location avoids invoking a platform BLAS merely to build an
    # unweighted graph (important on Windows when PyTorch owns the OMP runtime).
    adjacency = np.any(occupied[:, :, None] & occupied[:, None, :], axis=0)
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float32)


def _save_model_atomic(model: AlibabaOursModel, path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    model.save(temporary, metadata=metadata)
    os.replace(temporary, path)


def _save_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _shared_model_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "models/shared_preprocessing.npz": run_dir / "models" / "shared_preprocessing.npz",
        "models/shared_prototypes.npy": run_dir / "models" / "shared_prototypes.npy",
    }


def _trained_model_artifacts(model_path: Path) -> dict[str, Path]:
    return {
        "model.pt": model_path,
        "training_history.csv": model_path.parent / "training_history.csv",
    }


def _evaluation_artifacts(fragment: Path, origin_id: str) -> dict[str, Path]:
    parent = fragment.parent
    return {
        "predictions.parquet": fragment,
        "metrics.json": parent / f"metrics-{origin_id}.json",
        "metrics-all-h.parquet": parent / f"metrics-all-h-{origin_id}.parquet",
        "per-class.parquet": parent / f"per-class-{origin_id}.parquet",
        "calibration-bins.parquet": parent / f"calibration-bins-{origin_id}.parquet",
        "confusion.npz": parent / f"confusion-{origin_id}.npz",
        "diagnostics.parquet": parent / f"diagnostics-{origin_id}.parquet",
    }


def command_train(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_dir, identity = _bind_run(config, args)
    identity_digest = str(identity["identity_hash"])
    report, ok = _preflight_report(config)
    if not ok:
        _print_json(report)
        return EXIT_PREFLIGHT_FAILED
    mode_metadata = _mode_metadata(config)
    _write_run_metadata(config, run_dir, stage="train", formal_result=True)
    prepared_digest = _prepared_data_digest(run_dir, identity_digest)
    cache = _load_cache(run_dir)
    index = build_data_index(config)
    train_stop = index.split("train").stop_index
    validation_stop = index.split("validation").stop_index
    mean, std = _normalization(cache, train_stop)
    resources = _normalized_resources(cache, mean, std)
    active = cache["input_valid_mask"].astype(bool)
    times = np.arange(resources.shape[1], dtype=np.float32)
    deployment = cache["deployment"].astype(np.float32)
    if _link_mode(config) == LINK_MODE_TOPOLOGY_ONLY:
        # Deliberately do not resolve or read a link file. None selects neutral,
        # unweighted message passing over the observed deployment topology.
        model_topology = _unweighted_workload_adjacency(deployment)
        link_train = None
    else:
        model_topology = deployment
        link_rows = _load_link_rows(config)
        first_bucket = int(config.start_seconds // config.resample_seconds)
        train_cutoff_bucket = first_bucket + train_stop
        link_train = _link_matrix(
            link_rows[link_rows["time_bucket"] < train_cutoff_bucket], cache["node_ids"]
        )
    graph_coverage = _topology_coverage(
        model_topology, int(config.protocol["message_passing_steps"])
    )
    preprocessing_path = run_dir / "models" / "shared_preprocessing.npz"
    shared_path = run_dir / "models" / "shared_prototypes.npy"
    shared_guard_path = run_dir / "models" / "shared.complete.json"
    shared_relevant_digest = data_hash(
        {"prepared_data_hash": prepared_digest, "scope": "shared_preprocessing"}
    )
    if shared_path.exists() and preprocessing_path.exists() and args.resume:
        shared_guard = validate_artifact_guard(
            shared_guard_path,
            stage="train_shared",
            run_identity_digest=identity_digest,
            relevant_data_digest=shared_relevant_digest,
            artifacts=_shared_model_artifacts(run_dir),
        )
        shared_prototypes = np.load(shared_path, allow_pickle=False)
    else:
        atomic_write_npz(preprocessing_path, {"mean": mean, "std": std})
        base = AlibabaOursModel(_model_config(config, config.max_history, resources.shape[-1]))
        shared_prototypes = initialize_fixed_k_prototypes(
            base,
            torch.as_tensor(resources[:, :train_stop]),
            torch.as_tensor(active[:, :train_stop]),
            max_candidates=512,
            random_state=7,
        ).cpu().numpy()
        _save_npy_atomic(shared_path, shared_prototypes)
        shared_guard = write_artifact_guard(
            shared_guard_path,
            stage="train_shared",
            run_identity_digest=identity_digest,
            relevant_data_digest=shared_relevant_digest,
            artifacts=_shared_model_artifacts(run_dir),
        )
    shared_artifact_digest = str(shared_guard["artifact_hash"])

    tracker = _tracker(config, run_dir)
    try:
        tracker.start(total=len(config.history_lengths), stage="train", resume=args.resume)
    except ProgressStateError:
        tracker.start(total=len(config.history_lengths), stage="train", reset=True)
    for position, history_length in enumerate(config.history_lengths, start=1):
        model_path = run_dir / "models" / f"L={history_length}" / "model.pt"
        model_guard_path = model_path.parent / "complete.json"
        model_relevant_digest = data_hash(
            {
                "prepared_data_hash": prepared_digest,
                "shared_artifact_hash": shared_artifact_digest,
                "history_length": int(history_length),
            }
        )
        if args.resume and model_path.exists():
            validate_artifact_guard(
                model_guard_path,
                stage="train_model",
                run_identity_digest=identity_digest,
                relevant_data_digest=model_relevant_digest,
                artifacts=_trained_model_artifacts(model_path),
            )
            tracker.update(completed=position, stage="train", message=f"复用 L={history_length}")
            continue
        model = AlibabaOursModel(_model_config(config, history_length, resources.shape[-1]))
        model.shape_encoder.set_prototypes(shared_prototypes)
        checkpoint = model_path.with_name("best.pt")

        def epoch_callback(epoch: int, values: Mapping[str, float], _: AlibabaOursModel) -> None:
            tracker.update(
                stage="train",
                message=f"L={history_length}, epoch={epoch}",
                metrics={key: float(value) for key, value in values.items()},
                extra={"history_length": history_length, "epoch": epoch},
            )

        result = train_model(
            model,
            resources[:, :train_stop],
            times[:train_stop],
            active[:, :train_stop],
            model_topology,
            link_train,
            config=_training_config(
                config,
                checkpoint,
                resume=args.resume,
                run_identity_digest=identity_digest,
                relevant_data_digest=model_relevant_digest,
                run_id=run_dir.name,
            ),
            validation={
                "resources": resources[:, train_stop - history_length : validation_stop],
                "times": times[train_stop - history_length : validation_stop],
                "mask": active[:, train_stop - history_length : validation_stop],
                "topology": model_topology,
                "link_features": link_train,
            },
            epoch_callback=epoch_callback,
            initialize_prototypes=False,
        )
        _save_model_atomic(
            result.model,
            model_path,
            {
                "history_length": history_length,
                "seed": 7,
                "best_epoch": result.best_epoch,
                **mode_metadata,
                **graph_coverage,
            },
        )
        training_history = [
            {
                "seed": 7,
                **mode_metadata,
                "history_length": history_length,
                **dict(row),
            }
            for row in result.history
        ]
        atomic_write_csv(model_path.parent / "training_history.csv", training_history)
        write_artifact_guard(
            model_guard_path,
            stage="train_model",
            run_identity_digest=identity_digest,
            relevant_data_digest=model_relevant_digest,
            artifacts=_trained_model_artifacts(model_path),
            extra={"history_length": int(history_length)},
        )
        tracker.update(completed=position, stage="train", message=f"完成 L={history_length}")
    tracker.complete(summary={"models": len(config.history_lengths)})
    _write_run_metadata(
        config,
        run_dir,
        stage="train_complete",
        formal_result=True,
        graph_coverage=graph_coverage,
    )
    print(f"训练完成：{run_dir / 'models'}")
    return 0


def _true_memberships(
    model: AlibabaOursModel,
    resources: np.ndarray,
    active: np.ndarray,
    origin: OriginRecord,
    h_values: Iterable[int],
    workload_positions: np.ndarray,
) -> np.ndarray:
    values: list[np.ndarray] = []
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    with torch.no_grad():
        for h in h_values:
            target = origin.origin_index + int(h)
            start = max(0, target - model.config.history_window + 1)
            membership = model.encode_membership(
                torch.as_tensor(
                    resources[workload_positions, start : target + 1],
                    dtype=torch.float32,
                    device=model_device,
                ),
                torch.as_tensor(
                    active[workload_positions, start : target + 1],
                    dtype=torch.bool,
                    device=model_device,
                ),
            )
            values.append(membership.cpu().numpy())
    return np.stack(values, axis=1)


def _flatten_metrics(result: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in result.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_metrics(value, name + "."))
        elif np.isscalar(value) and not isinstance(value, (str, bytes)):
            flat[name] = value.item() if isinstance(value, np.generic) else value
    return flat


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_dir, identity = _bind_run(config, args)
    identity_digest = str(identity["identity_hash"])
    report, ok = _preflight_report(config)
    if not ok:
        _print_json(report)
        return EXIT_PREFLIGHT_FAILED
    mode_metadata = _mode_metadata(config)
    _write_run_metadata(config, run_dir, stage="evaluate", formal_result=True)
    prepared_digest = _prepared_data_digest(run_dir, identity_digest)
    cache = _load_cache(run_dir)
    shared_relevant_digest = data_hash(
        {"prepared_data_hash": prepared_digest, "scope": "shared_preprocessing"}
    )
    shared_guard = validate_artifact_guard(
        run_dir / "models" / "shared.complete.json",
        stage="train_shared",
        run_identity_digest=identity_digest,
        relevant_data_digest=shared_relevant_digest,
        artifacts=_shared_model_artifacts(run_dir),
    )
    shared_artifact_digest = str(shared_guard["artifact_hash"])
    with np.load(run_dir / "models" / "shared_preprocessing.npz", allow_pickle=False) as prep:
        mean, std = prep["mean"], prep["std"]
    resources = _normalized_resources(cache, mean, std)
    active = cache["input_valid_mask"].astype(bool)
    observed = cache["observed_values"].astype(np.float32)
    observed_mask = cache["target_observed_mask"].astype(bool)
    deployment = cache["deployment"].astype(np.float32)
    topology_only = _link_mode(config) == LINK_MODE_TOPOLOGY_ONLY
    workload_adjacency = (
        _unweighted_workload_adjacency(deployment) if topology_only else None
    )
    link_rows = (
        None
        if topology_only
        else _load_link_rows(config)
    )
    index = build_data_index(config)
    total = len(config.history_lengths) * len(index.origins)
    tracker = _tracker(config, run_dir)
    try:
        tracker.start(total=total, stage="evaluate", resume=args.resume)
    except ProgressStateError:
        tracker.start(total=total, stage="evaluate", reset=True)
    completed = 0
    for history_length in config.history_lengths:
        model_path = run_dir / "models" / f"L={history_length}" / "model.pt"
        model_relevant_digest = data_hash(
            {
                "prepared_data_hash": prepared_digest,
                "shared_artifact_hash": shared_artifact_digest,
                "history_length": int(history_length),
            }
        )
        model_guard = validate_artifact_guard(
            model_path.parent / "complete.json",
            stage="train_model",
            run_identity_digest=identity_digest,
            relevant_data_digest=model_relevant_digest,
            artifacts=_trained_model_artifacts(model_path),
        )
        model_artifact_digest = str(model_guard["artifact_hash"])
        model, _ = AlibabaOursModel.load(model_path)
        model.eval()
        for origin in index.origins:
            completed += 1
            fragment = (
                run_dir
                / "predictions"
                / f"L={history_length}"
                / f"split={origin.split}"
                / f"origin={origin.origin_id}.parquet"
            )
            metric_fragment = fragment.parent / f"metrics-{origin.origin_id}.json"
            all_h_metric_fragment = fragment.parent / f"metrics-all-h-{origin.origin_id}.parquet"
            per_class_fragment = fragment.parent / f"per-class-{origin.origin_id}.parquet"
            calibration_fragment = fragment.parent / f"calibration-bins-{origin.origin_id}.parquet"
            confusion_fragment = fragment.parent / f"confusion-{origin.origin_id}.npz"
            diagnostic_fragment = fragment.parent / f"diagnostics-{origin.origin_id}.parquet"
            completion_guard = fragment.parent / f"complete-{origin.origin_id}.json"
            evaluation_relevant_digest = data_hash(
                {
                    "prepared_data_hash": prepared_digest,
                    "model_artifact_hash": model_artifact_digest,
                    "origin_id": origin.origin_id,
                    "history_length": int(history_length),
                }
            )
            if (
                args.resume
                and fragment.exists()
                and metric_fragment.exists()
                and all_h_metric_fragment.exists()
                and per_class_fragment.exists()
                and calibration_fragment.exists()
                and confusion_fragment.exists()
                and diagnostic_fragment.exists()
            ):
                validate_artifact_guard(
                    completion_guard,
                    stage="evaluate_origin",
                    run_identity_digest=identity_digest,
                    relevant_data_digest=evaluation_relevant_digest,
                    artifacts=_evaluation_artifacts(fragment, origin.origin_id),
                )
                tracker.update(completed=completed, stage="evaluate", message=f"复用 {origin.origin_id}")
                continue
            eligible = _eligible_workloads_for_origin(
                config,
                active,
                observed_mask,
                origin,
            )
            positions = np.flatnonzero(eligible)
            if positions.size == 0:
                raise DatasetError(f"origin {origin.origin_id} 没有共同合格 workload")
            start = origin.origin_index - history_length + 1
            history = resources[positions, start : origin.origin_index + 1]
            history_mask = active[positions, start : origin.origin_index + 1]
            topology = (
                workload_adjacency[np.ix_(positions, positions)]
                if workload_adjacency is not None
                else deployment[:, positions]
            )
            graph_coverage = _topology_coverage(
                topology, int(config.protocol["message_passing_steps"])
            )
            if link_rows is None:
                links = None
            else:
                origin_bucket = int(cache["time_seconds"][origin.origin_index] // 60)
                links = _link_matrix(
                    link_rows,
                    cache["node_ids"],
                    time_bucket=origin_bucket,
                    latest_not_after=True,
                )
            recursive_checkpoint = (
                run_dir
                / "checkpoints"
                / "recursive"
                / f"L={history_length}"
                / f"split={origin.split}"
                / f"origin={origin.origin_id}.npz"
            )
            recursive_data_key = {
                "origin_id": origin.origin_id,
                "history_length": history_length,
                "workload_positions": positions,
                "link_mode": mode_metadata["link_mode"],
                "prepared_data_hash": prepared_digest,
                "model_artifact_hash": model_artifact_digest,
            }
            recursive_config_key = {
                "experiment": _raw_config(config),
                "run_identity_hash": identity_digest,
            }
            resume_state = None
            if args.resume and recursive_checkpoint.exists():
                loaded = load_recursive_checkpoint(
                    recursive_checkpoint,
                    expected_config=recursive_config_key,
                    expected_data=recursive_data_key,
                )
                resume_state = loaded.state["recursive"]

            def recursive_checkpoint_callback(h: int, state: Mapping[str, Any]) -> None:
                save_recursive_checkpoint(
                    recursive_checkpoint,
                    recursive_state=state,
                    forecast_horizon=h,
                    config=recursive_config_key,
                    data=recursive_data_key,
                    run_id=run_dir.name,
                    extra={
                        "history_length": history_length,
                        "split": origin.split,
                        "origin_id": origin.origin_id,
                        **mode_metadata,
                    },
                )
                tracker.update(
                    stage="evaluate",
                    forecast_horizon=h,
                    message=f"L={history_length}, {origin.origin_id}, h={h}",
                    extra={
                        "history_length": history_length,
                        "origin_id": origin.origin_id,
                    },
                )
            started = time.perf_counter()
            prediction = recursive_forecast(
                model,
                history,
                np.arange(start, origin.origin_index + 1, dtype=np.float32),
                config.forecast_horizons,
                history_mask,
                topology,
                links,
                device=str(config.training.get("device", "cpu")),
                future_times=np.arange(
                    origin.origin_index + 1,
                    origin.origin_index + config.max_forecast_horizon + 1,
                    dtype=np.float32,
                ),
                diagnostic_horizons=config.forecast_horizons,
                checkpoint_interval_h=int(config.runtime["recursive_checkpoint_interval_h"]),
                checkpoint_callback=recursive_checkpoint_callback,
                resume_state=resume_state,
            )
            elapsed = time.perf_counter() - started
            try:
                import psutil

                rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
            except ImportError:
                rss_mb = float("nan")
            throughput = positions.size * config.max_forecast_horizon / max(elapsed, 1e-12)
            mean_step_latency_ms = elapsed / config.max_forecast_horizon * 1000.0
            step_latency_ms = prediction.step_latency_seconds.numpy() * 1000.0
            latency_p50_ms, latency_p95_ms, latency_p99_ms = np.quantile(
                step_latency_ms, [0.50, 0.95, 0.99]
            )
            all_h_values = tuple(range(1, config.max_forecast_horizon + 1))
            true_mu = _true_memberships(
                model, resources, active, origin, all_h_values, positions
            )
            records: list[dict[str, Any]] = []
            metric_rows: list[dict[str, Any]] = []
            all_h_metric_rows: list[dict[str, Any]] = []
            per_class_rows: list[dict[str, Any]] = []
            calibration_rows: list[dict[str, Any]] = []
            confusion_arrays: dict[str, np.ndarray] = {}
            selected_h = set(config.forecast_horizons)
            for h_position, h in enumerate(all_h_values):
                target_index = origin.origin_index + h
                valid = observed_mask[positions, target_index]
                if not valid.any():
                    continue
                pred_mu = prediction.at_h(h).numpy()[valid]
                truth_mu = true_mu[:, h_position][valid]
                predicted_resources = model.shape_encoder.decode_resources(
                    torch.as_tensor(pred_mu)
                ).detach().cpu().numpy() * std + mean
                true_resources = observed[positions[valid], target_index]
                persistence_mu = prediction.initial_membership.numpy()[valid]
                persistence_resources = (
                    resources[positions[valid], origin.origin_index] * std + mean
                )
                resource_names = list(cache["resource_names"].astype(str))
                if {"disk_i", "disk_o"}.issubset(resource_names):
                    disk_i = resource_names.index("disk_i")
                    disk_o = resource_names.index("disk_o")
                    true_resources = np.column_stack(
                        [true_resources, true_resources[:, disk_i] + true_resources[:, disk_o]]
                    )
                    predicted_resources = np.column_stack(
                        [
                            predicted_resources,
                            predicted_resources[:, disk_i] + predicted_resources[:, disk_o],
                        ]
                    )
                    persistence_resources = np.column_stack(
                        [
                            persistence_resources,
                            persistence_resources[:, disk_i] + persistence_resources[:, disk_o],
                        ]
                    )
                    resource_names.append("disk_io_total")
                evaluated = evaluate_predictions(
                    truth_mu,
                    pred_mu,
                    labels=list(range(int(config.protocol["num_states"]))),
                    true_resources=true_resources,
                    predicted_resources=predicted_resources,
                    resource_names=resource_names,
                )
                persistence_evaluated = evaluate_predictions(
                    truth_mu,
                    persistence_mu,
                    labels=list(range(int(config.protocol["num_states"]))),
                    true_resources=true_resources,
                    predicted_resources=persistence_resources,
                    resource_names=resource_names,
                )
                row = {
                    "seed": 7,
                    **mode_metadata,
                    **graph_coverage,
                    "split": origin.split,
                    "history_length": history_length,
                    "h": h,
                    "origin_id": origin.origin_id,
                    "n_samples": int(valid.sum()),
                    "rollout_seconds": elapsed,
                    "workload_steps_per_second": throughput,
                    "mean_step_latency_ms": mean_step_latency_ms,
                    "step_latency_p50_ms": float(latency_p50_ms),
                    "step_latency_p95_ms": float(latency_p95_ms),
                    "step_latency_p99_ms": float(latency_p99_ms),
                    "rss_mb": rss_mb,
                    "metric_scope": "pointwise",
                }
                row.update(_flatten_metrics(evaluated))
                row.update(_flatten_metrics(persistence_evaluated, "baseline.persistence."))
                row["gain_vs_persistence.accuracy"] = float(
                    evaluated["classification"]["accuracy"]
                    - persistence_evaluated["classification"]["accuracy"]
                )
                row["gain_vs_persistence.f1_weighted"] = float(
                    evaluated["classification"]["f1_weighted"]
                    - persistence_evaluated["classification"]["f1_weighted"]
                )
                row["gain_vs_persistence.resource_mae_reduction"] = float(
                    persistence_evaluated["resources"]["overall"]["mae"]
                    - evaluated["resources"]["overall"]["mae"]
                )
                all_h_metric_rows.append(row)
                if h not in selected_h:
                    continue
                metric_rows.append(row)
                for class_row in evaluated["classification"].get("per_class", []):
                    per_class_rows.append(
                        {
                            "seed": 7,
                            **mode_metadata,
                            "split": origin.split,
                            "history_length": history_length,
                            "h": h,
                            "origin_id": origin.origin_id,
                            **class_row,
                        }
                    )
                for bin_row in evaluated["calibration"].get("bins", []):
                    calibration_rows.append(
                        {
                            "seed": 7,
                            **mode_metadata,
                            "split": origin.split,
                            "history_length": history_length,
                            "h": h,
                            "origin_id": origin.origin_id,
                            **bin_row,
                        }
                    )
                confusion_arrays[f"h_{h}"] = np.asarray(
                    evaluated["classification"]["confusion_matrix"], dtype=np.float64
                )
                for local, workload_position in enumerate(positions[valid]):
                    record: dict[str, Any] = {
                        "seed": 7,
                        **mode_metadata,
                        "split": origin.split,
                        "history_length": history_length,
                        "h": h,
                        "origin_id": origin.origin_id,
                        "origin_time": origin.origin_seconds,
                        "target_time": int(cache["time_seconds"][target_index]),
                        "workload_id": str(cache["workload_ids"][workload_position]),
                        "true_class": int(np.argmax(truth_mu[local])),
                        "pred_class": int(np.argmax(pred_mu[local])),
                        "confidence": float(np.max(pred_mu[local])),
                        "true_observed": True,
                    }
                    for state in range(pred_mu.shape[1]):
                        record[f"true_mu_{state}"] = float(truth_mu[local, state])
                        record[f"pred_mu_{state}"] = float(pred_mu[local, state])
                    records.append(record)
            atomic_write_parquet(fragment, records)
            atomic_write_json(metric_fragment, metric_rows)
            atomic_write_parquet(all_h_metric_fragment, all_h_metric_rows)
            atomic_write_parquet(per_class_fragment, per_class_rows)
            atomic_write_parquet(calibration_fragment, calibration_rows)
            atomic_write_npz(confusion_fragment, confusion_arrays)
            diagnostic_rows: list[dict[str, Any]] = []
            for diagnostic in prediction.diagnostics:
                h = int(diagnostic["h"])
                dynamic = np.asarray(diagnostic["dynamic"])
                backoff = np.asarray(diagnostic["backoff"])
                spatial = np.asarray(diagnostic["spatial_membership"])
                graph = diagnostic["graph"]
                backoff_details = diagnostic["backoff_details"]
                for local, workload_position in enumerate(positions):
                    item: dict[str, Any] = {
                        "seed": 7,
                        **mode_metadata,
                        **graph_coverage,
                        "split": origin.split,
                        "history_length": history_length,
                        "h": h,
                        "origin_id": origin.origin_id,
                        "workload_id": str(cache["workload_ids"][workload_position]),
                        "confidence": float(np.asarray(diagnostic["confidence"])[local]),
                        "entropy": float(np.asarray(diagnostic["entropy"])[local]),
                        "dynamic_backoff_l1": float(
                            np.asarray(diagnostic["dynamic_backoff_l1"])[local]
                        ),
                        "neighbor_count": int(np.asarray(graph["neighbour_count"])[local]),
                        "link_gate_mean": (
                            float("nan")
                            if mode_metadata["link_mode"] == LINK_MODE_TOPOLOGY_ONLY
                            else float(np.asarray(graph["link_gate_mean"])[local])
                        ),
                        "link_features_present": bool(
                            np.asarray(graph.get("link_features_present", False)).reshape(-1)[
                                local if np.asarray(graph.get("link_features_present", False)).size > 1 else 0
                            ]
                        ),
                        "unweighted_topology": bool(
                            np.asarray(graph.get("unweighted_topology", True)).reshape(-1)[
                                local if np.asarray(graph.get("unweighted_topology", True)).size > 1 else 0
                            ]
                        ),
                        "spatial_modulation_l1": float(
                            np.asarray(graph["modulation_l1"])[local]
                        ),
                        "message_context_norm": float(
                            np.asarray(graph["message_context_norm"])[local]
                        ),
                        "effective_order": int(
                            np.asarray(backoff_details["effective_order"])[local]
                        ),
                    }
                    for state in range(dynamic.shape[1]):
                        item[f"dynamic_mu_{state}"] = float(dynamic[local, state])
                        item[f"backoff_mu_{state}"] = float(backoff[local, state])
                        item[f"spatial_mu_{state}"] = float(spatial[local, state])
                    for order in range(int(config.protocol["max_order"]) + 1):
                        item[f"backoff_support_{order}"] = float(
                            np.asarray(backoff_details["supports"])[local, order]
                        )
                        item[f"backoff_gate_{order}"] = float(
                            np.asarray(backoff_details["gates"])[local, order]
                        )
                        item[f"backoff_weight_{order}"] = float(
                            np.asarray(backoff_details["order_weights"])[local, order]
                        )
                    diagnostic_rows.append(item)
            atomic_write_parquet(diagnostic_fragment, diagnostic_rows)
            write_artifact_guard(
                completion_guard,
                stage="evaluate_origin",
                run_identity_digest=identity_digest,
                relevant_data_digest=evaluation_relevant_digest,
                artifacts=_evaluation_artifacts(fragment, origin.origin_id),
                extra={
                    "history_length": int(history_length),
                    "split": origin.split,
                    "origin_id": origin.origin_id,
                },
            )
            tracker.update(
                completed=completed,
                stage="evaluate",
                forecast_horizon=config.max_forecast_horizon,
                message=f"L={history_length}, {origin.origin_id}",
                extra={"history_length": history_length, "origin_id": origin.origin_id},
            )
    tracker.complete(summary={"evaluation_units": total})
    _write_run_metadata(config, run_dir, stage="evaluate_complete", formal_result=True)
    return command_aggregate(args)


def command_aggregate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_dir, identity = _bind_run(config, args)
    identity_digest = str(identity["identity_hash"])
    prepared_digest = _prepared_data_digest(run_dir, identity_digest)
    shared_relevant_digest = data_hash(
        {"prepared_data_hash": prepared_digest, "scope": "shared_preprocessing"}
    )
    shared_guard = validate_artifact_guard(
        run_dir / "models" / "shared.complete.json",
        stage="train_shared",
        run_identity_digest=identity_digest,
        relevant_data_digest=shared_relevant_digest,
        artifacts=_shared_model_artifacts(run_dir),
    )
    shared_artifact_digest = str(shared_guard["artifact_hash"])
    _write_run_metadata(config, run_dir, stage="aggregate", formal_result=True)
    metric_paths = sorted((run_dir / "predictions").glob("L=*/split=*/metrics-*.json"))
    rows: list[dict[str, Any]] = []
    for path in metric_paths:
        origin_id = path.stem[len("metrics-") :]
        history_length = int(path.parents[1].name.split("=", 1)[1])
        model_path = run_dir / "models" / f"L={history_length}" / "model.pt"
        model_relevant_digest = data_hash(
            {
                "prepared_data_hash": prepared_digest,
                "shared_artifact_hash": shared_artifact_digest,
                "history_length": history_length,
            }
        )
        model_guard = validate_artifact_guard(
            model_path.parent / "complete.json",
            stage="train_model",
            run_identity_digest=identity_digest,
            relevant_data_digest=model_relevant_digest,
            artifacts=_trained_model_artifacts(model_path),
        )
        fragment = path.parent / f"origin={origin_id}.parquet"
        evaluation_relevant_digest = data_hash(
            {
                "prepared_data_hash": prepared_digest,
                "model_artifact_hash": str(model_guard["artifact_hash"]),
                "origin_id": origin_id,
                "history_length": history_length,
            }
        )
        validate_artifact_guard(
            path.parent / f"complete-{origin_id}.json",
            stage="evaluate_origin",
            run_identity_digest=identity_digest,
            relevant_data_digest=evaluation_relevant_digest,
            artifacts=_evaluation_artifacts(fragment, origin_id),
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(value if isinstance(value, list) else [value])
    if not rows:
        raise FileNotFoundError("没有可汇总的metrics分片；请先执行 evaluate")
    metrics_dir = run_dir / "metrics"
    atomic_write_csv(metrics_dir / "metrics_by_origin.csv", rows)
    atomic_write_parquet(metrics_dir / "metrics_by_origin.parquet", rows)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        mode_keys = [
            key
            for key in ("link_mode", "real_link_features", "full_ours", "experiment_variant")
            if key in frame.columns
        ]
        group_keys = ["split", "history_length", "h", *mode_keys]
        numeric = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in set(group_keys)
        ]
        grouped = frame.groupby(group_keys, as_index=False)[numeric].mean()
        atomic_write_csv(metrics_dir / "metrics_by_h.csv", grouped.to_dict(orient="records"))
        atomic_write_parquet(metrics_dir / "metrics_by_h.parquet", grouped)
        all_h_paths = sorted(
            (run_dir / "predictions").glob("L=*/split=*/metrics-all-h-*.parquet")
        )
        if all_h_paths:
            all_h_frame = pd.concat(
                [pd.read_parquet(path) for path in all_h_paths], ignore_index=True
            )
            atomic_write_parquet(metrics_dir / "metrics_all_h_by_origin.parquet", all_h_frame)
            all_h_numeric = [
                column
                for column in all_h_frame.select_dtypes(include=[np.number]).columns
                if column not in set(group_keys)
            ]
            all_h_grouped = all_h_frame.groupby(group_keys, as_index=False)[all_h_numeric].mean()
            atomic_write_csv(
                metrics_dir / "metrics_all_h.csv",
                all_h_grouped.to_dict(orient="records"),
            )
            atomic_write_parquet(metrics_dir / "metrics_all_h.parquet", all_h_grouped)
        else:
            all_h_grouped = grouped
        degradation_rows: list[dict[str, Any]] = []
        degradation_group_keys = ["split", "history_length", *mode_keys]
        for group_value, part in all_h_grouped.groupby(degradation_group_keys):
            if not isinstance(group_value, tuple):
                group_value = (group_value,)
            group_metadata = dict(zip(degradation_group_keys, group_value))
            for item in horizon_degradation(part.to_dict(orient="records"), reference_h=1):
                item.update(group_metadata)
                item["history_length"] = int(item["history_length"])
                degradation_rows.append(item)
        if degradation_rows:
            atomic_write_csv(metrics_dir / "horizon_degradation.csv", degradation_rows)
            atomic_write_parquet(metrics_dir / "horizon_degradation.parquet", degradation_rows)
        for pattern, stem in (
            ("L=*/split=*/per-class-*.parquet", "per_class"),
            ("L=*/split=*/calibration-bins-*.parquet", "calibration_bins"),
            ("L=*/split=*/diagnostics-*.parquet", "recursive_diagnostics"),
        ):
            paths = sorted((run_dir / "predictions").glob(pattern))
            if paths:
                combined = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
                atomic_write_csv(
                    metrics_dir / f"{stem}.csv", combined.to_dict(orient="records")
                )
                atomic_write_parquet(metrics_dir / f"{stem}.parquet", combined)
    except ImportError:
        pass
    _write_run_metadata(config, run_dir, stage="aggregate_complete", formal_result=True)
    print(f"汇总完成：{metrics_dir}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    status = read_status(run_dir / "progress" / "status.json")
    if status is None:
        print(f"没有状态文件：{run_dir}", file=sys.stderr)
        return 1
    _print_json(status)
    return 0


def _synthetic_links(nodes: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((nodes, nodes, 4), dtype=np.float32)
    values[..., 0] = rng.uniform(5.0, 20.0, size=(nodes, nodes))
    values[..., 1] = rng.uniform(0.5, 3.0, size=(nodes, nodes))
    values[..., 2] = rng.uniform(0.0, 0.05, size=(nodes, nodes))
    values[..., 3] = rng.uniform(0.9, 1.0, size=(nodes, nodes))
    return values


def command_smoke_test(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    mode_metadata = _mode_metadata(config)
    run_dir, _ = _bind_run(config, args)
    _write_run_metadata(config, run_dir, stage="smoke_test", formal_result=False)
    output = run_dir / "smoke-test"
    if output.exists() and not args.resume:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    workloads, total, resources_count = 5, 52, 4
    base = np.linspace(0.05, 0.95, total, dtype=np.float32)
    resources = np.stack(
        [
            np.stack(
                [
                    np.clip(base + 0.08 * np.sin(np.arange(total) / (3 + dim)) + 0.02 * rng.normal(size=total), 0, 1)
                    for dim in range(resources_count)
                ],
                axis=-1,
            )
            for _ in range(workloads)
        ],
        axis=0,
    ).astype(np.float32)
    mask = np.ones((workloads, total), dtype=bool)
    deployment = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=np.float32)
    smoke_topology = (
        _unweighted_workload_adjacency(deployment)
        if mode_metadata["link_mode"] == LINK_MODE_TOPOLOGY_ONLY
        else deployment
    )
    links = (
        None
        if mode_metadata["link_mode"] == LINK_MODE_TOPOLOGY_ONLY
        else _synthetic_links(2)
    )
    times = np.arange(total, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for history_length in (5, 15):
        model_config = _model_config(config, history_length, resources_count)
        model_config.prototype_length = 5
        model = AlibabaOursModel(model_config)
        training_config = TrainingConfig(
            epochs=1,
            learning_rate=3e-3,
            max_origins_per_epoch=3,
            early_stopping_patience=1,
            prototype_candidates=20,
            device="cpu",
            random_state=7,
        )
        result = train_model(
            model,
            resources[:, :35],
            times[:35],
            mask[:, :35],
            smoke_topology,
            links,
            config=training_config,
        )
        forecast = recursive_forecast(
            result.model,
            resources[:, 20:35],
            times[20:35],
            [1, 5, 15, 17],
            mask[:, 20:35],
            smoke_topology,
            links,
        )
        for h in (1, 5, 15, 17):
            prediction = forecast.at_h(h).numpy()
            target_resource = resources[:, 34 + h]
            target = result.model.encode_membership(
                torch.as_tensor(resources[:, max(0, 35 + h - history_length) : 35 + h]),
                torch.as_tensor(mask[:, max(0, 35 + h - history_length) : 35 + h]),
            ).detach().cpu().numpy()
            decoded = result.model.shape_encoder.decode_resources(torch.as_tensor(prediction)).detach().cpu().numpy()
            evaluated = evaluate_predictions(
                target,
                prediction,
                labels=list(range(8)),
                true_resources=np.column_stack(
                    [target_resource, target_resource[:, 2] + target_resource[:, 3]]
                ),
                predicted_resources=np.column_stack(
                    [decoded, decoded[:, 2] + decoded[:, 3]]
                ),
                resource_names=["cpu", "mem", "disk_i", "disk_o", "disk_io_total"],
            )
            row = {
                "seed": 7,
                **mode_metadata,
                "history_length": history_length,
                "h": h,
            }
            row.update(_flatten_metrics(evaluated))
            rows.append(row)
        _save_model_atomic(
            result.model,
            output / f"L={history_length}" / "model.pt",
            {"smoke": True, **mode_metadata},
        )
    atomic_write_csv(output / "metrics.csv", rows)
    atomic_write_parquet(output / "metrics.parquet", rows)
    atomic_write_json(
        output / "result.json",
        {
            "ok": True,
            "seed": 7,
            **mode_metadata,
            "history_lengths": [5, 15],
            "forecast_horizons": [1, 5, 15, 17],
            "uses_synthetic_links": links is not None,
            "formal_result": False,
        },
    )
    _write_run_metadata(config, run_dir, stage="smoke_test_complete", formal_result=False)
    print(f"冒烟测试通过：{output}")
    return 0


def command_run_all(args: argparse.Namespace) -> int:
    for function in (command_preflight, command_prepare, command_train, command_evaluate):
        code = function(args)
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alibaba Ours L×h recursive experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare", "train", "evaluate", "aggregate", "run-all", "smoke-test"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/experiment.json")
        child.add_argument("--run-id", default=None)
        child.add_argument("--resume", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "preflight": command_preflight,
        "prepare": command_prepare,
        "train": command_train,
        "evaluate": command_evaluate,
        "aggregate": command_aggregate,
        "run-all": command_run_all,
        "status": command_status,
        "smoke-test": command_smoke_test,
    }
    try:
        return int(commands[args.command](args))
    except KeyboardInterrupt:
        print("已中断；使用相同命令加 --resume 继续。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
