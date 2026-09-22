from __future__ import annotations

"""Portable CLI orchestrator for the K/P/R/lambda/tau sensitivity suite."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import alibaba_ours_exp.cli as base_cli
import alibaba_ours_exp.config as base_config
from alibaba_ours_exp.checkpoint import make_run_identity
from alibaba_ours_exp.progress import ProgressTracker

from .collector import collect_suite, write_csv_atomic, write_json_atomic
from .design import Condition, STUDIES, build_conditions, condition_rows, select_condition
from .enrichment import enrich_run
from .progress import SensitivityProgressTracker
from .protocol import apply_single_bucket_protocol, protocol_manifest
from .resample import ensure_resampled_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "sensitivity.json"
BASE_STAGES = ("preflight", "prepare", "train", "evaluate", "aggregate")
BASE_COMMANDS = (*BASE_STAGES, "run-all", "smoke-test")
VENDOR_ROOT = ROOT / "vendor"

_BASE_IDENTITY_INPUT_PATHS = base_cli._identity_input_paths
_BASE_TRAINING_CONFIG = base_cli._training_config
_ORIGINAL_VALIDATE = base_config.ExperimentConfig.validate
_ORIGINAL_TORCH_SAVE = torch.save
_ORIGINAL_TORCH_LOAD = torch.load
_BASE_FORECAST = base_cli.recursive_forecast


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_safe_torch_save(
    value: Any,
    destination: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if isinstance(destination, (str, os.PathLike)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            return _ORIGINAL_TORCH_SAVE(value, handle, *args, **kwargs)
    return _ORIGINAL_TORCH_SAVE(value, destination, *args, **kwargs)


def _windows_safe_torch_load(source: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(source, (str, os.PathLike)):
        with Path(source).open("rb") as handle:
            return _ORIGINAL_TORCH_LOAD(handle, *args, **kwargs)
    return _ORIGINAL_TORCH_LOAD(source, *args, **kwargs)


def _sensitivity_validate(self: base_config.ExperimentConfig) -> None:
    """Reuse base validation while allowing the five intentional sensitivities."""

    baseline_protocol = dict(self.protocol)
    baseline_protocol.update(
        {
            "seed": 7,
            "num_states": 8,
            "max_order": 3,
            "time_decay": 0.02,
            "message_passing_steps": 2,
            "time_period_steps": 1440,
        }
    )
    _ORIGINAL_VALIDATE(
        replace(self, resample_seconds=60, protocol=baseline_protocol)
    )
    allowed_tau = {60, 1800, 3600, 7200}
    if self.resample_seconds not in allowed_tau:
        raise base_config.ConfigError(f"tau must be one of {sorted(allowed_tau)}")
    if (self.end_seconds_exclusive - self.start_seconds) % self.resample_seconds:
        raise base_config.ConfigError("physical time span must be divisible by tau")
    if len(self.history_lengths) != 1:
        raise base_config.ConfigError("each condition must have exactly one history length")
    if self.history_lengths[0] * self.resample_seconds != 86_400:
        raise base_config.ConfigError("every condition must use exactly 24 hours of history")
    if self.forecast_horizons != (1,):
        raise base_config.ConfigError("the suite predicts exactly the next time bucket")
    if self.protocol["forecast_strategy"] != "direct_multi_step":
        raise base_config.ConfigError("h=1 must be recorded as direct prediction without feedback")
    if int(self.protocol["seed"]) < 0:
        raise base_config.ConfigError("the experiment seed must be non-negative")
    if int(self.protocol["num_states"]) <= 1:
        raise base_config.ConfigError("K must be greater than one")
    order = int(self.protocol["max_order"])
    if order <= 0 or order > self.history_lengths[0]:
        raise base_config.ConfigError("P must be in [1, history_length]")
    if int(self.protocol["message_passing_steps"]) <= 0:
        raise base_config.ConfigError("R must be positive")
    decay = float(self.protocol["time_decay"])
    if not math.isfinite(decay) or decay < 0.0:
        raise base_config.ConfigError("effective per-step lambda must be finite and non-negative")
    if int(self.protocol["time_period_steps"]) * self.resample_seconds != 86_400:
        raise base_config.ConfigError("time_period_steps must represent one physical day")


def _source_dataset(config_path: Path) -> Path:
    raw = _read_json(config_path)
    project_root = (config_path.parent / Path(str(raw.get("project_root", ".")))).resolve()
    return (project_root / Path(str(raw["portable_dataset_path"]))).resolve()


def _condition_dataset(
    config_path: Path,
    condition: Condition,
    *,
    resume: bool,
) -> Path:
    raw = _read_json(config_path)
    missing = raw.get("missing", {})
    suite = raw.get("sensitivity_suite", {})
    return ensure_resampled_dataset(
        _source_dataset(config_path),
        ROOT / "inputs" / "resampled",
        condition,
        split_ratios=(
            float(raw["split"]["train_ratio"]),
            float(raw["split"]["validation_ratio"]),
            float(raw["split"]["test_ratio"]),
        ),
        minimum_bucket_coverage=float(
            suite.get("minimum_resample_bucket_coverage", 0.80)
        ),
        minimum_history_coverage=float(missing.get("min_history_coverage", 0.80)),
        progress_root=ROOT / "runs" / "_input_progress",
        resume=resume,
    )


def materialize_condition_config(
    base_path: str | Path,
    condition: Condition,
    dataset_path: str | Path,
    *,
    device_override: str | None = None,
) -> Path:
    path = Path(base_path).resolve()
    raw = _read_json(path)
    raw["portable_dataset_path"] = str(Path(dataset_path).resolve().relative_to(ROOT)).replace(
        "\\", "/"
    )
    raw["time_axis"] = dict(raw["time_axis"])
    raw["time_axis"]["resample_seconds"] = condition.granularity_seconds
    raw["protocol"] = dict(raw["protocol"])
    raw["protocol"].update(
        {
            "experiment_name": f"Alibaba Ours KPR-lambda-tau {condition.condition_id}",
            "forecast_strategy": "direct_multi_step",
            "seed": condition.seed,
            "num_states": condition.num_states,
            "max_order": condition.max_order,
            "time_decay": condition.effective_time_decay_per_step,
            "time_decay_per_minute": condition.time_decay_per_minute,
            "message_passing_steps": condition.message_passing_steps,
            "time_period_steps": condition.time_period_steps,
            "membership_definition": "single_time_bucket_resource_vector",
            "prediction_feedback": False,
        }
    )
    raw["grid"] = {
        "history_lengths": [condition.history_length],
        "forecast_horizons": [condition.forecast_horizon],
    }
    raw["origins"] = dict(raw["origins"])
    raw["origins"]["stride_steps"] = condition.origin_stride_steps
    raw["sensitivity"] = {
        **condition.metadata(),
        "baseline_membership": list(STUDIES) if condition.is_shared_baseline else [],
        **protocol_manifest(),
    }
    if device_override is not None:
        if device_override not in {"cpu", "cuda"}:
            raise ValueError("device override must be cpu or cuda")
        raw["training"] = dict(raw["training"])
        raw["training"]["device"] = device_override
    generated = (
        ROOT
        / "generated"
        / f"{condition.condition_id}-seed{condition.seed}.json"
    )
    write_json_atomic(generated, raw)
    return generated


def _sensitivity_training_config(
    config: base_config.ExperimentConfig,
    checkpoint_path: Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train-setup] checkpoint_dir_ready={checkpoint_path.parent}", flush=True)
    return _BASE_TRAINING_CONFIG(config, checkpoint_path, *args, **kwargs)


def _patch_base_cli(condition: Condition) -> None:
    base_config.ExperimentConfig.validate = _sensitivity_validate
    torch.save = _windows_safe_torch_save
    torch.load = _windows_safe_torch_load
    apply_single_bucket_protocol()

    def direct_one_step_forecast(
        model: Any,
        history_resources: Any,
        history_times: Any,
        forecast_horizon: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        horizons = (
            (int(forecast_horizon),)
            if isinstance(forecast_horizon, (int, np.integer))
            else tuple(sorted({int(value) for value in forecast_horizon}))
        )
        if horizons != (1,):
            raise ValueError(
                "this sensitivity suite permits only direct next-bucket h=1 prediction"
            )
        result = _BASE_FORECAST(
            model,
            history_resources,
            history_times,
            horizons,
            *args,
            **kwargs,
        )
        result.metadata.update(
            {
                "feedback": "none_h1",
                "predicted_outputs_reused": False,
                "forecast_kernel": "single_step_from_observed_history",
                "predicted_backoff_update_effect": "post_output_discarded",
            }
        )
        return result

    def mode_metadata(config: base_config.ExperimentConfig) -> dict[str, Any]:
        raw = base_cli._raw_config(config)
        sensitivity = raw.get("sensitivity", condition.metadata())
        return {
            "link_mode": base_cli._link_mode(config),
            "link_features_required": False,
            "real_link_features": False,
            "full_ours": False,
            "experiment_variant": "Ours-TopologyOnly-KPRLambdaTau-Sensitivity",
            **protocol_manifest(),
            **dict(sensitivity),
        }

    def identity_input_paths(config: base_config.ExperimentConfig) -> dict[str, Path]:
        paths = _BASE_IDENTITY_INPUT_PATHS(config)
        for source in sorted((VENDOR_ROOT / "alibaba_ours_exp").glob("*.py")):
            paths[f"vendor_base:{source.name}"] = source
        for source in sorted((VENDOR_ROOT / "workload_fmm").glob("*.py")):
            paths[f"vendor_method:{source.name}"] = source
        for source in sorted((ROOT / "src" / "alibaba_ours_minganxing").glob("*.py")):
            paths[f"sensitivity_code:{source.name}"] = source
        paths["sensitivity:base_config"] = DEFAULT_CONFIG
        return paths

    def current_run_identity(config: base_config.ExperimentConfig) -> dict[str, Any]:
        return make_run_identity(
            raw_config=base_cli._raw_config(config),
            experiment_root=ROOT,
            project_root=config.project_root,
            input_paths=identity_input_paths(config),
        )

    def run_id(
        config: base_config.ExperimentConfig,
        identity: Mapping[str, Any] | None = None,
    ) -> str:
        current = dict(identity or current_run_identity(config))
        digest = str(current["identity_hash"])[:12]
        return (
            f"alibaba-minganxing-{condition.condition_id}"
            f"-seed{condition.seed}-{digest}"
        )

    def tracker(
        config: base_config.ExperimentConfig,
        run_dir: Path,
    ) -> SensitivityProgressTracker:
        return SensitivityProgressTracker(
            run_dir / "progress",
            run_id=run_dir.name,
            training_epochs=int(config.training["epochs"]),
            heartbeat_interval_seconds=float(
                config.runtime.get("progress_interval_seconds", 30)
            ),
        )

    base_cli._mode_metadata = mode_metadata
    base_cli._identity_input_paths = identity_input_paths
    base_cli._current_run_identity = current_run_identity
    base_cli._run_id = run_id
    base_cli._tracker = tracker
    base_cli._training_config = _sensitivity_training_config
    base_cli.recursive_forecast = direct_one_step_forecast


def _load_generated(
    base_path: Path,
    condition: Condition,
    *,
    device_override: str | None = None,
    resume: bool = True,
) -> tuple[Path, base_config.ExperimentConfig, Path]:
    dataset = _condition_dataset(base_path, condition, resume=resume)
    generated = materialize_condition_config(
        base_path,
        condition,
        dataset,
        device_override=device_override,
    )
    _patch_base_cli(condition)
    config = base_config.load_experiment_config(generated)
    identity = base_cli._current_run_identity(config)
    run_dir = config.output_dir / base_cli._run_id(config, identity)
    return generated, config, run_dir


def _run_base_stage(stage: str, generated: Path, *, resume: bool) -> int:
    if stage not in BASE_COMMANDS:
        raise ValueError(f"unsupported base stage {stage}")
    arguments = [stage, "--config", str(generated)]
    if resume:
        arguments.append("--resume")
    return int(base_cli.main(arguments))


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
        return {
            "available": True,
            "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def environment_manifest() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "captured_at": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": None if properties is None else int(properties.total_memory),
        "gpu_compute_capability": (
            None if properties is None else [int(properties.major), int(properties.minor)]
        ),
        "p2200_match": bool(gpu_name and "P2200" in gpu_name.upper()),
        "nvidia_smi": _nvidia_smi(),
    }


def _model_statistics(config: base_config.ExperimentConfig) -> dict[str, Any]:
    model_config = base_cli._model_config(
        config, config.history_lengths[0], len(config.resource_names)
    )
    model = base_cli.AlibabaOursModel(model_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {
        "trainable_parameter_count": int(trainable),
        "parameter_count": int(total),
        "buffer_element_count": int(buffers),
    }


def _artifact_statistics(run_dir: Path) -> dict[str, Any]:
    paths = [path for path in run_dir.glob("**/*") if path.is_file()]
    return {
        "artifact_file_count": len(paths),
        "artifact_bytes": sum(path.stat().st_size for path in paths),
        "checkpoint_count": len(list((run_dir / "models").glob("**/*.pt"))),
        "prediction_parquet_count": len(list((run_dir / "predictions").glob("**/*.parquet"))),
    }


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0**2)
    except (ImportError, OSError):
        return None


def run_condition(
    base_path: Path,
    condition: Condition,
    *,
    resume: bool,
    device_override: str | None = None,
) -> int:
    generated, config, run_dir = _load_generated(
        base_path,
        condition,
        device_override=device_override,
        resume=resume,
    )
    device = str(config.training.get("device", "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requires CUDA, but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started_at = _utc_now()
    started = time.perf_counter()
    status = "running"
    error: str | None = None
    exit_code = 1
    try:
        exit_code = _run_base_stage("run-all", generated, resume=resume)
        if exit_code == 0:
            exit_code = _run_base_stage("aggregate", generated, resume=True)
        if exit_code == 0:
            enrich_run(config, run_dir, condition, resume=resume)
        status = "complete" if exit_code == 0 else "failed"
        return exit_code
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        diagnostics: dict[str, Any] = {}
        for name, function in (
            ("model_statistics", lambda: _model_statistics(config)),
            ("artifact_statistics", lambda: _artifact_statistics(run_dir)),
        ):
            try:
                diagnostics.update(function())
            except Exception as exc:
                diagnostics[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        runtime = {
            **condition.metadata(),
            **protocol_manifest(),
            "status": status,
            "error": error,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "process_rss_mb": _process_rss_mb(),
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "cuda_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
            "generated_config": str(generated),
            **diagnostics,
            "environment": environment_manifest(),
        }
        write_json_atomic(run_dir / "condition_runtime.json", runtime)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def command_plan(config: Path, study: str) -> int:
    studies = None if study == "all" else (study,)
    conditions = build_conditions(config, studies=studies)
    rows = condition_rows(conditions)
    for order, row in enumerate(rows, start=1):
        row["execution_order"] = order
        row["device"] = "cuda"
        row["target_gpu"] = "NVIDIA Quadro P2200"
        row["command"] = (
            "python run_experiment.py run-condition "
            f"--config configs/{config.name} "
            f"--condition {row['sensitivity_condition_id']} --resume"
        )
    write_csv_atomic(ROOT / "runs" / f"experiment_plan_{study}.csv", rows)
    write_json_atomic(ROOT / "runs" / f"experiment_plan_{study}.json", rows)
    print(json.dumps({"study": study, "conditions": len(rows), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


def command_preflight(config: Path) -> int:
    conditions = build_conditions(config)
    reports: list[dict[str, Any]] = []
    for condition in conditions:
        generated, loaded, run_dir = _load_generated(
            config, condition, device_override="cpu", resume=True
        )
        report, ok = base_cli._preflight_report(loaded)
        reports.append(
            {
                "condition_id": condition.condition_id,
                "ok": bool(ok),
                "generated_config": str(generated),
                "run_dir": str(run_dir),
                "report": report,
            }
        )
    result = {
        "ok": all(item["ok"] for item in reports),
        "condition_count": len(conditions),
        "unique_condition_count_expected": 20,
        "portable_dataset": str(_source_dataset(config)),
        "portable_dataset_sha256": _sha256(_source_dataset(config)),
        "protocol": protocol_manifest(),
        "environment": environment_manifest(),
        "conditions": reports,
    }
    write_json_atomic(ROOT / "reports" / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] and len(conditions) == 20 else 2


def command_stage(
    config: Path,
    condition: Condition,
    stage: str,
    *,
    resume: bool,
    device: str | None,
) -> int:
    generated, _, _ = _load_generated(
        config, condition, device_override=device, resume=resume
    )
    if stage == "smoke-test":
        # The vendored smoke test intentionally exercises h={1,5,15,17}; keep
        # that broad kernel self-test separate from the formal h=1 guard.
        base_cli.recursive_forecast = _BASE_FORECAST
    return _run_base_stage(stage, generated, resume=resume)


def command_run_study(
    config: Path,
    study: str,
    *,
    resume: bool,
    device: str | None,
    require_p2200: bool,
) -> int:
    if require_p2200:
        manifest = environment_manifest()
        if not manifest["p2200_match"]:
            raise RuntimeError(
                "run-p2200 requires an NVIDIA Quadro P2200; "
                f"PyTorch reports {manifest.get('gpu_name') or 'no CUDA GPU'}"
            )
        write_json_atomic(ROOT / "runs" / "p2200_environment.json", manifest)
    studies = None if study == "all" else (study,)
    conditions = build_conditions(config, studies=studies)
    completed_ids: set[str] = set()
    for condition in conditions:
        _, _, run_dir = _load_generated(
            config,
            condition,
            device_override=device,
            resume=True,
        )
        metadata_path = run_dir / "run_metadata.json"
        runtime_path = run_dir / "condition_runtime.json"
        if metadata_path.is_file() and runtime_path.is_file():
            try:
                metadata = _read_json(metadata_path)
                runtime = _read_json(runtime_path)
                enrichment = run_dir / "metrics" / "enrichment.complete.json"
                if (
                    metadata.get("stage") == "aggregate_complete"
                    and runtime.get("status") == "complete"
                    and enrichment.is_file()
                ):
                    enriched = _read_json(enrichment)
                    if enriched.get("status") != "complete":
                        continue
                    completed_ids.add(condition.condition_id)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    suite_tracker = ProgressTracker(
        ROOT / "runs" / "_suite_progress" / study,
        run_id=f"minganxing-suite-{study}",
    )
    suite_tracker.start(
        total=len(conditions),
        stage=f"suite-{study}",
        message="starting K/P/R/lambda/tau sensitivity suite",
        reset=True,
    )
    completed = len({item.condition_id for item in conditions}.intersection(completed_ids))
    if completed:
        suite_tracker.update(
            completed=completed,
            message=f"resumed={completed} conditions",
            extra={"rebase_rate": True},
        )
    try:
        for index, condition in enumerate(conditions, start=1):
            if condition.condition_id in completed_ids:
                continue
            suite_tracker.update(
                completed=completed,
                message=(
                    f"condition={index}/{len(conditions)}:{condition.condition_id} "
                    f"K={condition.num_states} P={condition.max_order} "
                    f"R={condition.message_passing_steps} "
                    f"lambda={condition.time_decay_per_minute:g}/min "
                    f"tau={condition.granularity_seconds}s "
                    f"L={condition.history_length} h=1"
                ),
            )
            exit_code = run_condition(
                config,
                condition,
                resume=resume,
                device_override=device,
            )
            if exit_code != 0:
                suite_tracker.fail(
                    f"condition {condition.condition_id} exited with code {exit_code}"
                )
                return exit_code
            completed += 1
            suite_tracker.update(
                completed=completed,
                message=f"completed={condition.condition_id}",
            )
        suite_tracker.complete(
            summary={"conditions": len(conditions), "study": study},
            message="all requested conditions complete",
        )
        return 0
    except BaseException as exc:
        suite_tracker.fail(exc, context={"completed_conditions": completed})
        raise


def command_status(config: Path) -> int:
    conditions = build_conditions(config)
    expected = {item.condition_id for item in conditions}
    expected_seed = conditions[0].seed
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((ROOT / "runs").glob("alibaba-minganxing-*")):
        metadata_path = run_dir / "run_metadata.json"
        runtime_path = run_dir / "condition_runtime.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        runtime = _read_json(runtime_path) if runtime_path.is_file() else {}
        run_seed = metadata.get("sensitivity_seed", runtime.get("sensitivity_seed"))
        if run_seed is None or int(run_seed) != expected_seed:
            continue
        condition_id = str(metadata.get("sensitivity_condition_id") or runtime.get("sensitivity_condition_id") or "")
        if condition_id:
            rows.append(
                {
                    "condition_id": condition_id,
                    "run_id": run_dir.name,
                    "stage": metadata.get("stage"),
                    "status": runtime.get("status"),
                    "enrichment_complete": bool(
                        (run_dir / "metrics" / "enrichment.complete.json").is_file()
                    ),
                    "elapsed_seconds": runtime.get("elapsed_seconds"),
                }
            )
    complete = {
        row["condition_id"]
        for row in rows
        if row.get("stage") == "aggregate_complete"
        and row.get("status") == "complete"
        and row.get("enrichment_complete")
    }
    result = {
        "seed": expected_seed,
        "expected": len(expected),
        "complete": len(expected.intersection(complete)),
        "missing": sorted(expected - complete),
        "runs": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alibaba Ours K/P/R/lambda/tau direct next-bucket sensitivity experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--study", choices=("all", *STUDIES), default="all")
    for name in ("preflight", "check"):
        item = subparsers.add_parser(name)
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    stage = subparsers.add_parser("stage")
    stage.add_argument("stage", choices=(*BASE_STAGES, "smoke-test"))
    stage.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    stage.add_argument("--condition", required=True)
    stage.add_argument("--resume", action="store_true")
    stage.add_argument("--device", choices=("cpu", "cuda"))
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument(
        "--condition", default="baseline-k08-p03-r02-lambda0p02-tau60s-l1440-h1"
    )
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    enrich = subparsers.add_parser("enrich")
    enrich.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    enrich.add_argument("--condition", required=True)
    enrich.add_argument("--resume", action="store_true")
    condition = subparsers.add_parser("run-condition")
    condition.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    condition.add_argument("--condition", required=True)
    condition.add_argument("--resume", action="store_true")
    condition.add_argument("--device", choices=("cpu", "cuda"))
    study = subparsers.add_parser("run-study")
    study.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    study.add_argument("--study", choices=STUDIES, required=True)
    study.add_argument("--resume", action="store_true")
    study.add_argument("--device", choices=("cpu", "cuda"))
    p2200 = subparsers.add_parser("run-p2200")
    p2200.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p2200.add_argument("--study", choices=("all", *STUDIES), default="all")
    p2200.add_argument("--resume", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    collect.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    collect.add_argument("--output", type=Path, default=ROOT / "reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = args.config.resolve()
    if args.command == "plan":
        return command_plan(config, args.study)
    if args.command in {"preflight", "check"}:
        return command_preflight(config)
    if args.command == "stage":
        return command_stage(
            config,
            select_condition(config, args.condition),
            args.stage,
            resume=args.resume,
            device=args.device,
        )
    if args.command == "smoke-test":
        return command_stage(
            config,
            select_condition(config, args.condition),
            "smoke-test",
            resume=args.resume,
            device=args.device,
        )
    if args.command == "enrich":
        condition = select_condition(config, args.condition)
        _, loaded, run_dir = _load_generated(config, condition, resume=args.resume)
        enrich_run(loaded, run_dir, condition, resume=args.resume)
        return 0
    if args.command == "run-condition":
        return run_condition(
            config,
            select_condition(config, args.condition),
            resume=args.resume,
            device_override=args.device,
        )
    if args.command == "run-study":
        return command_run_study(
            config,
            args.study,
            resume=args.resume,
            device=args.device,
            require_p2200=False,
        )
    if args.command == "run-p2200":
        return command_run_study(
            config,
            args.study,
            resume=args.resume,
            device="cuda",
            require_p2200=True,
        )
    if args.command == "status":
        return command_status(config)
    if args.command == "collect":
        report = collect_suite(config, args.runs_root, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


__all__ = [
    "build_parser",
    "command_plan",
    "command_preflight",
    "environment_manifest",
    "main",
    "materialize_condition_config",
    "run_condition",
]
