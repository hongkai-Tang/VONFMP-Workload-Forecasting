from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

from .config import ConfigError, ExperimentConfig
from .data import (
    DataError,
    load_raw_dataset,
    prepare_data,
    split_indices,
    valid_origins,
)
from .engine import ExperimentError, ExperimentRunner
from .utils import code_hash, json_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alibaba 200容器直接多步L-H对比实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", required=True, help="实验JSON配置")

    preflight = subparsers.add_parser("preflight", help="检查数据、GPU和24任务协议")
    add_config(preflight)

    prepare = subparsers.add_parser("prepare", help="拟合训练段候选模式并缓存逐时隙隶属度")
    add_config(prepare)
    prepare.add_argument("--force", action="store_true", help="忽略兼容缓存并重新准备")

    smoke = subparsers.add_parser("smoke-test", help="运行6个边界组合的最小测试")
    add_config(smoke)
    smoke.add_argument("--run-id", default="alibaba-duibi1-smoke")

    run_all = subparsers.add_parser("run-all", help="按固定顺序运行24个正式模型")
    add_config(run_all)
    run_all.add_argument("--run-id", required=True)
    run_all.add_argument("--resume", action="store_true")

    summarize = subparsers.add_parser("summarize", help="从已保存预测重新计算全部指标")
    add_config(summarize)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--resume", action="store_true")

    verify = subparsers.add_parser("verify", help="验证模型、预测和指标是否完整")
    add_config(verify)
    verify.add_argument("--run-id", required=True)

    status = subparsers.add_parser("status", help="显示24任务当前状态")
    add_config(status)
    status.add_argument("--run-id", required=True)
    return parser


def _print_json(value: Any) -> None:
    print(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def preflight_report(config: ExperimentConfig) -> dict[str, Any]:
    raw = load_raw_dataset(config.dataset_path)
    cadence = np.diff(raw.time_seconds)
    train_end, validation_end = split_indices(raw.steps, config.split_ratios)
    stride = int(config.evaluation.get("origin_stride_steps", 60))
    task_rows: list[dict[str, Any]] = []
    for index, task in enumerate(config.task_queue(), start=1):
        train = valid_origins(
            split_start=0,
            split_end=train_end,
            history_length=max(config.history_lengths),
            horizon_steps=task.horizon.steps,
            stride=1,
            training=True,
        )
        validation = valid_origins(
            split_start=train_end,
            split_end=validation_end,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            stride=stride,
        )
        test = valid_origins(
            split_start=validation_end,
            split_end=raw.steps,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            stride=stride,
        )
        task_rows.append(
            {
                "queue_index": index,
                **task.as_dict(),
                "train_origins_available": len(train),
                "validation_origins": len(validation),
                "test_origins": len(test),
            }
        )
    adjacency = raw.deployment.T @ raw.deployment
    np.fill_diagonal(adjacency, 0)
    edge_count = int(np.count_nonzero(np.triu(adjacency > 0, k=1)))
    disk = shutil.disk_usage(config.output_dir.parent if config.output_dir.parent.exists() else config.root)
    cuda_available = torch.cuda.is_available()
    requested = str(config.training.get("device", "cpu"))
    device_ok = requested.lower() == "auto" or not requested.lower().startswith("cuda") or cuda_available
    report = {
        "ok": bool(
            raw.workloads == config.expected_workloads
            and (not cadence.size or np.all(cadence == 60))
            and all(row["train_origins_available"] > 0 for row in task_rows)
            and all(row["validation_origins"] > 0 for row in task_rows)
            and all(row["test_origins"] > 0 for row in task_rows)
            and device_ok
        ),
        "config": config.path,
        "experiment_root": config.root,
        "dataset": config.dataset_path,
        "forecast_strategy": config.forecast_strategy,
        "model_family": "Ours-TopologyOnly-Direct",
        "recursive_feedback": False,
        "time_step_seconds": config.time_step_seconds,
        "num_states": config.num_states,
        "max_order": int(config.model["max_order"]),
        "time_decay": float(config.model["time_decay"]),
        "backoff_blend": float(config.model["backoff_blend"]),
        "message_passing_steps": int(config.model["message_passing_steps"]),
        "workloads": raw.workloads,
        "expected_workloads": config.expected_workloads,
        "steps": raw.steps,
        "duration_days": raw.steps * 60 / 86400.0,
        "resources": list(map(str, raw.resource_names)),
        "input_valid_ratio": float(raw.input_mask.mean()),
        "target_observed_ratio": float(raw.target_mask.mean()),
        "time_axis_regular_60s": bool(not cadence.size or np.all(cadence == 60)),
        "train_steps": train_end,
        "validation_steps": validation_end - train_end,
        "test_steps": raw.steps - validation_end,
        "deployment_nodes": int(raw.deployment.shape[0]),
        "deployment_edges": edge_count,
        "link_mode": "topology_only",
        "requested_device": requested,
        "resolved_device": "cuda" if requested.lower() == "auto" and cuda_available else ("cpu" if requested.lower() == "auto" else requested),
        "cuda_available": cuda_available,
        "compute_device_ok": device_ok,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_compute_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "disk_free_gb": disk.free / (1024**3),
        "source_hash": code_hash(config.root),
        "task_count": len(task_rows),
        "task_queue": task_rows,
        "membership_note": "K=8逐时隙隶属度仅在训练段拟合一次，随后固定并供全部L/H复用。",
        "h0_note": "横轴可从0显示，但真实预测点从h=1开始；不存在伪造的H=0样本。",
    }
    return report


def status_report(config: ExperimentConfig, run_id: str) -> dict[str, Any]:
    run_dir = config.output_dir / run_id
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise ExperimentError(f"找不到运行状态: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for task in state.get("tasks", []):
        counts[task.get("status", "unknown")] = counts.get(task.get("status", "unknown"), 0) + 1
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "updated_at": state.get("updated_at"),
        "counts": counts,
        "tasks": state.get("tasks", []),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = ExperimentConfig.load(args.config)
        if args.command == "preflight":
            report = preflight_report(config)
            _print_json(report)
            return 0 if report["ok"] else 2
        if args.command == "prepare":
            data = prepare_data(config, force=bool(args.force))
            _print_json(
                {
                    "ok": True,
                    "cache_path": data.cache_path,
                    "cache_signature": data.cache_signature,
                    "workloads": data.workloads,
                    "steps": data.steps,
                    "num_states": data.num_states,
                }
            )
            return 0
        if args.command == "status":
            _print_json(status_report(config, args.run_id))
            return 0
        if args.command == "smoke-test":
            runner = ExperimentRunner(
                config,
                run_id=args.run_id,
                resume=True,
                smoke=True,
            )
            report = runner.run_all()
            _print_json(report)
            return 0 if report.get("all_complete") else 2
        if args.command == "run-all":
            runner = ExperimentRunner(
                config,
                run_id=args.run_id,
                resume=bool(args.resume),
                smoke=False,
            )
            report = runner.run_all()
            _print_json(report)
            return 0 if report.get("all_complete") else 2
        if args.command == "summarize":
            runner = ExperimentRunner(
                config,
                run_id=args.run_id,
                resume=True,
                smoke=False,
            )
            report = runner.summarize_existing()
            _print_json(report)
            return 0 if report.get("all_complete") else 2
        if args.command == "verify":
            runner = ExperimentRunner(
                config,
                run_id=args.run_id,
                resume=True,
                smoke=False,
            )
            report = runner.verify()
            _print_json(report)
            return 0 if report.get("ok") else 2
        raise AssertionError(args.command)
    except (ConfigError, DataError, ExperimentError, ValueError, RuntimeError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


__all__ = ["main", "preflight_report", "status_report"]
