from __future__ import annotations

"""CLI adapter that applies one ablation to the proven base experiment."""

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import alibaba_ours_exp.cli as base_cli
from alibaba_ours_exp.checkpoint import make_run_identity

from .direct import (
    direct_forecast,
    initialize_slot_prototypes,
    train_direct_model,
    true_slot_memberships,
)
from .model import (
    ABLATION_VARIANTS,
    DISPLAY_NAMES,
    AblationModel,
    AblationModelConfig,
    canonical_variant,
    hypergraph_adjacency,
    pairwise_adjacency,
)
from .suite import collect_suite, write_csv_atomic, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
BASE_EXPERIMENT_ROOT = ROOT.parent / "alibaba_ours_lh_recursive"
DEFAULT_CONFIG = ROOT / "configs" / "ablation.json"
BASE_COMMANDS = {
    "preflight",
    "prepare",
    "train",
    "evaluate",
    "aggregate",
    "run-all",
    "smoke-test",
}

# Keep pristine references because run-third executes more than one variant in
# the same Python process.  Capturing already-patched functions would stack
# adapters and leak the previous variant into the next run.
_BASE_MODEL_CONFIG = base_cli._model_config
_BASE_INPUT_PATHS = base_cli._identity_input_paths
_BASE_TRAINING_CONFIG = base_cli._training_config
_BASE_TOPOLOGY_COVERAGE = base_cli._topology_coverage

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration {path} must contain one JSON object")
    return value


def _variant_from_config(path: Path) -> str:
    raw = _read_json(path)
    ablation = raw.get("ablation", {})
    if not isinstance(ablation, Mapping) or not ablation.get("variant"):
        raise ValueError("generated configuration is missing ablation.variant")
    return canonical_variant(str(ablation["variant"]))


def materialize_variant_config(
    base_path: Path,
    variant: str,
    *,
    device_override: str | None = None,
) -> Path:
    variant = canonical_variant(variant)
    raw = _read_json(base_path)
    raw["ablation"] = {
        "variant": variant,
        "display_name": DISPLAY_NAMES[variant],
        "single_substitution": True,
        "effective_overrides": {
            "max_order": 1 if variant == "vo_markov" else raw["protocol"]["max_order"],
            "message_passing_steps": (
                1 if variant == "multi_hop" else raw["protocol"]["message_passing_steps"]
            ),
            "dynamic_transition": variant != "dyn_trans",
            "neighbour_membership": variant != "ns_mem",
            "graph_structure": "pairwise" if variant == "hypergraph" else "hypergraph",
            "forecast_strategy": "direct_multi_step",
            "membership_definition": "per_slot",
            "prediction_feedback": False,
        },
    }
    raw["protocol"] = dict(raw["protocol"])
    raw["protocol"]["experiment_name"] = f"Alibaba Ours ablation {DISPLAY_NAMES[variant]}"
    raw["protocol"]["forecast_strategy"] = "direct_multi_step"
    raw["protocol"]["membership_definition"] = "per_slot"
    raw["protocol"]["target_window_steps"] = 1
    if device_override is not None:
        if device_override not in {"cpu", "cuda"}:
            raise ValueError("device_override must be cpu or cuda")
        raw["training"] = dict(raw["training"])
        raw["training"]["device"] = device_override
    output_rel = Path(str(raw["output_dir"]))
    generated_dir = (base_path.parent.parent / "generated").resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)
    destination = generated_dir / f"{variant}.json"
    # output_dir stays shared; identity-aware run names prevent collisions.
    del output_rel
    write_json_atomic(destination, raw)
    return destination


def _patch_base_cli(config_path: Path, variant: str) -> None:
    variant = canonical_variant(variant)

    def model_config(config: Any, history_length: int, resource_dim: int) -> AblationModelConfig:
        base = _BASE_MODEL_CONFIG(config, history_length, resource_dim)
        return AblationModelConfig(
            **asdict(base),
            variant=variant,
            max_forecast_horizon=int(config.max_forecast_horizon),
            direct_hidden_dim=int(
                config.protocol.get("direct_hidden_dim", config.protocol["transition_hidden_dim"])
            ),
            membership_definition="per_slot",
            diagnostic_max_order=int(config.protocol["max_order"]),
        )

    def training_config(config: Any, checkpoint_path: Path, **kwargs: Any) -> Any:
        value = _BASE_TRAINING_CONFIG(config, checkpoint_path, **kwargs)
        # The sibling base pipeline can touch the best/resume checkpoint before
        # the direct trainer reaches its own checkpoint setup.  Create both
        # parents here so a completely fresh per-L run is portable across the
        # base-pipeline versions installed on the P2200 machines.
        for candidate in (
            getattr(value, "checkpoint_path", None),
            getattr(value, "resume_checkpoint_path", None),
        ):
            if candidate is not None:
                Path(candidate).parent.mkdir(parents=True, exist_ok=True)
        value.direct_horizon_batch_size = int(
            config.training.get("direct_horizon_batch_size", 60)
        )
        value.direct_validation_origins = int(
            config.training.get("direct_validation_origins", 32)
        )
        return value

    def mode_metadata(config: Any) -> dict[str, Any]:
        return {
            "link_mode": base_cli._link_mode(config),
            "link_features_required": False,
            "real_link_features": False,
            "full_ours": False,
            "experiment_variant": DISPLAY_NAMES[variant],
            "ablation_variant": variant,
            "ablation_display_name": DISPLAY_NAMES[variant],
            "single_substitution": True,
            "forecast_strategy": "direct_multi_step",
            "membership_definition": "per_slot",
            "target_window_steps": 1,
            "prediction_feedback": False,
        }

    def identity_input_paths(config: Any) -> dict[str, Path]:
        paths = _BASE_INPUT_PATHS(config)
        # The adapter imports the sibling base pipeline.  Include those Python
        # files in the identity so a base implementation change invalidates
        # stale checkpoints instead of silently reusing them.
        for path in sorted((BASE_EXPERIMENT_ROOT / "src" / "alibaba_ours_exp").glob("*.py")):
            paths[f"base_code:{path.name}"] = path
        return paths

    def current_run_identity(config: Any) -> dict[str, Any]:
        return make_run_identity(
            raw_config=base_cli._raw_config(config),
            experiment_root=ROOT,
            project_root=config.project_root,
            input_paths=identity_input_paths(config),
        )

    def run_id(config: Any, identity: Mapping[str, Any] | None = None) -> str:
        current = dict(identity or current_run_identity(config))
        digest = str(current["identity_hash"])[:12]
        return f"alibaba-ablation-{variant}-seed7-{digest}"

    def topology_coverage(topology: Any, hops: int) -> dict[str, Any]:
        effective_hops = 1 if variant == "multi_hop" else int(hops)
        return _BASE_TOPOLOGY_COVERAGE(topology, effective_hops)

    base_cli.AlibabaOursModel = AblationModel
    base_cli._model_config = model_config
    base_cli._training_config = training_config
    base_cli._mode_metadata = mode_metadata
    base_cli._identity_input_paths = identity_input_paths
    base_cli._current_run_identity = current_run_identity
    base_cli._run_id = run_id
    base_cli._topology_coverage = topology_coverage
    # Keep the ablation package robust when a machine still has an older copy
    # of the sibling base experiment. Undefined metrics remain NaN in numeric
    # CSV/Parquet outputs but are encoded as JSON null in strict fragments.
    base_cli.atomic_write_json = write_json_atomic
    base_cli.initialize_fixed_k_prototypes = initialize_slot_prototypes
    base_cli.train_model = train_direct_model
    base_cli.recursive_forecast = direct_forecast
    base_cli._true_memberships = true_slot_memberships
    base_cli._unweighted_workload_adjacency = (
        pairwise_adjacency if variant == "hypergraph" else hypergraph_adjacency
    )

    configured = _variant_from_config(config_path)
    if configured != variant:
        raise ValueError(f"variant mismatch: CLI={variant}, config={configured}")


def _run_base(
    command: str,
    variant: str,
    config: Path,
    extra: Sequence[str],
    *,
    device_override: str | None = None,
) -> int:
    generated = materialize_variant_config(config, variant, device_override=device_override)
    _patch_base_cli(generated, variant)
    return int(base_cli.main([command, "--config", str(generated), *extra]))


def _suite_variants(config: Path) -> list[str]:
    raw = _read_json(config)
    supplied = raw.get("ablation_suite", {}).get("variants", ABLATION_VARIANTS)
    variants = [canonical_variant(item) for item in supplied]
    if sorted(variants) != sorted(ABLATION_VARIANTS) or len(variants) != len(set(variants)):
        raise ValueError("ablation_suite.variants must contain each variant exactly once")
    return variants


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return {"available": True, "rows": rows}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def environment_manifest(machine_label: str, variants: Sequence[str]) -> dict[str, Any]:
    gpu_name = None
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "captured_at": _utc_now(),
        "machine_label": str(machine_label),
        "reference_machine": "second",
        "assigned_variants": list(variants),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_memory,
        "nvidia_smi": _nvidia_smi(),
        "p2200_expected": True,
    }


def command_plan(config: Path) -> int:
    variants = _suite_variants(config)
    rows: list[dict[str, Any]] = []
    for order, variant in enumerate(variants, start=1):
        rows.append(
            {
                "target_machine": "third",
                "order": order,
                "variant": variant,
                "display_name": DISPLAY_NAMES[variant],
                "device": "cuda",
                "gpu_expected": "NVIDIA P2200",
                "execution_mode": "sequential_resume_safe",
                "forecast_strategy": "direct_multi_step",
                "membership_definition": "per_slot",
                "prediction_feedback": False,
                "suite_command": (
                    "python run_experiment.py run-third "
                    "--config configs/ablation.json --resume"
                ),
            }
        )
    write_csv_atomic(ROOT / "runs" / "experiment_plan.csv", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def command_design_check(config: Path) -> int:
    raw = _read_json(config)
    project_root = (config.parent / Path(str(raw.get("project_root", ".")))).resolve()
    deployment_name = raw["deployment_priority"][0]
    deployment_config = raw["deployment_sources"][deployment_name]
    deployment_path = (project_root / deployment_config["path"]).resolve()
    with deployment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    workload_column = deployment_config["workload_column"]
    node_column = deployment_config["node_column"]
    by_node: dict[str, set[str]] = {}
    workloads: set[str] = set()
    for row in rows:
        workload = str(row[workload_column])
        node = str(row[node_column])
        workloads.add(workload)
        by_node.setdefault(node, set()).add(workload)
    neighbours = {workload: set() for workload in workloads}
    for members in by_node.values():
        for source in members:
            neighbours[source].update(members - {source})
    indirect_pairs: set[tuple[str, str]] = set()
    for source, direct in neighbours.items():
        second = set().union(*(neighbours[item] for item in direct)) if direct else set()
        for target in second - direct - {source}:
            indirect_pairs.add(tuple(sorted((source, target))))
    sizes = [len(members) for members in by_node.values()]
    neighbour_workloads = sum(bool(values) for values in neighbours.values())
    report = {
        "deployment_path": str(deployment_path),
        "workload_count": len(workloads),
        "node_count": len(by_node),
        "isolated_workloads": len(workloads) - neighbour_workloads,
        "workloads_with_neighbours": neighbour_workloads,
        "neighbour_coverage": neighbour_workloads / max(len(workloads), 1),
        "nodes_with_at_least_2_workloads": sum(size >= 2 for size in sizes),
        "high_order_hyperedges_size_ge_3": sum(size >= 3 for size in sizes),
        "maximum_hyperedge_size": max(sizes, default=0),
        "indirect_two_hop_pairs": len(indirect_pairs),
        "ns_mem_identifiable": neighbour_workloads > 0,
        "hypergraph_identifiable": any(size >= 3 for size in sizes),
        "multi_hop_indirect_identifiable": bool(indirect_pairs),
        "interpretation": (
            "Structural ablations without identifiable support must be reported as exploratory/N/A; "
            "a near-zero performance change is not evidence that the module is unnecessary."
        ),
    }
    write_json_atomic(ROOT / "reports" / "design_identifiability.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_run_third(config: Path, resume: bool) -> int:
    variants = _suite_variants(config)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    manifests = ROOT / "runs" / "machine_manifests"
    manifest = environment_manifest("third", variants)
    gpu_name = str(manifest.get("gpu_name") or "")
    if "P2200" not in gpu_name.upper():
        raise RuntimeError(
            "the third machine is expected to match the second machine's NVIDIA P2200, "
            f"but PyTorch reports {gpu_name or 'no CUDA GPU'}"
        )
    write_json_atomic(manifests / "third-machine.json", manifest)
    for variant in variants:
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        extra = ["--resume"] if resume else []
        exit_code = _run_base("run-all", variant, config, extra)
        record = {
            "machine_label": "third",
            "reference_machine": "second",
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "started_at": manifest["captured_at"],
            "ended_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "exit_code": exit_code,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        }
        write_json_atomic(manifests / f"third-machine-{variant}.json", record)
        if exit_code != 0:
            return exit_code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alibaba five-variant ablation experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in sorted(BASE_COMMANDS):
        child = subparsers.add_parser(command)
        child.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--run-id")
        child.add_argument("--resume", action="store_true")
        child.add_argument("--device", choices=("cpu", "cuda"))

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    design = subparsers.add_parser("design-check")
    design.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    third = subparsers.add_parser("run-third")
    third.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    third.add_argument("--resume", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    collect.add_argument("--output", type=Path, default=ROOT / "reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in BASE_COMMANDS:
        extra: list[str] = []
        if args.run_id:
            extra.extend(["--run-id", args.run_id])
        if args.resume:
            extra.append("--resume")
        return _run_base(
            args.command,
            args.variant,
            args.config.resolve(),
            extra,
            device_override=args.device,
        )
    if args.command == "plan":
        return command_plan(args.config.resolve())
    if args.command == "design-check":
        return command_design_check(args.config.resolve())
    if args.command == "run-third":
        return command_run_third(args.config.resolve(), args.resume)
    if args.command == "collect":
        summary = collect_suite(args.runs_root.resolve(), args.output.resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


__all__ = [
    "environment_manifest",
    "main",
    "materialize_variant_config",
]
