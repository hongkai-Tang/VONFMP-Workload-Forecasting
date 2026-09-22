from __future__ import annotations

"""Post-evaluation metrics retained for later sensitivity visualizations."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

import alibaba_ours_exp.cli as base_cli
from alibaba_ours_exp.config import ExperimentConfig
from alibaba_ours_exp.metrics import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    evaluate_predictions,
)

from .design import Condition
from .progress import SensitivityProgressTracker


EPS = 1e-12


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, Mapping):
            result.update(_flatten(item, name + "."))
        elif np.isscalar(item) and not isinstance(item, (str, bytes)):
            result[name] = item.item() if isinstance(item, np.generic) else item
    return result


def _input_signature(run_dir: Path, fragments: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    identity = run_dir / "run_identity.json"
    if identity.is_file():
        digest.update(identity.read_bytes())
    model_files = sorted((run_dir / "models").glob("L=*/complete.json"))
    guards = sorted(
        fragment.parent / f"complete-{fragment.stem[len('origin='): ]}.json"
        for fragment in fragments
    )
    for path in (*model_files, *guards):
        digest.update(str(path.relative_to(run_dir)).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _membership_error_columns(
    truth: np.ndarray,
    prediction: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    clipped_truth = np.clip(truth, EPS, 1.0)
    clipped_prediction = np.clip(prediction, EPS, 1.0)
    difference = prediction - truth
    absolute = np.abs(difference)
    squared = difference * difference
    midpoint = 0.5 * (clipped_truth + clipped_prediction)
    truth_norm = np.linalg.norm(truth, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    denominator = truth_norm * prediction_norm
    return {
        f"{prefix}_mu_mae": np.mean(absolute, axis=1),
        f"{prefix}_mu_rmse": np.sqrt(np.mean(squared, axis=1)),
        f"{prefix}_mu_l1": np.sum(absolute, axis=1),
        f"{prefix}_mu_l2": np.sqrt(np.sum(squared, axis=1)),
        f"{prefix}_mu_total_variation": 0.5 * np.sum(absolute, axis=1),
        f"{prefix}_mu_brier_soft": np.sum(squared, axis=1),
        f"{prefix}_mu_kl_true_pred": np.sum(
            clipped_truth * np.log(clipped_truth / clipped_prediction), axis=1
        ),
        f"{prefix}_mu_js_divergence": 0.5
        * (
            np.sum(clipped_truth * np.log(clipped_truth / midpoint), axis=1)
            + np.sum(clipped_prediction * np.log(clipped_prediction / midpoint), axis=1)
        ),
        f"{prefix}_mu_hellinger": np.sqrt(
            0.5 * np.sum((np.sqrt(clipped_truth) - np.sqrt(clipped_prediction)) ** 2, axis=1)
        ),
        f"{prefix}_mu_cosine_similarity": np.divide(
            np.sum(truth * prediction, axis=1),
            denominator,
            out=np.zeros(truth.shape[0], dtype=np.float64),
            where=denominator > EPS,
        ),
        f"{prefix}_mu_soft_cross_entropy": -np.sum(
            truth * np.log(clipped_prediction), axis=1
        ),
        f"{prefix}_predicted_entropy": -np.sum(
            prediction * np.log(clipped_prediction), axis=1
        ),
    }


def _with_disk_total(
    values: np.ndarray,
    resource_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if {"disk_i", "disk_o"}.issubset(resource_names):
        disk_i = resource_names.index("disk_i")
        disk_o = resource_names.index("disk_o")
        return (
            np.column_stack([values, values[:, disk_i] + values[:, disk_o]]),
            [*resource_names, "disk_io_total"],
        )
    return values, list(resource_names)


def _condition_prefix(condition: Condition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "study": condition.study,
        "parameter": condition.parameter_name,
        "parameter_value": condition.parameter_value,
        "K": condition.num_states,
        "P": condition.max_order,
        "R": condition.message_passing_steps,
        "lambda_per_minute": condition.time_decay_per_minute,
        "lambda_per_step": condition.effective_time_decay_per_step,
        "tau_seconds": condition.granularity_seconds,
        "history_steps": condition.history_length,
        "history_seconds": condition.history_duration_seconds,
        "forecast_steps": 1,
        "forecast_seconds": condition.granularity_seconds,
        "seed": condition.seed,
    }


def _evaluate(
    truth_mu: np.ndarray,
    predicted_mu: np.ndarray,
    truth_resources: np.ndarray,
    predicted_resources: np.ndarray,
    resource_names: list[str],
    num_states: int,
) -> dict[str, Any]:
    return evaluate_predictions(
        truth_mu,
        predicted_mu,
        labels=list(range(num_states)),
        true_resources=truth_resources,
        predicted_resources=predicted_resources,
        resource_names=resource_names,
    )


def _append_pool(
    pool: dict[str, dict[str, dict[str, list[np.ndarray]]]],
    scope: str,
    predictor: str,
    truth_mu: np.ndarray,
    predicted_mu: np.ndarray,
    truth_resources: np.ndarray,
    predicted_resources: np.ndarray,
    transition_reference_mu: np.ndarray,
) -> None:
    item = pool.setdefault(scope, {}).setdefault(
        predictor,
        {
            "truth_mu": [],
            "predicted_mu": [],
            "truth_resources": [],
            "predicted_resources": [],
            "transition_reference_mu": [],
        },
    )
    item["truth_mu"].append(truth_mu)
    item["predicted_mu"].append(predicted_mu)
    item["truth_resources"].append(truth_resources)
    item["predicted_resources"].append(predicted_resources)
    item["transition_reference_mu"].append(transition_reference_mu)


def _stack_pool(item: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in item.items()}


def _write_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_csv(path.with_suffix(".csv"), rows)
    atomic_write_parquet(path.with_suffix(".parquet"), rows)


@torch.no_grad()
def _enrich_run_impl(
    config: ExperimentConfig,
    run_dir: Path,
    condition: Condition,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Add per-sample errors and persistence/seasonal baselines after evaluation."""

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fragments = sorted((run_dir / "predictions").glob("L=*/split=*/origin=*.parquet"))
    if not fragments:
        raise FileNotFoundError(f"no prediction fragments exist under {run_dir / 'predictions'}")
    signature = _input_signature(run_dir, fragments)
    completion = metrics_dir / "enrichment.complete.json"
    required = (
        metrics_dir / "enriched_predictions.parquet",
        metrics_dir / "baseline_metrics_by_origin.csv",
        metrics_dir / "baseline_metrics_by_condition.csv",
        metrics_dir / "error_quantiles.csv",
        metrics_dir / "transition_metrics.csv",
    )
    if resume and completion.is_file() and all(path.is_file() for path in required):
        manifest = json.loads(completion.read_text(encoding="utf-8"))
        if manifest.get("input_signature") == signature and manifest.get("status") == "complete":
            print(f"[enrich] reuse complete metrics: {metrics_dir}", flush=True)
            return manifest

    cache = base_cli._load_cache(run_dir)
    with np.load(run_dir / "models" / "shared_preprocessing.npz", allow_pickle=False) as prep:
        mean = prep["mean"].astype(np.float32)
        std = prep["std"].astype(np.float32)
    normalized = base_cli._normalized_resources(cache, mean, std)
    input_values = cache["input_values"].astype(np.float32)
    observed = cache["observed_values"].astype(np.float32)
    active = cache["input_valid_mask"].astype(bool)
    imputed = cache["imputed_mask"].astype(bool)
    time_values = cache["time_seconds"].astype(np.int64)
    workload_ids = cache["workload_ids"].astype(str)
    resource_names = list(cache["resource_names"].astype(str))
    _, names = _with_disk_total(
        np.zeros((1, len(resource_names)), dtype=np.float32), resource_names
    )
    workload_position = {value: index for index, value in enumerate(workload_ids)}
    time_position = {int(value): index for index, value in enumerate(time_values)}
    model_path = run_dir / "models" / f"L={condition.history_length}" / "model.pt"
    model, _ = base_cli.AlibabaOursModel.load(model_path, map_location="cpu")
    model.eval()

    tracker = SensitivityProgressTracker(
        run_dir / "progress-enrichment",
        run_id=run_dir.name + "-enrichment",
        training_epochs=1,
        heartbeat_interval_seconds=float(config.runtime.get("progress_interval_seconds", 30)),
    )
    tracker.start(
        total=len(fragments),
        stage="enrich",
        message="per-sample, persistence and daily-seasonal metrics",
        reset=True,
    )

    prefix = _condition_prefix(condition)
    sample_frames: list[pd.DataFrame] = []
    origin_rows: list[dict[str, Any]] = []
    pool: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    total_samples = 0
    for completed, fragment in enumerate(fragments, start=1):
        frame = pd.read_parquet(fragment)
        if frame.empty:
            tracker.update(completed=completed, message=f"empty={fragment.stem}")
            continue
        origin_time = int(frame["origin_time"].iloc[0])
        target_time = int(frame["target_time"].iloc[0])
        origin_index = time_position[origin_time]
        target_index = time_position[target_time]
        if target_index != origin_index + 1:
            raise ValueError(f"{fragment}: expected direct next-bucket target")
        positions = np.asarray([workload_position[str(value)] for value in frame["workload_id"]], dtype=int)
        k = condition.num_states
        truth_mu = frame[[f"true_mu_{state}" for state in range(k)]].to_numpy(dtype=np.float64)
        model_mu = frame[[f"pred_mu_{state}" for state in range(k)]].to_numpy(dtype=np.float64)
        current_tensor = torch.from_numpy(normalized[positions, origin_index][:, None, :])
        current_mask = torch.from_numpy(active[positions, origin_index][:, None])
        persistence_mu = model.encode_membership(current_tensor, current_mask).cpu().numpy()
        model_resources = (
            model.shape_encoder.decode_resources(
                torch.from_numpy(model_mu.astype(np.float32))
            ).cpu().numpy()
            * std
            + mean
        )
        truth_resources = observed[positions, target_index]
        persistence_resources = input_values[positions, origin_index]

        seasonal_index = target_index - condition.time_period_steps
        seasonal_valid = np.zeros(len(positions), dtype=bool)
        seasonal_mu = np.full_like(model_mu, np.nan)
        seasonal_resources = np.full_like(truth_resources, np.nan)
        if seasonal_index >= 0:
            seasonal_valid = active[positions, seasonal_index]
            if np.any(seasonal_valid):
                selected_positions = positions[seasonal_valid]
                seasonal_tensor = torch.from_numpy(
                    normalized[selected_positions, seasonal_index][:, None, :]
                )
                seasonal_mask = torch.ones((selected_positions.size, 1), dtype=torch.bool)
                seasonal_mu[seasonal_valid] = model.encode_membership(
                    seasonal_tensor, seasonal_mask
                ).cpu().numpy()
                seasonal_resources[seasonal_valid] = input_values[
                    selected_positions, seasonal_index
                ]

        truth_resources, names = _with_disk_total(truth_resources, resource_names)
        model_resources, _ = _with_disk_total(model_resources, resource_names)
        persistence_resources, _ = _with_disk_total(persistence_resources, resource_names)
        seasonal_resources, _ = _with_disk_total(seasonal_resources, resource_names)

        history_start = origin_index - condition.history_length + 1
        persistence_class = np.argmax(persistence_mu, axis=1)
        seasonal_class = np.where(
            seasonal_valid, np.argmax(np.nan_to_num(seasonal_mu), axis=1), -1
        )
        true_class = frame["true_class"].to_numpy()
        extra_columns: dict[str, Any] = {
            **{
                key: np.repeat(value, len(frame))
                for key, value in prefix.items()
                if key not in frame.columns
            },
            "origin_index": np.repeat(origin_index, len(frame)),
            "target_index": np.repeat(target_index, len(frame)),
            "seasonal_source_index": np.repeat(seasonal_index, len(frame)),
            "seasonal_valid": seasonal_valid,
            "history_observed_coverage": active[
                positions, history_start : origin_index + 1
            ].mean(axis=1),
            "history_imputed_fraction": imputed[
                positions, history_start : origin_index + 1
            ].mean(axis=1),
            "persistence_class": persistence_class,
            "seasonal_class": seasonal_class,
            "true_transition": true_class != persistence_class,
            "model_correct": frame["pred_class"].to_numpy() == true_class,
            "persistence_correct": persistence_class == true_class,
            "seasonal_correct": seasonal_valid & (seasonal_class == true_class),
        }
        for values in (
            _membership_error_columns(truth_mu, model_mu, "model"),
            _membership_error_columns(truth_mu, persistence_mu, "persistence"),
        ):
            extra_columns.update(values)
        seasonal_errors = _membership_error_columns(
            truth_mu[seasonal_valid], seasonal_mu[seasonal_valid], "seasonal"
        ) if np.any(seasonal_valid) else {}
        for key in _membership_error_columns(truth_mu[:1], model_mu[:1], "seasonal"):
            column = np.full(len(frame), np.nan)
            if key in seasonal_errors:
                column[seasonal_valid] = seasonal_errors[key]
            extra_columns[key] = column
        for state in range(k):
            extra_columns[f"persistence_mu_{state}"] = persistence_mu[:, state]
            extra_columns[f"seasonal_mu_{state}"] = seasonal_mu[:, state]
        for index, name in enumerate(names):
            extra_columns[f"true_resource_{name}"] = truth_resources[:, index]
            for predictor, values in (
                ("model", model_resources),
                ("persistence", persistence_resources),
                ("seasonal", seasonal_resources),
            ):
                extra_columns[f"{predictor}_resource_{name}"] = values[:, index]
                extra_columns[f"{predictor}_resource_error_{name}"] = (
                    values[:, index] - truth_resources[:, index]
                )
                extra_columns[f"{predictor}_resource_abs_error_{name}"] = np.abs(
                    values[:, index] - truth_resources[:, index]
                )
        frame = pd.concat(
            [frame.reset_index(drop=True), pd.DataFrame(extra_columns)], axis=1
        )
        sample_frames.append(frame)
        total_samples += len(frame)

        split = str(frame["split"].iloc[0])
        predictor_values = {
            "model": (model_mu, model_resources, np.ones(len(frame), dtype=bool)),
            "persistence": (persistence_mu, persistence_resources, np.ones(len(frame), dtype=bool)),
            "seasonal": (seasonal_mu, seasonal_resources, seasonal_valid),
        }
        for predictor, (predicted_mu, predicted_resources, valid) in predictor_values.items():
            if not np.any(valid):
                continue
            evaluated = _evaluate(
                truth_mu[valid], predicted_mu[valid], truth_resources[valid],
                predicted_resources[valid], names, k,
            )
            origin_rows.append(
                {
                    **prefix,
                    "split": split,
                    "origin_id": str(frame["origin_id"].iloc[0]),
                    "origin_time": origin_time,
                    "target_time": target_time,
                    "predictor": predictor,
                    "n_samples": int(np.sum(valid)),
                    "seasonal_coverage": float(np.mean(seasonal_valid)),
                    **_flatten(evaluated),
                }
            )
            for scope in ("all", split):
                _append_pool(
                    pool, scope, predictor, truth_mu[valid], predicted_mu[valid],
                    truth_resources[valid], predicted_resources[valid],
                    persistence_mu[valid],
                )
        tracker.update(
            completed=completed,
            message=f"{split}:{frame['origin_id'].iloc[0]} samples={len(frame)}",
        )

    enriched = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    atomic_write_parquet(metrics_dir / "enriched_predictions.parquet", enriched)
    _write_summary_table(metrics_dir / "baseline_metrics_by_origin", origin_rows)

    condition_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    for scope, predictors in sorted(pool.items()):
        for predictor, raw_item in sorted(predictors.items()):
            item = _stack_pool(raw_item)
            evaluated = _evaluate(
                item["truth_mu"], item["predicted_mu"], item["truth_resources"],
                item["predicted_resources"], names, condition.num_states,
            )
            flat = _flatten(evaluated)
            condition_rows.append(
                {**prefix, "split_scope": scope, "predictor": predictor,
                 "n_samples": int(item["truth_mu"].shape[0]), **flat}
            )
            truth_class = np.argmax(item["truth_mu"], axis=1)
            transition_mask = truth_class != np.argmax(
                item["transition_reference_mu"], axis=1
            )
            for subset_name, selected in (
                ("transition", transition_mask), ("stationary", ~transition_mask)
            ):
                if np.any(selected):
                    subset = _evaluate(
                        item["truth_mu"][selected], item["predicted_mu"][selected],
                        item["truth_resources"][selected], item["predicted_resources"][selected],
                        names, condition.num_states,
                    )
                    transition_rows.append(
                        {**prefix, "split_scope": scope, "predictor": predictor,
                         "subset": subset_name, "n_samples": int(np.sum(selected)),
                         "subset_fraction": float(np.mean(selected)), **_flatten(subset)}
                    )
            membership_l1 = np.sum(
                np.abs(item["predicted_mu"] - item["truth_mu"]), axis=1
            )
            for q, value in zip(quantiles, np.quantile(membership_l1, quantiles)):
                quantile_rows.append(
                    {**prefix, "split_scope": scope, "predictor": predictor,
                     "metric": "membership_l1", "resource": "all", "quantile": q,
                     "value": float(value), "n_samples": int(membership_l1.size)}
                )
            absolute = np.abs(item["predicted_resources"] - item["truth_resources"])
            for resource_index, resource_name in enumerate(names):
                values = absolute[:, resource_index]
                for q, value in zip(quantiles, np.quantile(values, quantiles)):
                    quantile_rows.append(
                        {**prefix, "split_scope": scope, "predictor": predictor,
                         "metric": "resource_absolute_error", "resource": resource_name,
                         "quantile": q, "value": float(value), "n_samples": int(values.size)}
                    )

    for row in condition_rows:
        if row["predictor"] == "model" and row["split_scope"] in pool:
            persistence = next(
                (item for item in condition_rows
                 if item["split_scope"] == row["split_scope"] and item["predictor"] == "persistence"),
                None,
            )
            if persistence:
                row["gain_vs_persistence.accuracy"] = (
                    row.get("classification.accuracy", np.nan)
                    - persistence.get("classification.accuracy", np.nan)
                )
                row["gain_vs_persistence.f1_weighted"] = (
                    row.get("classification.f1_weighted", np.nan)
                    - persistence.get("classification.f1_weighted", np.nan)
                )
                row["gain_vs_persistence.resource_mae_reduction"] = (
                    persistence.get("resources.overall.mae", np.nan)
                    - row.get("resources.overall.mae", np.nan)
                )

    _write_summary_table(metrics_dir / "baseline_metrics_by_condition", condition_rows)
    _write_summary_table(metrics_dir / "transition_metrics", transition_rows)
    _write_summary_table(metrics_dir / "error_quantiles", quantile_rows)
    manifest = {
        "status": "complete",
        "input_signature": signature,
        "condition_id": condition.condition_id,
        "prediction_fragments": len(fragments),
        "per_sample_rows": total_samples,
        "baseline_origin_rows": len(origin_rows),
        "baseline_condition_rows": len(condition_rows),
        "transition_rows": len(transition_rows),
        "error_quantile_rows": len(quantile_rows),
        "predictors": ["model", "persistence", "daily_seasonal"],
        "membership_label_uses_one_bucket": True,
        "history_changes_label_definition": False,
        "forecast_feedback": False,
        "output_files": [str(path.relative_to(run_dir)) for path in (*required,)],
    }
    atomic_write_json(completion, manifest)
    tracker.complete(summary=manifest, message=f"rows={total_samples}")
    return manifest


def enrich_run(
    config: ExperimentConfig,
    run_dir: Path,
    condition: Condition,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Run enrichment and leave a terminal progress state on every failure."""

    try:
        return _enrich_run_impl(config, run_dir, condition, resume=resume)
    except BaseException as exc:
        tracker = SensitivityProgressTracker(
            run_dir / "progress-enrichment",
            run_id=run_dir.name + "-enrichment",
            training_epochs=1,
            heartbeat_interval_seconds=float(
                config.runtime.get("progress_interval_seconds", 30)
            ),
        )
        tracker.fail(exc, context={"condition_id": condition.condition_id})
        raise


__all__ = ["enrich_run"]
