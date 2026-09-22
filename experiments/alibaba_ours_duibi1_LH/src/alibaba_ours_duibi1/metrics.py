from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import TaskSpec
from .data import PreparedData
from .utils import atomic_save_npz, atomic_write_csv, atomic_write_json, json_safe


EPS = 1e-12


def _ratio(numerator: Any, denominator: Any) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    output = np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape))
    return np.divide(
        numerator_array,
        denominator_array,
        out=output,
        where=denominator_array != 0,
    )


def _nanmean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(values)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def _classification_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = _ratio(true_positive, predicted_support)
    recall = _ratio(true_positive, true_support)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    supported = true_support > 0
    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else math.nan
    balanced = float(recall[supported].mean()) if supported.any() else math.nan
    macro_precision = float(precision[supported].mean()) if supported.any() else math.nan
    macro_recall = float(recall[supported].mean()) if supported.any() else math.nan
    macro_f1 = float(f1[supported].mean()) if supported.any() else math.nan
    weights = _ratio(true_support, true_support.sum())
    weighted_precision = float(np.sum(weights * precision))
    weighted_recall = float(np.sum(weights * recall))
    weighted_f1 = float(np.sum(weights * f1))
    if total:
        expected = float(np.dot(true_support, predicted_support) / (total * total))
        kappa = (accuracy - expected) / (1.0 - expected) if expected < 1.0 else math.nan
        numerator = float(true_positive.sum() * total - np.dot(true_support, predicted_support))
        denominator = math.sqrt(
            max(total * total - float(np.dot(predicted_support, predicted_support)), 0.0)
            * max(total * total - float(np.dot(true_support, true_support)), 0.0)
        )
        mcc = numerator / denominator if denominator > 0 else math.nan
    else:
        kappa = math.nan
        mcc = math.nan
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_micro": accuracy,
        "precision_macro": macro_precision,
        "precision_weighted": weighted_precision,
        "recall_micro": accuracy,
        "recall_macro": macro_recall,
        "recall_weighted": weighted_recall,
        "f1_micro": accuracy,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "true_support": true_support,
        "predicted_support": predicted_support,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
    }


def _calibration(
    true_hard: np.ndarray,
    predicted: np.ndarray,
    bins: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    predicted_hard = np.argmax(predicted, axis=1)
    confidence = np.max(predicted, axis=1)
    correct = (true_hard == predicted_hard).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, bins - 1)
    index = np.maximum(index, 0)
    counts = np.bincount(index, minlength=bins).astype(np.int64)
    confidence_sum = np.bincount(index, weights=confidence, minlength=bins)
    accuracy_sum = np.bincount(index, weights=correct, minlength=bins)
    mean_confidence = _ratio(confidence_sum, counts)
    accuracy = _ratio(accuracy_sum, counts)
    gaps = np.abs(mean_confidence - accuracy)
    ece = float(np.sum(gaps * counts) / max(int(counts.sum()), 1))
    mce = float(np.max(gaps[counts > 0])) if np.any(counts > 0) else math.nan
    selected_probability = predicted[np.arange(len(true_hard)), true_hard]
    onehot = np.eye(predicted.shape[1], dtype=np.float64)[true_hard]
    entropy = -np.sum(predicted * np.log(predicted.clip(min=EPS)), axis=1)
    result = {
        "ece": ece,
        "mce": mce,
        "nll": float(-np.mean(np.log(selected_probability.clip(min=EPS)))),
        "brier_onehot": float(np.mean(np.sum((predicted - onehot) ** 2, axis=1))),
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_normalized_entropy": float(np.mean(entropy) / math.log(predicted.shape[1])),
    }
    return result, counts, mean_confidence, accuracy


def _membership(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = true / true.sum(axis=1, keepdims=True).clip(min=EPS)
    predicted = predicted / predicted.sum(axis=1, keepdims=True).clip(min=EPS)
    difference = predicted - true
    absolute = np.abs(difference)
    squared = difference**2
    midpoint = 0.5 * (true + predicted)
    kl = np.sum(true * np.log((true + EPS) / (predicted + EPS)), axis=1)
    js = 0.5 * np.sum(true * np.log((true + EPS) / (midpoint + EPS)), axis=1)
    js += 0.5 * np.sum(predicted * np.log((predicted + EPS) / (midpoint + EPS)), axis=1)
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(true) - np.sqrt(predicted)) ** 2, axis=1))
    cosine = np.sum(true * predicted, axis=1) / (
        np.linalg.norm(true, axis=1) * np.linalg.norm(predicted, axis=1)
    ).clip(min=EPS)
    true_entropy = -np.sum(true * np.log(true.clip(min=EPS)), axis=1)
    predicted_entropy = -np.sum(predicted * np.log(predicted.clip(min=EPS)), axis=1)
    return {
        "mae_mu": float(absolute.mean()),
        "rmse_mu": float(np.sqrt(squared.mean())),
        "brier_soft": float(np.mean(np.sum(squared, axis=1))),
        "l1_mu": float(np.mean(np.sum(absolute, axis=1))),
        "l2_mu": float(np.mean(np.sqrt(np.sum(squared, axis=1)))),
        "total_variation": float(np.mean(0.5 * np.sum(absolute, axis=1))),
        "kl_true_pred": float(np.mean(kl)),
        "js_divergence": float(np.mean(js)),
        "hellinger": float(np.mean(hellinger)),
        "cosine_similarity": float(np.mean(cosine)),
        "soft_cross_entropy": float(
            np.mean(-np.sum(true * np.log(predicted.clip(min=EPS)), axis=1))
        ),
        "mean_true_entropy": float(np.mean(true_entropy)),
        "mean_predicted_entropy": float(np.mean(predicted_entropy)),
    }


def _resource_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    difference = predicted - true
    absolute = np.abs(difference)
    squared = difference**2
    nonzero = np.abs(true) > 1e-8
    mape = float(np.mean(absolute[nonzero] / np.abs(true[nonzero])) * 100) if nonzero.any() else math.nan
    smape_denominator = np.abs(true) + np.abs(predicted)
    smape_active = smape_denominator > 1e-8
    smape = (
        float(np.mean(2.0 * absolute[smape_active] / smape_denominator[smape_active]) * 100)
        if smape_active.any()
        else math.nan
    )
    centered = true - true.mean()
    ss_total = float(np.sum(centered**2))
    ss_residual = float(np.sum(squared))
    if true.size > 1 and np.std(true) > 0 and np.std(predicted) > 0:
        pearson = float(np.corrcoef(true.reshape(-1), predicted.reshape(-1))[0, 1])
    else:
        pearson = math.nan
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "median_absolute_error": float(np.median(absolute)),
        "max_absolute_error": float(np.max(absolute)),
        "bias": float(np.mean(difference)),
        "mape_pct": mape,
        "smape_pct": smape,
        "wape_pct": float(np.sum(absolute) / max(float(np.sum(np.abs(true))), EPS) * 100),
        "r2": 1.0 - ss_residual / ss_total if ss_total > 0 else math.nan,
        "pearson_r": pearson,
    }


def _bootstrap_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    origins, horizons = numerator.shape
    point = _ratio(numerator.sum(axis=0), denominator.sum(axis=0))
    if origins <= 1 or samples <= 1:
        return point, point
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, horizons), dtype=np.float64)
    for offset in range(samples):
        chosen = rng.integers(0, origins, size=origins)
        estimates[offset] = _ratio(
            numerator[chosen].sum(axis=0), denominator[chosen].sum(axis=0)
        )
    return np.nanpercentile(estimates, 2.5, axis=0), np.nanpercentile(estimates, 97.5, axis=0)


def _write_parquet_optional(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            pd.DataFrame([json_safe(row) for row in rows]).to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        # CSV and NPZ are mandatory; Parquet is a best-effort convenience.
        return


def write_metric_bundle(
    *,
    destination: Path,
    task: TaskSpec,
    split: str,
    protocol: str,
    origins: Sequence[int],
    predictions: np.ndarray,
    data: PreparedData,
    calibration_bins: int,
    bootstrap_samples: int,
    seed: int,
    efficiency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    origins_array = np.asarray(origins, dtype=np.int64)
    horizon = task.horizon.steps
    expected = (len(origins), data.workloads, horizon, data.num_states)
    if predictions.shape != expected:
        raise ValueError(f"prediction shape must be {expected}, got {predictions.shape}")
    truth = np.stack(
        [data.target_memberships[:, origin + 1 : origin + horizon + 1] for origin in origins],
        axis=0,
    ).astype(np.float64)
    valid = np.stack(
        [data.target_mask[:, origin + 1 : origin + horizon + 1] for origin in origins],
        axis=0,
    ).astype(bool)
    observed = np.stack(
        [data.observed_values[:, origin + 1 : origin + horizon + 1] for origin in origins],
        axis=0,
    ).astype(np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    predicted = np.clip(predicted, EPS, None)
    predicted /= predicted.sum(axis=-1, keepdims=True).clip(min=EPS)
    truth /= truth.sum(axis=-1, keepdims=True).clip(min=EPS)
    true_hard = np.argmax(truth, axis=-1)
    predicted_hard = np.argmax(predicted, axis=-1)
    correct = (true_hard == predicted_hard) & valid
    sample_mae = np.mean(np.abs(predicted - truth), axis=-1)
    sample_brier = np.sum((predicted - truth) ** 2, axis=-1)
    predicted_resource_normalized = np.einsum(
        "onhk,kr->onhr", predicted, data.centroids_normalized.astype(np.float64)
    )
    predicted_resource = (
        predicted_resource_normalized * data.resource_std[None, None, None, :]
        + data.resource_mean[None, None, None, :]
    )
    sample_resource_mae = np.mean(np.abs(predicted_resource - observed), axis=-1)

    valid_count_origin = valid.sum(axis=1).astype(np.float64)
    correct_count_origin = correct.sum(axis=1).astype(np.float64)
    mae_sum_origin = np.where(valid, sample_mae, 0.0).sum(axis=1)
    brier_sum_origin = np.where(valid, sample_brier, 0.0).sum(axis=1)
    resource_mae_sum_origin = np.where(valid, sample_resource_mae, 0.0).sum(axis=1)
    acc_low, acc_high = _bootstrap_interval(
        correct_count_origin,
        valid_count_origin,
        samples=bootstrap_samples,
        seed=seed,
    )
    mae_low, mae_high = _bootstrap_interval(
        mae_sum_origin,
        valid_count_origin,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    resource_low, resource_high = _bootstrap_interval(
        resource_mae_sum_origin,
        valid_count_origin,
        samples=bootstrap_samples,
        seed=seed + 2,
    )

    confusion = np.zeros((horizon, data.num_states, data.num_states), dtype=np.int64)
    calibration_count = np.zeros((horizon, calibration_bins), dtype=np.int64)
    calibration_confidence = np.zeros((horizon, calibration_bins), dtype=np.float64)
    calibration_accuracy = np.zeros((horizon, calibration_bins), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for h_index in range(horizon):
        active = valid[:, :, h_index]
        sample_count = int(active.sum())
        base = {
            **task.as_dict(),
            "split": split,
            "origin_protocol": protocol,
            "h": h_index + 1,
            "lead_minutes": h_index + 1,
            "lead_hours": (h_index + 1) / 60.0,
            "lead_days": (h_index + 1) / 1440.0,
            "display_x": (h_index + 1) / task.horizon.display_divisor,
            "n_origins": len(origins),
            "n_samples": sample_count,
        }
        if sample_count == 0:
            rows.append(base)
            continue
        true_selected = true_hard[:, :, h_index][active]
        predicted_selected_hard = predicted_hard[:, :, h_index][active]
        matrix = np.bincount(
            true_selected * data.num_states + predicted_selected_hard,
            minlength=data.num_states * data.num_states,
        ).reshape(data.num_states, data.num_states)
        confusion[h_index] = matrix
        classification = _classification_from_confusion(matrix)
        true_selected_mu = truth[:, :, h_index][active]
        predicted_selected_mu = predicted[:, :, h_index][active]
        membership = _membership(true_selected_mu, predicted_selected_mu)
        calibration, counts, mean_conf, bin_acc = _calibration(
            true_selected, predicted_selected_mu, calibration_bins
        )
        calibration_count[h_index] = counts
        calibration_confidence[h_index] = mean_conf
        calibration_accuracy[h_index] = bin_acc
        true_resource = observed[:, :, h_index][active]
        predicted_selected_resource = predicted_resource[:, :, h_index][active]
        resources = _resource_metrics(true_resource, predicted_selected_resource)
        row = {
            **base,
            **{f"classification.{key}": value for key, value in classification.items() if not isinstance(value, np.ndarray)},
            **{f"membership.{key}": value for key, value in membership.items()},
            "membership.dominant_state_accuracy": classification["accuracy"],
            **{f"calibration.{key}": value for key, value in calibration.items()},
            **{f"resources.overall.{key}": value for key, value in resources.items()},
            "ci95.accuracy_low": acc_low[h_index],
            "ci95.accuracy_high": acc_high[h_index],
            "ci95.membership_mae_low": mae_low[h_index],
            "ci95.membership_mae_high": mae_high[h_index],
            "ci95.resource_mae_low": resource_low[h_index],
            "ci95.resource_mae_high": resource_high[h_index],
        }
        for resource_index, resource_name in enumerate(data.resource_names):
            resource_result = _resource_metrics(
                true_resource[:, resource_index],
                predicted_selected_resource[:, resource_index],
            )
            row.update(
                {
                    f"resources.{resource_name}.{key}": value
                    for key, value in resource_result.items()
                }
            )
        rows.append(row)
        for state in range(data.num_states):
            per_class_rows.append(
                {
                    **base,
                    "state_id": state,
                    "support": classification["true_support"][state],
                    "predicted_support": classification["predicted_support"][state],
                    "true_positive": matrix[state, state],
                    "precision": classification["precision_per_class"][state],
                    "recall": classification["recall_per_class"][state],
                    "f1": classification["f1_per_class"][state],
                }
            )

    if rows:
        reference_accuracy = rows[0].get("classification.accuracy")
        reference_membership_mae = rows[0].get("membership.mae_mu")
        reference_resource_mae = rows[0].get("resources.overall.mae")
        for row in rows:
            accuracy_value = row.get("classification.accuracy")
            membership_value = row.get("membership.mae_mu")
            resource_value = row.get("resources.overall.mae")
            if reference_accuracy is not None and accuracy_value is not None:
                row["degradation.accuracy_absolute"] = reference_accuracy - accuracy_value
                row["degradation.accuracy_pct"] = (
                    (reference_accuracy - accuracy_value) / reference_accuracy * 100
                    if reference_accuracy
                    else math.nan
                )
            if reference_membership_mae is not None and membership_value is not None:
                row["degradation.membership_mae_increase"] = (
                    membership_value - reference_membership_mae
                )
            if reference_resource_mae is not None and resource_value is not None:
                row["degradation.resource_mae_increase"] = (
                    resource_value - reference_resource_mae
                )

    prefix = f"{split}-{protocol}"
    per_h_csv = destination / f"{prefix}-metrics_by_h.csv"
    per_class_csv = destination / f"{prefix}-per_class.csv"
    atomic_write_csv(per_h_csv, rows)
    atomic_write_csv(per_class_csv, per_class_rows)
    _write_parquet_optional(destination / f"{prefix}-metrics_by_h.parquet", rows)
    _write_parquet_optional(destination / f"{prefix}-per_class.parquet", per_class_rows)

    per_origin_accuracy = _ratio(correct_count_origin, valid_count_origin)
    per_origin_mae = _ratio(mae_sum_origin, valid_count_origin)
    per_origin_brier = _ratio(brier_sum_origin, valid_count_origin)
    per_origin_resource_mae = _ratio(resource_mae_sum_origin, valid_count_origin)
    valid_count_workload = valid.sum(axis=0).astype(np.float64)
    per_workload_accuracy = _ratio(correct.sum(axis=0), valid_count_workload)
    per_workload_mae = _ratio(
        np.where(valid, sample_mae, 0.0).sum(axis=0), valid_count_workload
    )
    per_workload_brier = _ratio(
        np.where(valid, sample_brier, 0.0).sum(axis=0), valid_count_workload
    )
    per_workload_resource_mae = _ratio(
        np.where(valid, sample_resource_mae, 0.0).sum(axis=0), valid_count_workload
    )
    arrays_path = atomic_save_npz(
        destination / f"{prefix}-metric_arrays.npz",
        compressed=True,
        origins=origins_array,
        workload_ids=data.workload_ids,
        h=np.arange(1, horizon + 1, dtype=np.int32),
        confusion_matrices=confusion,
        calibration_count=calibration_count,
        calibration_mean_confidence=calibration_confidence,
        calibration_accuracy=calibration_accuracy,
        per_origin_valid_count=valid_count_origin,
        per_origin_accuracy=per_origin_accuracy,
        per_origin_membership_mae=per_origin_mae,
        per_origin_brier_soft=per_origin_brier,
        per_origin_resource_mae=per_origin_resource_mae,
        per_workload_valid_count=valid_count_workload,
        per_workload_accuracy=per_workload_accuracy,
        per_workload_membership_mae=per_workload_mae,
        per_workload_brier_soft=per_workload_brier,
        per_workload_resource_mae=per_workload_resource_mae,
    )
    valid_rows = [row for row in rows if row.get("n_samples", 0)]
    accuracy_values = np.asarray(
        [row.get("classification.accuracy", np.nan) for row in valid_rows], dtype=float
    )
    summary = {
        **task.as_dict(),
        "split": split,
        "origin_protocol": protocol,
        "origins": list(map(int, origins)),
        "origin_count": len(origins),
        "horizon_steps": horizon,
        "metrics_by_h_csv": per_h_csv,
        "per_class_csv": per_class_csv,
        "metric_arrays_npz": arrays_path,
        "mean_accuracy_across_h": float(np.nanmean(accuracy_values)) if accuracy_values.size else None,
        "accuracy_h1": rows[0].get("classification.accuracy") if rows else None,
        "accuracy_hmax": rows[-1].get("classification.accuracy") if rows else None,
        "efficiency": dict(efficiency or {}),
    }
    atomic_write_json(destination / f"{prefix}-summary.json", summary)
    return summary


__all__ = ["write_metric_bundle"]
