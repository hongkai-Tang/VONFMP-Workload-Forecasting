"""Metric computation and durable artifact writers for the Alibaba experiment.

The module intentionally has no scikit-learn dependency.  All public metric
functions accept NumPy-compatible inputs and return ordinary dictionaries or
lists of dictionaries so that callers can write stable long-form tables.

Terminology is deliberately strict:

* ``h`` is the evaluated forecast horizon (in resampled time steps).
* fuzzy-membership metrics compare two simplex-valued vectors directly.
* calibration metrics use the dominant fuzzy state as a categorical target.

Parquet support is loaded lazily.  CSV, JSON and NPZ writers are always
available and use a temporary file in the destination directory followed by
``os.replace`` for atomic publication.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


EPS = 1e-12


class MetricInputError(ValueError):
    """Raised when metric inputs are inconsistent or invalid."""


def _as_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise MetricInputError(f"{name} must be one-dimensional; got {array.shape}")
    if array.size == 0:
        raise MetricInputError(f"{name} must not be empty")
    return array


def _as_2d(values: Any, name: str, dtype: Any = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise MetricInputError(f"{name} must be one- or two-dimensional; got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise MetricInputError(f"{name} must not be empty")
    return array


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_compatible(value: Any) -> Any:
    """Return strict-JSON data, representing undefined metrics as null."""

    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    value = _python_scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(denominator) or abs(float(denominator)) <= EPS:
        return float(default)
    return float(numerator) / float(denominator)


def validate_simplex(
    values: Any,
    *,
    name: str = "probabilities",
    atol: float = 1e-6,
    normalize: bool = False,
) -> np.ndarray:
    """Validate a finite, non-negative row-wise simplex.

    ``normalize=True`` is an explicit repair mode.  Zero-mass rows still fail
    because silently turning a missing row into a uniform state would violate
    the experiment's missing-data contract.
    """

    array = _as_2d(values, name, dtype=float)
    if not np.all(np.isfinite(array)):
        raise MetricInputError(f"{name} contains NaN or infinity")
    if np.any(array < -atol):
        minimum = float(np.min(array))
        raise MetricInputError(f"{name} contains negative values (minimum={minimum})")
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= atol):
        raise MetricInputError(f"{name} contains a zero-mass row")
    if normalize:
        return array / totals
    if not np.allclose(totals, 1.0, atol=atol, rtol=0.0):
        deviation = float(np.max(np.abs(totals - 1.0)))
        raise MetricInputError(f"{name} rows do not sum to one (max deviation={deviation})")
    return array


def resolve_labels(y_true: Any, y_pred: Any, labels: Sequence[Any] | None = None) -> np.ndarray:
    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise MetricInputError("y_true and y_pred must have the same shape")
    if labels is None:
        try:
            return np.unique(np.concatenate([truth, prediction]))
        except TypeError:
            # Mixed, non-orderable Python label types are uncommon but valid.
            ordered: list[Any] = []
            for value in np.concatenate([truth, prediction]).tolist():
                if value not in ordered:
                    ordered.append(value)
            return np.asarray(ordered, dtype=object)
    resolved = np.asarray(list(labels))
    if resolved.ndim != 1 or resolved.size == 0:
        raise MetricInputError("labels must be a non-empty one-dimensional sequence")
    if len({_python_scalar(v) for v in resolved}) != resolved.size:
        raise MetricInputError("labels must be unique")
    return resolved


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Sequence[Any] | None = None,
    sample_weight: Any | None = None,
) -> np.ndarray:
    """Return a rows=true, columns=predicted confusion matrix."""

    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    resolved = resolve_labels(truth, prediction, labels)
    label_to_index = {_python_scalar(label): index for index, label in enumerate(resolved)}
    try:
        true_index = np.asarray([label_to_index[_python_scalar(v)] for v in truth], dtype=int)
        pred_index = np.asarray([label_to_index[_python_scalar(v)] for v in prediction], dtype=int)
    except KeyError as exc:
        raise MetricInputError(f"observed label is absent from labels: {exc.args[0]!r}") from exc

    if sample_weight is None:
        weights = np.ones(truth.size, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != truth.shape:
            raise MetricInputError("sample_weight must have the same shape as y_true")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise MetricInputError("sample_weight must be finite and non-negative")
    matrix = np.zeros((resolved.size, resolved.size), dtype=float)
    np.add.at(matrix, (true_index, pred_index), weights)
    return matrix


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[Any] | None = None,
    sample_weight: Any | None = None,
    zero_division: float = 0.0,
) -> dict[str, Any]:
    """Compute multiclass classification, agreement and per-class metrics."""

    truth = _as_1d(y_true, "y_true")
    prediction = _as_1d(y_pred, "y_pred")
    resolved = resolve_labels(truth, prediction, labels)
    matrix = confusion_matrix(truth, prediction, resolved, sample_weight)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    false_positive = predicted - true_positive
    false_negative = support - true_positive

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.full_like(true_positive, float(zero_division)),
        where=(true_positive + false_positive) > EPS,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.full_like(true_positive, float(zero_division)),
        where=(true_positive + false_negative) > EPS,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.full_like(precision, float(zero_division)),
        where=(precision + recall) > EPS,
    )

    total = float(matrix.sum())
    correct = float(true_positive.sum())
    accuracy = _safe_ratio(correct, total, default=float("nan"))
    supported = support > EPS
    balanced_accuracy = float(np.mean(recall[supported])) if np.any(supported) else float("nan")
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    weighted_precision = _safe_ratio(float(np.dot(precision, support)), float(support.sum()))
    weighted_recall = _safe_ratio(float(np.dot(recall, support)), float(support.sum()))
    weighted_f1 = _safe_ratio(float(np.dot(f1, support)), float(support.sum()))

    micro_tp = correct
    micro_fp = float(false_positive.sum())
    micro_fn = float(false_negative.sum())
    micro_precision = _safe_ratio(micro_tp, micro_tp + micro_fp, default=float(zero_division))
    micro_recall = _safe_ratio(micro_tp, micro_tp + micro_fn, default=float(zero_division))
    micro_f1 = _safe_ratio(
        2.0 * micro_precision * micro_recall,
        micro_precision + micro_recall,
        default=float(zero_division),
    )

    # Gorodkin's multiclass Matthews correlation coefficient.
    sum_product = float(np.dot(predicted, support))
    mcc_numerator = correct * total - sum_product
    mcc_denominator = math.sqrt(
        max(total * total - float(np.dot(predicted, predicted)), 0.0)
        * max(total * total - float(np.dot(support, support)), 0.0)
    )
    mcc = _safe_ratio(mcc_numerator, mcc_denominator, default=float("nan"))

    expected_agreement = _safe_ratio(sum_product, total * total, default=float("nan"))
    kappa = _safe_ratio(accuracy - expected_agreement, 1.0 - expected_agreement, default=float("nan"))

    per_class: list[dict[str, Any]] = []
    for index, label in enumerate(resolved):
        per_class.append(
            {
                "state_id": _python_scalar(label),
                "support": float(support[index]),
                "predicted_support": float(predicted[index]),
                "true_positive": float(true_positive[index]),
                "false_positive": float(false_positive[index]),
                "false_negative": float(false_negative[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )

    return {
        "n_samples": float(total),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_micro": micro_precision,
        "precision_macro": macro_precision,
        "precision_weighted": weighted_precision,
        "recall_micro": micro_recall,
        "recall_macro": macro_recall,
        "recall_weighted": weighted_recall,
        "f1_micro": micro_f1,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "labels": [_python_scalar(v) for v in resolved],
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def membership_metrics(
    true_membership: Any,
    predicted_membership: Any,
    *,
    normalize: bool = False,
    include_per_state: bool = True,
) -> dict[str, Any]:
    """Compute distances and proper scores between fuzzy memberships."""

    truth = validate_simplex(true_membership, name="true_membership", normalize=normalize)
    prediction = validate_simplex(
        predicted_membership, name="predicted_membership", normalize=normalize
    )
    if truth.shape != prediction.shape:
        raise MetricInputError("membership matrices must have the same shape")

    difference = prediction - truth
    absolute = np.abs(difference)
    squared = difference * difference
    clipped_truth = np.clip(truth, EPS, 1.0)
    clipped_prediction = np.clip(prediction, EPS, 1.0)
    midpoint = 0.5 * (clipped_truth + clipped_prediction)
    kl_truth_pred = np.sum(clipped_truth * np.log(clipped_truth / clipped_prediction), axis=1)
    kl_truth_mid = np.sum(clipped_truth * np.log(clipped_truth / midpoint), axis=1)
    kl_pred_mid = np.sum(clipped_prediction * np.log(clipped_prediction / midpoint), axis=1)
    js = 0.5 * (kl_truth_mid + kl_pred_mid)
    cosine_denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    cosine = np.divide(
        np.sum(truth * prediction, axis=1),
        cosine_denominator,
        out=np.zeros(truth.shape[0], dtype=float),
        where=cosine_denominator > EPS,
    )
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(truth) - np.sqrt(prediction)) ** 2, axis=1))
    soft_cross_entropy = -np.sum(truth * np.log(clipped_prediction), axis=1)

    result: dict[str, Any] = {
        "n_samples": int(truth.shape[0]),
        "n_states": int(truth.shape[1]),
        "mae_mu": float(np.mean(absolute)),
        "rmse_mu": float(np.sqrt(np.mean(squared))),
        "brier_soft": float(np.mean(np.sum(squared, axis=1))),
        "l1_mu": float(np.mean(np.sum(absolute, axis=1))),
        "l2_mu": float(np.mean(np.sqrt(np.sum(squared, axis=1)))),
        "total_variation": float(np.mean(0.5 * np.sum(absolute, axis=1))),
        "kl_true_pred": float(np.mean(kl_truth_pred)),
        "js_divergence": float(np.mean(js)),
        "hellinger": float(np.mean(hellinger)),
        "cosine_similarity": float(np.mean(cosine)),
        "soft_cross_entropy": float(np.mean(soft_cross_entropy)),
        "dominant_state_accuracy": float(
            np.mean(np.argmax(truth, axis=1) == np.argmax(prediction, axis=1))
        ),
        "mean_true_entropy": float(np.mean(-np.sum(truth * np.log(clipped_truth), axis=1))),
        "mean_predicted_entropy": float(
            np.mean(-np.sum(prediction * np.log(clipped_prediction), axis=1))
        ),
    }
    if include_per_state:
        result["per_state"] = [
            {
                "state_id": int(state),
                "mae_mu": float(np.mean(absolute[:, state])),
                "rmse_mu": float(np.sqrt(np.mean(squared[:, state]))),
                "bias_mu": float(np.mean(difference[:, state])),
                "true_mean_membership": float(np.mean(truth[:, state])),
                "predicted_mean_membership": float(np.mean(prediction[:, state])),
            }
            for state in range(truth.shape[1])
        ]
    return result


def _bin_edges(confidence: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if n_bins < 2:
        raise MetricInputError("n_bins must be at least 2")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if edges.size < 2:
            return np.asarray([0.0, 1.0])
        return edges
    raise MetricInputError("strategy must be 'uniform' or 'quantile'")


def _binary_ece(target: np.ndarray, probability: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(probability, edges[1:-1], right=True)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(target[mask])) - float(np.mean(probability[mask])))
    return float(ece)


def calibration_metrics(
    y_true: Any,
    predicted_probability: Any,
    *,
    labels: Sequence[Any] | None = None,
    n_bins: int = 15,
    strategy: str = "uniform",
    normalize: bool = False,
) -> dict[str, Any]:
    """Compute dominant-state calibration, NLL and multiclass Brier score."""

    truth = _as_1d(y_true, "y_true")
    probability = validate_simplex(
        predicted_probability, name="predicted_probability", normalize=normalize
    )
    if truth.shape[0] != probability.shape[0]:
        raise MetricInputError("y_true and predicted_probability must contain the same samples")
    resolved = np.arange(probability.shape[1]) if labels is None else np.asarray(list(labels))
    if resolved.size != probability.shape[1] or resolved.ndim != 1:
        raise MetricInputError("labels length must equal the probability column count")
    label_to_index = {_python_scalar(label): index for index, label in enumerate(resolved)}
    try:
        true_index = np.asarray([label_to_index[_python_scalar(v)] for v in truth], dtype=int)
    except KeyError as exc:
        raise MetricInputError(f"true label is absent from labels: {exc.args[0]!r}") from exc

    predicted_index = np.argmax(probability, axis=1)
    confidence = probability[np.arange(probability.shape[0]), predicted_index]
    correct = (predicted_index == true_index).astype(float)
    edges = _bin_edges(confidence, n_bins, strategy)
    assignments = np.digitize(confidence, edges[1:-1], right=True)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for bin_index in range(edges.size - 1):
        mask = assignments == bin_index
        count = int(np.sum(mask))
        if count:
            average_confidence = float(np.mean(confidence[mask]))
            accuracy = float(np.mean(correct[mask]))
            gap = abs(average_confidence - accuracy)
            ece += (count / probability.shape[0]) * gap
            mce = max(mce, gap)
        else:
            average_confidence = float("nan")
            accuracy = float("nan")
            gap = float("nan")
        bins.append(
            {
                "bin_index": int(bin_index),
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_confidence": average_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    one_hot = np.eye(probability.shape[1], dtype=float)[true_index]
    clipped = np.clip(probability, EPS, 1.0)
    entropy = -np.sum(probability * np.log(clipped), axis=1)
    normalization = math.log(probability.shape[1]) if probability.shape[1] > 1 else 1.0
    classwise = [
        _binary_ece((true_index == state).astype(float), probability[:, state], n_bins)
        for state in range(probability.shape[1])
    ]
    return {
        "n_samples": int(probability.shape[0]),
        "n_bins_requested": int(n_bins),
        "n_bins_effective": int(edges.size - 1),
        "binning_strategy": strategy,
        "ece": float(ece),
        "mce": float(mce),
        "classwise_ece": float(np.mean(classwise)),
        "classwise_ece_per_state": [float(v) for v in classwise],
        "brier_onehot": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(clipped[np.arange(probability.shape[0]), true_index]))),
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_normalized_entropy": float(np.mean(entropy / normalization)),
        "dominant_state_accuracy": float(np.mean(correct)),
        "labels": [_python_scalar(v) for v in resolved],
        "bins": bins,
    }


def calibration_from_memberships(
    true_membership: Any,
    predicted_membership: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    truth = validate_simplex(true_membership, name="true_membership")
    prediction = validate_simplex(predicted_membership, name="predicted_membership")
    if truth.shape != prediction.shape:
        raise MetricInputError("membership matrices must have the same shape")
    result = calibration_metrics(np.argmax(truth, axis=1), prediction, **kwargs)
    result["target_type"] = "dominant_fuzzy_state"
    return result


def _resource_vector_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    difference = prediction - truth
    absolute = np.abs(difference)
    squared = difference * difference
    nonzero = np.abs(truth) > EPS
    mape = float(np.mean(absolute[nonzero] / np.abs(truth[nonzero])) * 100.0) if np.any(nonzero) else float("nan")
    denominator = np.abs(truth) + np.abs(prediction)
    smape_mask = denominator > EPS
    smape = (
        float(np.mean(2.0 * absolute[smape_mask] / denominator[smape_mask]) * 100.0)
        if np.any(smape_mask)
        else float("nan")
    )
    centered = truth - float(np.mean(truth))
    sum_squares = float(np.sum(centered * centered))
    r2 = 1.0 - _safe_ratio(float(np.sum(squared)), sum_squares, default=float("nan"))
    truth_std = float(np.std(truth))
    prediction_std = float(np.std(prediction))
    if truth.size > 1 and truth_std > EPS and prediction_std > EPS:
        # Compute Pearson directly instead of np.corrcoef.  On Windows, calling
        # np.corrcoef after a PyTorch training pass can load a second Intel
        # OpenMP runtime from NumPy's linear-algebra backend.
        truth_centered = truth - float(np.mean(truth))
        prediction_centered = prediction - float(np.mean(prediction))
        covariance = float(np.mean(truth_centered * prediction_centered))
        pearson = covariance / (truth_std * prediction_std)
    else:
        pearson = float("nan")
    return {
        "n": int(truth.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "median_absolute_error": float(np.median(absolute)),
        "max_absolute_error": float(np.max(absolute)),
        "bias": float(np.mean(difference)),
        "mape_pct": mape,
        "smape_pct": smape,
        "wape_pct": _safe_ratio(float(np.sum(absolute)), float(np.sum(np.abs(truth))), default=float("nan")) * 100.0,
        "r2": float(r2),
        "pearson_r": pearson,
    }


def resource_metrics(
    true_resources: Any,
    predicted_resources: Any,
    *,
    resource_names: Sequence[str] | None = None,
    mask: Any | None = None,
) -> dict[str, Any]:
    """Compute finite-value-aware continuous resource prediction metrics."""

    truth = _as_2d(true_resources, "true_resources")
    prediction = _as_2d(predicted_resources, "predicted_resources")
    if truth.shape != prediction.shape:
        raise MetricInputError("resource matrices must have the same shape")
    if resource_names is None:
        names = [f"resource_{index}" for index in range(truth.shape[1])]
    else:
        names = [str(name) for name in resource_names]
        if len(names) != truth.shape[1]:
            raise MetricInputError("resource_names length must equal the resource column count")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.ndim == 1 and supplied.shape[0] == truth.shape[0]:
            supplied = np.repeat(supplied[:, None], truth.shape[1], axis=1)
        if supplied.shape != truth.shape:
            raise MetricInputError("mask must have shape [samples] or [samples, resources]")
        finite &= supplied

    per_resource: list[dict[str, Any]] = []
    for column, name in enumerate(names):
        valid = finite[:, column]
        if not np.any(valid):
            per_resource.append({"resource": name, "n": 0, "status": "insufficient_data"})
            continue
        metrics = _resource_vector_metrics(truth[valid, column], prediction[valid, column])
        per_resource.append({"resource": name, "status": "ok", **metrics})
    if np.any(finite):
        overall = {"status": "ok", **_resource_vector_metrics(truth[finite], prediction[finite])}
    else:
        overall = {"status": "insufficient_data", "n": 0}
    return {"overall": overall, "per_resource": per_resource}


def efficiency_metrics(
    *,
    n_samples: int | None = None,
    training_seconds: float | None = None,
    inference_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
    model_size_bytes: int | None = None,
    checkpoint_size_bytes: int | None = None,
) -> dict[str, float | int | None]:
    """Normalize resource-use measurements and derive latency/throughput."""

    result: dict[str, float | int | None] = {
        "n_samples": None if n_samples is None else int(n_samples),
        "training_seconds": None if training_seconds is None else float(training_seconds),
        "inference_seconds": None if inference_seconds is None else float(inference_seconds),
        "peak_memory_bytes": None if peak_memory_bytes is None else int(peak_memory_bytes),
        "peak_memory_mb": None if peak_memory_bytes is None else float(peak_memory_bytes) / (1024.0**2),
        "model_size_bytes": None if model_size_bytes is None else int(model_size_bytes),
        "model_size_mb": None if model_size_bytes is None else float(model_size_bytes) / (1024.0**2),
        "checkpoint_size_bytes": None if checkpoint_size_bytes is None else int(checkpoint_size_bytes),
    }
    if n_samples is not None and inference_seconds is not None and inference_seconds > 0.0:
        result["throughput_samples_per_second"] = float(n_samples) / float(inference_seconds)
        result["latency_ms_per_sample"] = 1000.0 * float(inference_seconds) / max(int(n_samples), 1)
    else:
        result["throughput_samples_per_second"] = None
        result["latency_ms_per_sample"] = None
    return result


def _scalar_metrics(result: Mapping[str, Any], *, excluded: set[str] | None = None) -> dict[str, float]:
    skip = excluded or set()
    scalars: dict[str, float] = {}
    for key, value in result.items():
        if key in skip or isinstance(value, (Mapping, list, tuple, np.ndarray)) or value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalars[key] = float(value)
    return scalars


def evaluate_predictions(
    true_membership: Any,
    predicted_membership: Any,
    *,
    labels: Sequence[Any] | None = None,
    n_calibration_bins: int = 15,
    true_resources: Any | None = None,
    predicted_resources: Any | None = None,
    resource_names: Sequence[str] | None = None,
    resource_mask: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one aligned prediction batch using all applicable families."""

    truth = validate_simplex(true_membership, name="true_membership")
    prediction = validate_simplex(predicted_membership, name="predicted_membership")
    if truth.shape != prediction.shape:
        raise MetricInputError("membership matrices must have the same shape")
    y_true = np.argmax(truth, axis=1)
    y_pred = np.argmax(prediction, axis=1)
    result = {
        "classification": classification_metrics(y_true, y_pred, labels=labels),
        "membership": membership_metrics(truth, prediction),
        "calibration": calibration_from_memberships(
            truth, prediction, labels=labels, n_bins=n_calibration_bins
        ),
    }
    if (true_resources is None) != (predicted_resources is None):
        raise MetricInputError("true_resources and predicted_resources must be supplied together")
    if true_resources is not None:
        result["resources"] = resource_metrics(
            true_resources,
            predicted_resources,
            resource_names=resource_names,
            mask=resource_mask,
        )
    return result


def evaluate_by_h(
    true_membership: Any,
    predicted_membership: Any,
    h: Any,
    *,
    labels: Sequence[Any] | None = None,
    n_calibration_bins: int = 15,
) -> list[dict[str, Any]]:
    """Return one flat scalar metric record per observed ``h``."""

    truth = validate_simplex(true_membership, name="true_membership")
    prediction = validate_simplex(predicted_membership, name="predicted_membership")
    horizons = _as_1d(h, "h")
    if truth.shape != prediction.shape or horizons.shape[0] != truth.shape[0]:
        raise MetricInputError("memberships and h must align by sample")
    records: list[dict[str, Any]] = []
    for horizon in np.unique(horizons):
        selected = horizons == horizon
        evaluated = evaluate_predictions(
            truth[selected],
            prediction[selected],
            labels=labels,
            n_calibration_bins=n_calibration_bins,
        )
        record: dict[str, Any] = {"h": _python_scalar(horizon), "n_samples": int(np.sum(selected))}
        record.update(_scalar_metrics(evaluated["classification"], excluded={"n_samples"}))
        record.update(_scalar_metrics(evaluated["membership"], excluded={"n_samples", "n_states"}))
        record.update(_scalar_metrics(evaluated["calibration"], excluded={"n_samples"}))
        records.append(record)
    return sorted(records, key=lambda row: float(row["h"]))


def metric_direction(metric: str) -> str | None:
    """Return ``higher``/``lower`` for known metrics, otherwise ``None``."""

    name = metric.lower()
    lower_tokens = (
        "mae",
        "rmse",
        "error",
        "loss",
        "nll",
        "brier",
        "divergence",
        "ece",
        "mce",
        "hellinger",
        "latency",
        "seconds",
        "memory",
    )
    higher_tokens = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "kappa",
        "similarity",
        "throughput",
        "r2",
        "pearson",
    )
    if any(token in name for token in lower_tokens):
        return "lower"
    if any(token in name for token in higher_tokens):
        return "higher"
    return None


def horizon_degradation(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_h: float | int | None = None,
    directions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert wide per-``h`` records into long absolute/relative degradation rows."""

    if not records:
        return []
    ordered = sorted(records, key=lambda row: float(row["h"]))
    reference = ordered[0]
    if reference_h is not None:
        matches = [row for row in ordered if float(row["h"]) == float(reference_h)]
        if not matches:
            raise MetricInputError(f"reference_h={reference_h} is absent")
        reference = matches[0]
    reserved = {"h", "n_samples", "n_workloads", "seed", "run_id", "variant", "method"}
    output: list[dict[str, Any]] = []
    for row in ordered:
        for name, value in row.items():
            if name in reserved or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            direction = (directions or {}).get(name) or metric_direction(name)
            if direction not in {"higher", "lower"}:
                continue
            baseline = float(reference.get(name, float("nan")))
            current = float(value)
            if direction == "higher":
                degradation = baseline - current
            else:
                degradation = current - baseline
            relative = degradation / abs(baseline) * 100.0 if np.isfinite(baseline) and abs(baseline) > EPS else float("nan")
            output.append(
                {
                    "h": _python_scalar(row["h"]),
                    "reference_h": _python_scalar(reference["h"]),
                    "metric": name,
                    "metric_direction": direction,
                    "value": current,
                    "reference_value": baseline,
                    "absolute_degradation": float(degradation),
                    "degradation_pct": float(relative),
                    "n_samples": row.get("n_samples"),
                }
            )
    return output


def long_horizon_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Summarize trajectories with normalized AUC, slope and threshold time."""

    if not records:
        return []
    ordered = sorted(records, key=lambda row: float(row["h"]))
    x = np.asarray([float(row["h"]) for row in ordered], dtype=float)
    if metrics is None:
        candidates = set.intersection(
            *[
                {
                    key
                    for key, value in row.items()
                    if key != "h" and isinstance(value, (int, float, np.integer, np.floating))
                }
                for row in ordered
            ]
        )
        selected_metrics = sorted(name for name in candidates if metric_direction(name) is not None)
    else:
        selected_metrics = [str(name) for name in metrics]
    summaries: list[dict[str, Any]] = []
    for name in selected_metrics:
        y = np.asarray([float(row[name]) for row in ordered], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        xf, yf = x[finite], y[finite]
        if xf.size == 1 or float(np.max(xf) - np.min(xf)) <= EPS:
            normalized_auc = float(yf[0])
            slope = float("nan")
        else:
            normalized_auc = float(np.trapz(yf, xf) / (xf[-1] - xf[0]))
            slope = float(np.polyfit(xf, yf, 1)[0])
        threshold_h: float | None = None
        threshold = (thresholds or {}).get(name)
        direction = metric_direction(name)
        if threshold is not None and direction is not None:
            crossed = yf <= threshold if direction == "higher" else yf >= threshold
            if np.any(crossed):
                threshold_h = float(xf[np.flatnonzero(crossed)[0]])
        initial, final = float(yf[0]), float(yf[-1])
        degradation = initial - final if direction == "higher" else final - initial
        summaries.append(
            {
                "metric": name,
                "metric_direction": direction,
                "initial_h": float(xf[0]),
                "final_h": float(xf[-1]),
                "initial_value": initial,
                "final_value": final,
                "absolute_degradation": float(degradation),
                "degradation_pct": float(degradation / abs(initial) * 100.0) if abs(initial) > EPS else float("nan"),
                "normalized_auc": normalized_auc,
                "slope_per_h": slope,
                "threshold": None if threshold is None else float(threshold),
                "threshold_h": threshold_h,
                "n_h": int(xf.size),
            }
        )
    return summaries


def summary_statistics(
    values: Any,
    *,
    confidence: float = 0.95,
    bootstrap_replicates: int = 0,
    random_state: int = 7,
) -> dict[str, Any]:
    """Summarize repeated-run values without a SciPy dependency."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"status": "insufficient_data", "n": 0}
    if not 0.0 < confidence < 1.0:
        raise MetricInputError("confidence must be between zero and one")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    alpha = 1.0 - confidence
    if bootstrap_replicates > 0 and array.size > 1:
        rng = np.random.default_rng(random_state)
        indices = rng.integers(0, array.size, size=(int(bootstrap_replicates), array.size))
        bootstrap_mean = np.mean(array[indices], axis=1)
        ci_low, ci_high = np.quantile(bootstrap_mean, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci_method = "percentile_bootstrap"
    else:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        half_width = z * std / math.sqrt(array.size) if array.size > 1 else 0.0
        ci_low, ci_high = mean - half_width, mean + half_width
        ci_method = "normal"
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "status": "ok",
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_level": float(confidence),
        "ci_method": ci_method,
        "bootstrap_replicates": int(bootstrap_replicates),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path, suffix: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix or ".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw_path)


def _publish_temporary(temporary: Path, destination: Path) -> Path:
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return destination


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_compatible(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temporary(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _csv_cell(value: Any) -> Any:
    value = _python_scalar(value)
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        serializable = value.tolist() if isinstance(value, np.ndarray) else value
        return json.dumps(serializable, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    destination = Path(path)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        ordered: list[str] = []
        for row in materialized:
            for key in row:
                if key not in ordered:
                    ordered.append(str(key))
        fieldnames = ordered
    fields = [str(name) for name in fieldnames]
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in materialized:
                writer.writerow({field: _csv_cell(row.get(field)) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temporary(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_npz(
    path: str | Path,
    arrays: Mapping[str, Any] | None = None,
    *,
    compressed: bool = True,
    **named_arrays: Any,
) -> Path:
    destination = Path(path)
    payload = dict(arrays or {})
    payload.update(named_arrays)
    converted: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype == object:
            raise MetricInputError(f"NPZ array {name!r} has unsafe object dtype")
        converted[str(name)] = array
    temporary = _temporary_path(destination, suffix=".npz")
    try:
        with temporary.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **converted)
            else:
                np.savez(handle, **converted)
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_temporary(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_parquet(
    path: str | Path,
    data: Any,
    *,
    compression: str = "zstd",
) -> Path:
    """Atomically write a PyArrow/Pandas/table-like object as Parquet."""

    destination = Path(path)
    temporary = _temporary_path(destination, suffix=".parquet")
    try:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pyarrow; install the experiment dependencies"
            ) from exc

        if isinstance(data, pa.Table):
            table = data
        elif hasattr(data, "columns") and hasattr(data, "to_dict"):
            table = pa.Table.from_pandas(data, preserve_index=False)
        elif isinstance(data, Mapping):
            table = pa.table(data)
        else:
            table = pa.Table.from_pylist([dict(row) for row in data])
        pq.write_table(table, temporary, compression=compression)
        # Windows does not permit fsync on a read-only descriptor.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        return _publish_temporary(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_metrics_bundle(
    output_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    stem: str = "metrics_long",
    arrays: Mapping[str, Any] | None = None,
    write_parquet: bool = True,
) -> dict[str, Path]:
    """Write aligned CSV/Parquet records and optional NPZ arrays atomically."""

    directory = Path(output_dir)
    materialized = [dict(record) for record in records]
    written = {"csv": atomic_write_csv(directory / f"{stem}.csv", materialized)}
    if write_parquet:
        written["parquet"] = atomic_write_parquet(directory / f"{stem}.parquet", materialized)
    if arrays is not None:
        written["npz"] = atomic_write_npz(directory / f"{stem}.npz", arrays)
    return written


__all__ = [
    "MetricInputError",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_parquet",
    "calibration_from_memberships",
    "calibration_metrics",
    "classification_metrics",
    "confusion_matrix",
    "efficiency_metrics",
    "evaluate_by_h",
    "evaluate_predictions",
    "horizon_degradation",
    "long_horizon_summary",
    "membership_metrics",
    "metric_direction",
    "resolve_labels",
    "resource_metrics",
    "summary_statistics",
    "validate_simplex",
    "write_metrics_bundle",
]
