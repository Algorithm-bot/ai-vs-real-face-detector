"""Shared classification metrics, threshold search, ECE, and bootstrap CIs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def _as_arrays(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must have the same length")
    return y, s


def metric_summary(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary metrics with AI (label 1) as the positive class."""
    y_true, scores = _as_arrays(labels, ai_probabilities)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    both = len(np.unique(y_true)) == 2
    try:
        roc_auc = float(roc_auc_score(y_true, scores)) if both else None
    except ValueError:
        roc_auc = None
    try:
        ap = float(average_precision_score(y_true, scores)) if both else None
    except ValueError:
        ap = None
    return {
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": precision,
        "recall": recall,
        "f1_score": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def expected_calibration_error(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, float]:
    scores = np.clip(np.asarray(ai_probabilities, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = max(len(y), 1)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if not np.any(mask):
            continue
        gap = abs(float(y[mask].mean()) - float(scores[mask].mean()))
        ece += gap * float(mask.sum()) / total
        mce = max(mce, gap)
    return {
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "n_bins": int(n_bins),
    }


def best_balanced_accuracy_threshold(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Log-spaced search on ``[1e-9, 1]`` maximising balanced accuracy."""
    y, scores = _as_arrays(labels, ai_probabilities)
    if grid is None:
        grid = np.logspace(-9, 0, 1000)
    best_thr = 0.5
    best_bacc = -1.0
    for thr in grid:
        summary = metric_summary(y, scores, threshold=float(thr))
        bacc = summary["balanced_accuracy"]
        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = float(thr)
    return best_thr, float(best_bacc)


def selection_score(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    """Checkpoint score: mean of DEV AUC and best-threshold balanced accuracy."""
    summary = metric_summary(labels, ai_probabilities, threshold=0.5)
    thr, bacc = best_balanced_accuracy_threshold(labels, ai_probabilities)
    auc = summary["roc_auc"]
    auc_value = 0.5 if auc is None else float(auc)
    score = 0.5 * auc_value + 0.5 * float(bacc)
    return float(score), {
        "auc": auc_value,
        "balanced_accuracy": float(bacc),
        "threshold": float(thr),
        "selection_score": float(score),
    }


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale binary (or 2-col) probabilities in log space."""
    probs = np.asarray(probabilities, dtype=float)
    temp = max(float(temperature), 1e-6)
    if probs.ndim == 1:
        p = np.clip(probs, 1e-8, 1.0 - 1e-8)
        logits = np.log(p) - np.log(1.0 - p)
        scaled = logits / temp
        return 1.0 / (1.0 + np.exp(-scaled))
    clipped = np.clip(probs, 1e-8, 1.0)
    logits = np.log(clipped)
    logits = logits / temp
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
    grid: Optional[np.ndarray] = None,
) -> float:
    """Minimise NLL of the AI-class probability on a temperature grid."""
    y, scores = _as_arrays(labels, ai_probabilities)
    if grid is None:
        grid = np.linspace(0.5, 5.0, 46)
    best_t = 1.0
    best_nll = float("inf")
    for temp in grid:
        calibrated = np.clip(apply_temperature(scores, float(temp)), 1e-8, 1.0 - 1e-8)
        nll = -np.mean(y * np.log(calibrated) + (1 - y) * np.log(1.0 - calibrated))
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(temp)
    return best_t


def bootstrap_ci(
    labels: Sequence[int],
    ai_probabilities: Sequence[float],
    folders: Sequence[str],
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 0,
    metrics: Sequence[str] = ("roc_auc", "average_precision", "accuracy"),
) -> Dict[str, Dict[str, float]]:
    """Percentile CIs; images are resampled with replacement *within folder*."""
    y, scores = _as_arrays(labels, ai_probabilities)
    folders_arr = np.asarray(list(folders))
    if folders_arr.shape[0] != y.shape[0]:
        raise ValueError("folders must align with labels")
    rng = np.random.RandomState(seed)
    groups: Dict[str, np.ndarray] = {}
    for folder in np.unique(folders_arr):
        groups[str(folder)] = np.flatnonzero(folders_arr == folder)

    collected = {name: [] for name in metrics}
    for _ in range(int(n_resamples)):
        idx_parts = [
            rng.choice(inds, size=len(inds), replace=True)
            for inds in groups.values()
            if len(inds)
        ]
        if not idx_parts:
            break
        idx = np.concatenate(idx_parts)
        summary = metric_summary(y[idx], scores[idx], threshold=threshold)
        for name in metrics:
            value = summary.get(name)
            collected[name].append(np.nan if value is None else float(value))

    out: Dict[str, Dict[str, float]] = {}
    for name, values in collected.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[name] = {"low": None, "high": None, "mean": None}
            continue
        out[name] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "mean": float(arr.mean()),
        }
    return out


def with_ci(
    summary: Mapping[str, Any],
    ci: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(summary)
    merged["ci"] = {k: dict(v) for k, v in ci.items()}
    return merged
