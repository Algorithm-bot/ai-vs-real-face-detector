"""Build overall + per-generator metric tables with bootstrap CIs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.eval.metrics import bootstrap_ci, metric_summary, with_ci
from src.eval.splits import folder_of


def _subset(rows: Sequence[Dict[str, Any]], folder: Optional[str] = None) -> List[Dict[str, Any]]:
    if folder is None:
        return list(rows)
    return [row for row in rows if folder_of(row.get("folder") or row.get("source") or "") == folder]


def rows_to_arrays(rows: Sequence[Dict[str, Any]], score_key: str = "p_ai"):
    labels = np.array([int(float(row["label"])) for row in rows], dtype=int)
    scores = np.array([float(row[score_key]) for row in rows], dtype=float)
    folders = [str(row.get("folder") or folder_of(row.get("source") or "unknown")) for row in rows]
    return labels, scores, folders


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 0,
    score_key: str = "p_ai",
) -> Dict[str, Any]:
    if not rows:
        return {"samples": 0}
    labels, scores, folders = rows_to_arrays(rows, score_key=score_key)
    summary = metric_summary(labels, scores, threshold=threshold)
    ci = bootstrap_ci(
        labels, scores, folders, threshold=threshold, n_resamples=n_bootstrap, seed=seed
    )
    return with_ci(summary, ci)


def per_generator_table(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 0,
    score_key: str = "p_ai",
) -> Dict[str, Any]:
    folders = sorted({folder_of(row.get("folder") or row.get("source") or "") for row in rows})
    out: Dict[str, Any] = {}
    for folder in folders:
        subset = _subset(rows, folder)
        if not subset:
            continue
        out[folder] = summarize_rows(
            subset, threshold=threshold, n_bootstrap=n_bootstrap, seed=seed, score_key=score_key
        )
    return out


def loso_thresholds(
    rows: Sequence[Dict[str, Any]],
    score_key: str = "p_ai",
) -> Dict[str, float]:
    """Fit a balanced-accuracy threshold on all REPORT folders except one."""
    from src.eval.metrics import best_balanced_accuracy_threshold

    folders = sorted({folder_of(row.get("folder") or row.get("source") or "") for row in rows})
    thresholds: Dict[str, float] = {}
    for held_out in folders:
        others = [row for row in rows if folder_of(row.get("folder") or row.get("source") or "") != held_out]
        if len(others) < 2:
            thresholds[held_out] = 0.5
            continue
        labels, scores, _ = rows_to_arrays(others, score_key=score_key)
        thr, _ = best_balanced_accuracy_threshold(labels, scores)
        thresholds[held_out] = thr
    return thresholds


def loso_table(
    rows: Sequence[Dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 0,
    score_key: str = "p_ai",
) -> Dict[str, Any]:
    thresholds = loso_thresholds(rows, score_key=score_key)
    folders = sorted(thresholds)
    per: Dict[str, Any] = {}
    all_pred_rows = []
    for folder in folders:
        subset = _subset(rows, folder)
        if not subset:
            continue
        thr = thresholds[folder]
        summary = summarize_rows(
            subset, threshold=thr, n_bootstrap=n_bootstrap, seed=seed, score_key=score_key
        )
        summary["loso_threshold"] = thr
        per[folder] = summary
        all_pred_rows.extend(subset)
    # Overall LOSO uses each image's own held-out-folder threshold.
    if all_pred_rows:
        labels, scores, folders_arr = rows_to_arrays(all_pred_rows, score_key=score_key)
        pred = np.zeros(len(labels), dtype=int)
        for i, folder in enumerate(folders_arr):
            pred[i] = int(scores[i] >= thresholds[folder_of(folder)])
        from sklearn.metrics import accuracy_score

        overall = {
            "samples": int(len(labels)),
            "accuracy": float(accuracy_score(labels, pred)),
            "roc_auc": metric_summary(labels, scores, threshold=0.5).get("roc_auc"),
            "average_precision": metric_summary(labels, scores, threshold=0.5).get("average_precision"),
            "thresholds": thresholds,
        }
    else:
        overall = {"samples": 0}
    return {"overall": overall, "by_generator": per}


def build_eval_tables(
    rows: Sequence[Dict[str, Any]],
    calibrated_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 0,
    score_key: str = "p_ai",
) -> Dict[str, Any]:
    return {
        "at_0.5": {
            "overall": summarize_rows(rows, 0.5, n_bootstrap, seed, score_key),
            "by_generator": per_generator_table(rows, 0.5, n_bootstrap, seed, score_key),
        },
        "calibrated": {
            "threshold": calibrated_threshold,
            "overall": summarize_rows(rows, calibrated_threshold, n_bootstrap, seed, score_key),
            "by_generator": per_generator_table(
                rows, calibrated_threshold, n_bootstrap, seed, score_key
            ),
        },
        "loso": loso_table(rows, n_bootstrap, seed, score_key),
    }


def write_tables_csv(path: Path, tables: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, Any]] = []

    def add(setting: str, scope: str, name: str, summary: Dict[str, Any]) -> None:
        if not summary:
            return
        ci = summary.get("ci") or {}
        rows_out.append(
            {
                "setting": setting,
                "scope": scope,
                "generator": name,
                "samples": summary.get("samples"),
                "threshold": summary.get("threshold", summary.get("loso_threshold")),
                "accuracy": summary.get("accuracy"),
                "accuracy_ci_low": (ci.get("accuracy") or {}).get("low"),
                "accuracy_ci_high": (ci.get("accuracy") or {}).get("high"),
                "roc_auc": summary.get("roc_auc"),
                "roc_auc_ci_low": (ci.get("roc_auc") or {}).get("low"),
                "roc_auc_ci_high": (ci.get("roc_auc") or {}).get("high"),
                "average_precision": summary.get("average_precision"),
                "ap_ci_low": (ci.get("average_precision") or {}).get("low"),
                "ap_ci_high": (ci.get("average_precision") or {}).get("high"),
                "recall": summary.get("recall"),
                "specificity": summary.get("specificity"),
                "balanced_accuracy": summary.get("balanced_accuracy"),
            }
        )

    for setting in ("at_0.5", "calibrated"):
        block = tables.get(setting) or {}
        add(setting, "overall", "ALL", block.get("overall") or {})
        for gen, summary in (block.get("by_generator") or {}).items():
            add(setting, "generator", gen, summary)
    loso = tables.get("loso") or {}
    add("loso", "overall", "ALL", loso.get("overall") or {})
    for gen, summary in (loso.get("by_generator") or {}).items():
        add("loso", "generator", gen, summary)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else ["setting"])
        writer.writeheader()
        writer.writerows(rows_out)


def write_tables_json(path: Path, tables: Dict[str, Any]) -> None:
    path.write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
