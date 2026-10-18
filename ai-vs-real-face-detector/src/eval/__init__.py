"""Evaluation helpers: splits, metrics, prediction I/O, cache meta."""

from src.eval.metrics import (
    best_balanced_accuracy_threshold,
    bootstrap_ci,
    expected_calibration_error,
    metric_summary,
    selection_score,
)
from src.eval.splits import EvalSplitSpec, folder_of, load_split_spec, partition_sources

__all__ = [
    "EvalSplitSpec",
    "best_balanced_accuracy_threshold",
    "bootstrap_ci",
    "expected_calibration_error",
    "folder_of",
    "load_split_spec",
    "metric_summary",
    "partition_sources",
    "selection_score",
]
