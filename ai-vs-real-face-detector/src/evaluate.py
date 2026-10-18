"""Evaluate a saved checkpoint on REPORT folders (DEV is never a headline split)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.ablation import ABLATION_CONFIGS
from src.deep_branch.preprocessing import get_val_transforms
from src.eval.calibration_io import load_calibration
from src.eval.forward import predict_loader, predictions_to_rows
from src.eval.metrics import (
    apply_temperature,
    expected_calibration_error,
    metric_summary,
)
from src.eval.predictions import write_predictions_csv
from src.eval.splits import DEFAULT_SPLITS_PATH, load_split_spec
from src.eval.tables import build_eval_tables, write_tables_csv, write_tables_json
from src.inference import load_model
from src.train import FaceBinaryDataset, _has_explicit_split_layout, _split_has_labels


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_leakage_report(train_paths: Iterable[str], test_paths: Iterable[str], output: Path) -> None:
    train_by_hash: dict[str, list[str]] = defaultdict(list)
    for item in train_paths:
        train_by_hash[sha256(Path(item))].append(item)
    test_list = list(test_paths)
    overlaps = []
    for item in test_list:
        matching_train_paths = train_by_hash.get(sha256(Path(item)), [])
        if matching_train_paths:
            overlaps.append((item, matching_train_paths))
    lines = ["Data leakage report (SHA-256 exact-file comparison)", ""]
    lines.append(f"Training images checked: {sum(map(len, train_by_hash.values()))}")
    lines.append(f"Test images checked: {len(test_list)}")
    lines.append(f"Overlapping test images: {len(overlaps)}")
    if overlaps:
        lines.extend(["", "Overlaps:"])
        for test_path, train_matches in overlaps:
            lines.append(f"TEST: {test_path}")
            for train_path in train_matches:
                lines.append(f"  TRAIN: {train_path}")
    else:
        lines.append("No exact file-hash overlap found.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_options(checkpoint_args: dict[str, Any], mode: str, feature_cache: Path | None):
    return {
        "use_physics": mode in {"hybrid", "full_hybrid", "physics_only"},
        "use_prnu": mode in {"full_hybrid", "prnu_only"} or "prnu" in str(checkpoint_args.get("branches") or ""),
        "use_semantic": mode in {"full_hybrid", "semantic_only", "clip_linear"},
        "use_freq": "freq" in str(checkpoint_args.get("branches") or "") or mode == "freq_only",
        "semantic_model": checkpoint_args.get("semantic_model", "vit_small_patch16_224"),
        "semantic_pretrained": checkpoint_args.get(
            "semantic_pretrained",
            not checkpoint_args.get("no_semantic_pretrained", False),
        ),
        "transform": get_val_transforms(),
        "feature_cache_dir": str(feature_cache) if feature_cache else None,
        "legacy_cache": bool(checkpoint_args.get("legacy", False)),
    }


def _dataset_for_evaluation(
    data_dir: Path,
    checkpoint_args: dict[str, Any],
    mode: str,
    feature_cache: Path | None = None,
    folder_filter: Sequence[str] | None = None,
):
    """Prefer an explicit test directory; otherwise mirror the training split."""
    dataset_options = _dataset_options(checkpoint_args, mode, feature_cache)
    if folder_filter:
        dataset_options["folder_filter"] = list(folder_filter)
    seed = int(checkpoint_args.get("seed", 42))
    if _has_explicit_split_layout(data_dir) or all(
        _split_has_labels(data_dir, split) for split in ("train", "val", "test")
    ):
        ds = FaceBinaryDataset(str(data_dir), split="test", seed=seed, **dataset_options)
        train_ds = FaceBinaryDataset(
            str(data_dir),
            split="train",
            seed=seed,
            use_physics=False,
            transform=get_val_transforms(),
        )
        return ds, [path for path, _, _ in train_ds.samples], "explicit train/val/test split"

    test_root = data_dir / "test"
    if (test_root / "real").exists() or (test_root / "fake").exists():
        ds = FaceBinaryDataset(
            str(test_root), split="train", val_ratio=0.0, seed=seed, **dataset_options
        )
        train_root = data_dir / "train"
        if (train_root / "real").exists() or (train_root / "fake").exists():
            train_ds = FaceBinaryDataset(
                str(train_root), split="train", val_ratio=0.0, seed=seed, use_physics=False
            )
            train_paths = [path for path, _, _ in train_ds.samples]
        else:
            train_paths = []
        return ds, train_paths, "explicit test directory"

    val_ratio = float(checkpoint_args.get("val_ratio", 0.15))
    test_ds = FaceBinaryDataset(
        str(data_dir), split="val", val_ratio=val_ratio, seed=seed, **dataset_options
    )
    train_ds = FaceBinaryDataset(
        str(data_dir), split="train", val_ratio=val_ratio, seed=seed, **dataset_options
    )
    return test_ds, [path for path, _, _ in train_ds.samples], "training-validation split"


def save_plots(labels: Sequence[int], scores: Sequence[float], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

    predicted = (np.asarray(scores) >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Real", "AI"], yticklabels=["Real", "AI"],
        xlabel="Predicted", ylabel="Actual", title="Confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    if len(set(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=160)
        plt.close(fig)
        precision, recall, _ = precision_recall_curve(labels, scores)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
        fig.tight_layout()
        fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
        plt.close(fig)


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    device_name: str | None = None,
    feature_cache: Path | None = None,
    splits_yaml: Path | None = None,
    eval_set: str = "report",
    legacy: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name
        or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    )
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = raw_checkpoint.get("mode", "hybrid")
    ckpt_args = dict(raw_checkpoint.get("args") or {})
    if legacy:
        ckpt_args["legacy"] = True
        folders = None
        split_tag = "all-test"
    else:
        spec = load_split_spec(splits_yaml or DEFAULT_SPLITS_PATH)
        if eval_set == "dev":
            folders = spec.dev
            split_tag = "dev"
        elif eval_set == "all":
            folders = None
            split_tag = "all-test"
        else:
            folders = spec.report
            split_tag = "report"
        if eval_set == "report":
            print(f"Headline evaluation uses REPORT folders only: {spec.report}")

    try:
        dataset, train_paths, split_name = _dataset_for_evaluation(
            data_dir, ckpt_args, mode, feature_cache, folder_filter=folders
        )
    except FileNotFoundError as exc:
        (output_dir / "data_leakage_report.txt").write_text(
            "Data leakage report unavailable: no supported labeled test/validation images were found.\n"
            f"Reason: {exc}\n",
            encoding="utf-8",
        )
        raise
    test_paths = [path for path, _, _ in dataset.samples]
    write_leakage_report(train_paths, test_paths, output_dir / "data_leakage_report.txt")
    if not dataset.samples:
        raise RuntimeError("No held-out images exist after applying the eval split filter.")

    model, mode = load_model(str(checkpoint), device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    collected = predict_loader(model, mode, loader, device)
    rows = predictions_to_rows(collected, split=split_tag)
    write_predictions_csv(output_dir / "predictions.csv", rows)

    calibration = load_calibration(checkpoint) or {}
    temperature = float(calibration.get("temperature") or ckpt_args.get("temperature") or 1.0)
    calibrated_threshold = float(
        calibration.get("threshold") or ckpt_args.get("calibrated_threshold") or 0.5
    )
    labels = collected["label"]
    raw_scores = np.asarray(collected["p_ai"], dtype=float)
    scaled = apply_temperature(raw_scores, temperature)
    ece_before = expected_calibration_error(labels, raw_scores)
    ece_after = expected_calibration_error(labels, scaled)

    tables = build_eval_tables(
        [{**row, "p_ai": float(s)} for row, s in zip(rows, scaled)],
        calibrated_threshold=calibrated_threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_tables_json(output_dir / "tables.json", tables)
    write_tables_csv(output_dir / "tables.csv", tables)

    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "split": f"{split_name} / {split_tag}",
        "eval_set": split_tag,
        "positive_class": "ai",
        "headline_uses_dev": False if split_tag == "report" else split_tag != "report",
        "overall_at_0.5": metric_summary(labels, scaled, threshold=0.5),
        "overall_at_calibrated": metric_summary(labels, scaled, threshold=calibrated_threshold),
        "calibrated_threshold": calibrated_threshold,
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "mean_gates": collected.get("mean_gates") or {},
        "tables": tables,
        "by_source": {},
        "by_generator": tables.get("at_0.5", {}).get("by_generator", {}),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    save_plots(labels, scaled, output_dir)
    if collected.get("mean_gates"):
        print("Mean gate values:", json.dumps(collected["mean_gates"], indent=2))
    return metrics


def run_ablation(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    device_name: str | None = None,
    feature_cache: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, Path] = {}
    if checkpoint.is_dir():
        for cfg in ABLATION_CONFIGS:
            nested = checkpoint / cfg.train_mode / cfg.checkpoint_name
            flat = checkpoint / cfg.checkpoint_name
            path = nested if nested.exists() else flat
            if path.exists():
                checkpoints[cfg.variant.value] = path
    elif checkpoint.is_file():
        checkpoints["single"] = checkpoint
    else:
        raise FileNotFoundError(f"No ablation checkpoints found at {checkpoint}")

    results: dict[str, Any] = {"variants": {}, "comparison": []}
    for name, path in checkpoints.items():
        variant_dir = output_dir / name
        metrics = evaluate(path, data_dir, variant_dir, device_name, feature_cache, **kwargs)
        overall = metrics.get("overall_at_calibrated") or metrics.get("overall") or {}
        results["variants"][name] = {
            "checkpoint": str(path),
            "metrics": overall,
            "by_generator": metrics.get("by_generator", {}),
        }
        results["comparison"].append(
            {
                "variant": name,
                "accuracy": overall.get("accuracy"),
                "roc_auc": overall.get("roc_auc"),
                "average_precision": overall.get("average_precision"),
            }
        )
    (output_dir / "ablation_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on REPORT folders")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "hybrid_best.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation_outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--splits-yaml", type=Path, default=None)
    parser.add_argument("--eval-set", choices=["report", "dev", "all"], default="report")
    parser.add_argument("--legacy", action="store_true", help="Evaluate the full test tree; keep 0.5-only metrics.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    kwargs = dict(
        splits_yaml=args.splits_yaml,
        eval_set="all" if args.legacy else args.eval_set,
        legacy=args.legacy,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if args.ablation:
        print(json.dumps(run_ablation(
            args.checkpoint, args.data_dir, args.output_dir, args.device, args.feature_cache, **kwargs
        ), indent=2, default=str))
    else:
        print(json.dumps(evaluate(
            args.checkpoint, args.data_dir, args.output_dir, args.device, args.feature_cache, **kwargs
        ), indent=2, default=str))


if __name__ == "__main__":
    main()
