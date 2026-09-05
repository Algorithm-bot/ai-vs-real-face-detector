"""Evaluate a saved AI vs real checkpoint without training it.

Preferred layout is ``data/{train,val,test}/{real,fake}`` (nested CNNDetection
``0_real`` / ``1_fake`` trees are also accepted). Labels are 0=real and 1=AI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model
from src.deep_branch.preprocessing import get_val_transforms
from src.train import FaceBinaryDataset, _split_has_labels
from src.calibration.calibrator import UncertaintyEstimator
from src.classifier.ablation import ABLATION_CONFIGS


def metric_summary(labels: Sequence[int], ai_probabilities: Sequence[float]) -> dict[str, Any]:
    """Return binary metrics, treating AI (label 1) as the positive class."""
    y_true = np.asarray(labels, dtype=int)
    scores = np.asarray(ai_probabilities, dtype=float)
    y_pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_leakage_report(train_paths: Iterable[str], test_paths: Iterable[str], output: Path) -> None:
    train_by_hash: dict[str, list[str]] = defaultdict(list)
    for item in train_paths:
        train_by_hash[sha256(Path(item))].append(item)
    overlaps = []
    for item in test_paths:
        matching_train_paths = train_by_hash.get(sha256(Path(item)), [])
        if matching_train_paths:
            overlaps.append((item, matching_train_paths))
    lines = ["Data leakage report (SHA-256 exact-file comparison)", ""]
    lines.append(f"Training images checked: {sum(map(len, train_by_hash.values()))}")
    lines.append(f"Test images checked: {len(list(test_paths))}")
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


def _dataset_for_evaluation(
    data_dir: Path,
    checkpoint_args: dict[str, Any],
    mode: str,
):
    """Prefer an explicit test directory; otherwise mirror the training split."""
    dataset_options = {
        "use_physics": mode in {"hybrid", "full_hybrid", "physics_only"},
        "use_prnu": mode in {"full_hybrid", "prnu_only"},
        "use_semantic": mode in {"full_hybrid", "semantic_only"},
        "semantic_model": checkpoint_args.get(
            "semantic_model", "vit_small_patch16_224"
        ),
        "semantic_pretrained": checkpoint_args.get(
            "semantic_pretrained",
            not checkpoint_args.get("no_semantic_pretrained", False),
        ),
        "transform": get_val_transforms(),
    }
    seed = int(checkpoint_args.get("seed", 42))
    if all(_split_has_labels(data_dir, split) for split in ("train", "val", "test")):
        ds = FaceBinaryDataset(
            str(data_dir),
            split="test",
            seed=seed,
            **dataset_options,
        )
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
        root = test_root
        # With val_ratio=0 the train split includes every explicit test image.
        ds = FaceBinaryDataset(
            str(root),
            split="train",
            val_ratio=0.0,
            seed=seed,
            **dataset_options,
        )
        train_root = data_dir / "train"
        if (train_root / "real").exists() or (train_root / "fake").exists():
            train_ds = FaceBinaryDataset(str(train_root), split="train", val_ratio=0.0, seed=seed, use_physics=False)
            train_paths = [path for path, _, _ in train_ds.samples]
        else:
            train_paths = []
        return ds, train_paths, "explicit test directory"

    val_ratio = float(checkpoint_args.get("val_ratio", 0.15))
    seed = int(checkpoint_args.get("seed", 42))
    test_ds = FaceBinaryDataset(
        str(data_dir),
        split="val",
        val_ratio=val_ratio,
        seed=seed,
        **dataset_options,
    )
    train_ds = FaceBinaryDataset(
        str(data_dir),
        split="train",
        val_ratio=val_ratio,
        seed=seed,
        **dataset_options,
    )
    return test_ds, [path for path, _, _ in train_ds.samples], "training-validation split"


def save_plots(labels: Sequence[int], scores: Sequence[float], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = (np.asarray(scores) >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "AI"], yticklabels=["Real", "AI"], xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    if len(set(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots(); ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}"); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "roc_curve.png", dpi=160); plt.close(fig)
        precision, recall, _ = precision_recall_curve(labels, scores)
        fig, ax = plt.subplots(); ax.plot(recall, precision); ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve"); fig.tight_layout(); fig.savefig(output_dir / "precision_recall_curve.png", dpi=160); plt.close(fig)


def evaluate(checkpoint: Path, data_dir: Path, output_dir: Path, device_name: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = raw_checkpoint.get("mode", "hybrid")
    try:
        dataset, train_paths, split_name = _dataset_for_evaluation(
            data_dir,
            raw_checkpoint.get("args", {}),
            mode,
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
        raise RuntimeError("No held-out test/validation images exist. Add data/test/{real,fake} or at least two images per label/source.")

    model, mode = load_model(str(checkpoint), device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            if mode == "stage1":
                logits, _ = model(image)
            elif mode == "hybrid":
                physics = item["physics"].unsqueeze(0).to(device)
                logits, _ = model(image, physics)
            elif mode == "full_hybrid":
                physics = item["physics"].unsqueeze(0).to(device)
                prnu = item["prnu"].unsqueeze(0).to(device)
                semantic = item["semantic"].unsqueeze(0).to(device)
                logits, _ = model(image, physics, prnu, semantic)
            elif mode == "physics_only":
                logits, _ = model(item["physics"].unsqueeze(0).to(device))
            elif mode == "prnu_only":
                logits, _ = model(item["prnu"].unsqueeze(0).to(device))
            elif mode == "semantic_only":
                logits, _ = model(item["semantic"].unsqueeze(0).to(device))
            else:
                raise ValueError(f"Unsupported checkpoint mode: {mode}")
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            rows.append({"path": item["path"], "source": item["source"], "true_label": int(item["label"]), "predicted_label": int(np.argmax(probs)), "probability_real": float(probs[0]), "probability_ai": float(probs[1])})

    labels = [row["true_label"] for row in rows]
    scores = [row["probability_ai"] for row in rows]
    probs_list = [np.array([row["probability_real"], row["probability_ai"]]) for row in rows]

    uncertainty_est = UncertaintyEstimator()
    calibration = uncertainty_est.calibration_metrics(probs_list, labels)

    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "split": split_name,
        "positive_class": "ai",
        "overall": metric_summary(labels, scores),
        "calibration": calibration,
        "by_source": {},
        "by_generator": {},
    }
    sources = sorted({row["source"] for row in rows})
    for source in sources:
        subset = [row for row in rows if row["source"] == source]
        if len(subset) < 1:
            continue
        metrics["by_source"][source] = metric_summary(
            [row["true_label"] for row in subset],
            [row["probability_ai"] for row in subset],
        )
    # Generator-wise evaluation for unseen generators
    for gen in (
        "stylegan2", "diffusion", "midjourney", "stable_diffusion", "dalle",
        "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan",
        "crn", "imle", "sitd", "san", "deepfake", "whichfaceisreal", "ffhq", "lsun",
    ):
        gen_rows = [
            row for row in rows
            if row["source"].lower() == gen
            or row["source"].lower().startswith((gen + "/", gen + "_", gen + "-"))
        ]
        if gen_rows:
            metrics["by_generator"][gen] = metric_summary(
                [row["true_label"] for row in gen_rows],
                [row["probability_ai"] for row in gen_rows],
            )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_plots(labels, scores, output_dir)
    return metrics


def run_ablation(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate independently trained branch checkpoints plus the full model.

    If ``checkpoint`` is a directory, look up the standard ablation filenames
    from :data:`ABLATION_CONFIGS`. If it is a file, evaluate that checkpoint
    only (kept for backward compatibility).
    """
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
        metrics = evaluate(path, data_dir, variant_dir, device_name)
        overall = metrics.get("overall", {})
        results["variants"][name] = {
            "checkpoint": str(path),
            "metrics": overall,
            "by_source": metrics.get("by_source", {}),
            "by_generator": metrics.get("by_generator", {}),
        }
        results["comparison"].append({
            "variant": name,
            "accuracy": overall.get("accuracy"),
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "f1_score": overall.get("f1_score"),
            "roc_auc": overall.get("roc_auc"),
        })

    (output_dir / "ablation_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on held-out AI vs real images")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "hybrid_best.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation_outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ablation", action="store_true", help="Compare independently trained branch checkpoints")
    args = parser.parse_args()
    if args.ablation:
        print(json.dumps(run_ablation(args.checkpoint, args.data_dir, args.output_dir, args.device), indent=2))
    else:
        print(json.dumps(evaluate(args.checkpoint, args.data_dir, args.output_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
