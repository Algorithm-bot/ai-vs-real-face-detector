"""Fit a global threshold and temperature on DEV folders only.

    python src/calibrate_threshold.py --checkpoint models/full_hybrid_best.pt \
        --data-dir /mnt/data --feature-cache /mnt/data/feature_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_branch.preprocessing import get_val_transforms
from src.eval.calibration_io import write_calibration
from src.eval.forward import predict_loader
from src.eval.metrics import (
    apply_temperature,
    best_balanced_accuracy_threshold,
    expected_calibration_error,
    fit_temperature,
    metric_summary,
)
from src.eval.splits import EvalSplitSpec, filter_samples, load_split_spec
from src.inference import load_model
from src.train import FaceBinaryDataset


def _dev_dataset(data_dir: Path, checkpoint_args: dict, mode: str, feature_cache, spec: EvalSplitSpec):
    options = {
        "use_physics": mode in {"hybrid", "full_hybrid", "physics_only"},
        "use_prnu": mode in {"full_hybrid", "prnu_only"},
        "use_semantic": mode in {"full_hybrid", "semantic_only", "clip_linear"},
        "use_freq": mode == "freq_only" or "freq" in str(checkpoint_args.get("branches") or ""),
        "semantic_model": checkpoint_args.get("semantic_model", "vit_small_patch16_224"),
        "semantic_pretrained": checkpoint_args.get(
            "semantic_pretrained",
            not checkpoint_args.get("no_semantic_pretrained", False),
        ),
        "transform": get_val_transforms(),
        "feature_cache_dir": str(feature_cache) if feature_cache else None,
        "folder_filter": spec.dev,
    }
    return FaceBinaryDataset(
        str(data_dir),
        split="test",
        seed=int(checkpoint_args.get("seed", 42)),
        **options,
    )


def calibrate(
    checkpoint: Path,
    data_dir: Path,
    feature_cache: Path | None,
    splits_yaml: Path | None,
    device_name: str | None = None,
) -> dict:
    device = torch.device(
        device_name
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = raw.get("mode", "hybrid")
    spec = load_split_spec(splits_yaml)
    dataset = _dev_dataset(data_dir, raw.get("args") or {}, mode, feature_cache, spec)
    if not dataset.samples:
        raise RuntimeError("No DEV images found. Check --splits-yaml against the test tree.")
    model, mode = load_model(str(checkpoint), device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    collected = predict_loader(model, mode, loader, device)
    labels = collected["label"]
    scores = np.asarray(collected["p_ai"], dtype=float)
    ece_before = expected_calibration_error(labels, scores)
    temperature = fit_temperature(labels, scores)
    calibrated_scores = apply_temperature(scores, temperature)
    ece_after = expected_calibration_error(labels, calibrated_scores)
    threshold, bacc = best_balanced_accuracy_threshold(labels, calibrated_scores)
    payload = {
        "checkpoint": str(checkpoint),
        "splits_yaml": str(spec.path) if spec.path else None,
        "dev_folders": spec.dev,
        "n_dev": int(len(labels)),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "dev_balanced_accuracy": float(bacc),
        "dev_metrics_at_calibrated": metric_summary(labels, calibrated_scores, threshold),
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    dest = write_calibration(checkpoint, payload)
    # Also store inside the checkpoint so inference can load a single file.
    raw["calibration"] = payload
    args = dict(raw.get("args") or {})
    args["calibrated_threshold"] = float(threshold)
    args["temperature"] = float(temperature)
    raw["args"] = args
    torch.save(raw, checkpoint)
    payload["sidecar"] = str(dest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate threshold + temperature on DEV folders")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--splits-yaml", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(calibrate(
        args.checkpoint, args.data_dir, args.feature_cache, args.splits_yaml, args.device
    ), indent=2))


if __name__ == "__main__":
    main()
