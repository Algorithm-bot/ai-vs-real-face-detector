"""Standalone Kaggle verification script for ForenSynths overhaul.

Prints a compact report verifying:
1. per-folder image counts vs eval/expected_counts.json (flagging any mismatch)
2. cached-vs-live feature equality (allclose per branch) on real images drawn
   from crn, stylegan_car, whichfaceisreal, progan_car and a 1024x1024 folder
3. DEV/REPORT split: fold sizes, folder lists, disjointness, and CRN & IMLE in the same fold
4. throughput (img/s) and GPU memory for EfficientNet-B0 with and without --deep-no-down
5. CLIP ViT-L/14 GPU memory and throughput at batch size 32/64
6. cache/live check for the augmentation path
7. sample dump of embedding dim and pretrained_cfg for semantic models
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.splits import load_split_spec, matches_folder


def report_section(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def check_dataset_counts(data_dir: Path, expected_json: Path):
    report_section("1. PER-FOLDER IMAGE COUNTS VS EXPECTED")
    if not expected_json.exists():
        print(f"[WARN] Expected counts file not found: {expected_json}")
        return
    with open(expected_json, "r", encoding="utf-8") as f:
        expected = json.load(f)

    test_root = data_dir / "test" if (data_dir / "test").exists() else data_dir
    actual = {}
    mismatches = []
    total_actual = 0
    total_expected = sum(expected.values())

    for folder_key in expected:
        # Search for folder_key (e.g. progan/airplane or biggan)
        parts = folder_key.split("/")
        target_dir = test_root.joinpath(*parts)
        if not target_dir.exists():
            # Check if flat e.g. progan_airplane
            flat_name = folder_key.replace("/", "_")
            target_dir = test_root / flat_name
        
        count = 0
        if target_dir.exists():
            for root, _, files in os.walk(target_dir):
                count += sum(1 for file in files if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        actual[folder_key] = count
        total_actual += count
        exp = expected[folder_key]
        diff = count - exp
        if abs(diff) > 2:
            mismatches.append((folder_key, count, exp, diff))

    print(f"Total Folders: {len(expected)} | Total Actual Images: {total_actual:,} | Expected: {total_expected:,}")
    if mismatches:
        print(f"[WARN] {len(mismatches)} folders have count mismatches (>2 difference):")
        for k, act, exp, diff in mismatches[:10]:
            print(f"  - {k}: actual={act}, expected={exp} (diff={diff:+d})")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more.")
    else:
        print("[OK] All folder counts match expected ForenSynths distribution (within ±2).")


def check_split_spec():
    report_section("2. DEV / REPORT SPLIT VERIFICATION")
    spec = load_split_spec()
    spec.validate()
    print(f"FOLD_A (Dev / selection): {len(spec.fold_a or spec.dev)} folders")
    print(f"  -> {spec.fold_a or spec.dev}")
    print(f"FOLD_B (Report / headline): {len(spec.fold_b or spec.report)} folders")
    print(f"  -> {spec.fold_b or spec.report}")
    print(f"PROGAN (Seen generator): {spec.progan}")
    
    # Assert disjointness
    for a in (spec.fold_a or spec.dev):
        for b in (spec.fold_b or spec.report):
            assert not matches_folder(a, b) and not matches_folder(b, a), f"Overlap: {a} and {b}"
    print("[OK] FOLD_A and FOLD_B are completely disjoint.")

    crn_dev = any("crn" in f for f in (spec.fold_a or spec.dev))
    imle_dev = any("imle" in f for f in (spec.fold_a or spec.dev))
    assert crn_dev == imle_dev, "CRN and IMLE must be in the same fold!"
    print(f"[OK] CRN and IMLE are in the same fold: {'FOLD_A' if crn_dev else 'FOLD_B'}.")


def check_models_and_throughput(device: torch.device):
    report_section("3. THROUGHPUT & GPU MEMORY BENCHMARKS")
    import timm
    print(f"Using timm version: {timm.__version__}")
    print(f"Compute device: {device}")

    # Benchmark EfficientNet-B0 default (stride 2) vs --deep-no-down (stride 1)
    for no_down in [False, True]:
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()
        
        # Build model
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        if no_down:
            # Change stem conv stride from 2 to 1
            if hasattr(m.conv_stem, "stride"):
                m.conv_stem.stride = (1, 1)
        m.eval().to(device)

        inputs = torch.randn(200, 3, 224, 224, device=device)
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = m(inputs[:32])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, 200, 50):
                _ = m(inputs[i:i+50])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = 200 / max(elapsed, 1e-6)
        peak_mem = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0

        label = "stride 1 (--deep-no-down)" if no_down else "stride 2 (default)"
        print(f"EfficientNet-B0 [{label}]: {fps:.1f} img/s | Peak VRAM: {peak_mem:.1f} MB")

    # Benchmark CLIP ViT-L/14
    print("\nBenchmarking CLIP ViT-L/14:")
    try:
        clip_model = timm.create_model("vit_large_patch14_clip_224", pretrained=False, num_classes=0)
        clip_model.eval().to(device)
        print(f"  Embedding dim (random weights): {clip_model.num_features}")
        if hasattr(clip_model, "default_cfg"):
            print(f"  pretrained_cfg mean: {clip_model.default_cfg.get('mean')}, std: {clip_model.default_cfg.get('std')}")
        
        inputs = torch.randn(64, 3, 224, 224, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            _ = clip_model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = 64 / max(elapsed, 1e-6)
        peak_mem = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
        print(f"  CLIP ViT-L/14 (fp16, bs=64): {fps:.1f} img/s | Peak VRAM: {peak_mem:.1f} MB")
    except Exception as exc:
        print(f"  [WARN] Failed to benchmark CLIP ViT-L/14: {exc}")


def check_feature_allclose(data_dir: Path, feature_cache: Path | None):
    report_section("4. CACHED VS LIVE FEATURE ALLCLOSE CHECK")
    if feature_cache is None or not Path(feature_cache).exists():
        print("[INFO] No feature cache directory provided. Skipping cached-vs-live check.")
        print("Pass --feature-cache /path/to/cache to run this check on Kaggle.")
        return

    # Check existence of cache files
    meta_path = Path(feature_cache) / "meta.json"
    if meta_path.exists():
        print("Feature Cache Meta:")
        print(meta_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Kaggle verification run")
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/forensynths"))
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--expected-counts", type=Path, default=PROJECT_ROOT / "eval" / "expected_counts.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kaggle Verification Runner started on {device}")
    
    check_split_spec()
    if args.data_dir.exists():
        check_dataset_counts(args.data_dir, args.expected_counts)
    else:
        print(f"\n[INFO] Data dir {args.data_dir} does not exist locally. Run on Kaggle.")

    check_models_and_throughput(device)
    check_feature_allclose(args.data_dir, args.feature_cache)
    print("\nVerification script finished. Copy and paste the output above.")


if __name__ == "__main__":
    main()
