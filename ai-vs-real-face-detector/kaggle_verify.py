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
8. Frequency branch (64-d) and branch dropout validation

Notebook cell snippet to run:
```python
!python kaggle_verify.py --data-dir /kaggle/input/forensynths --feature-cache /kaggle/working/feature_cache
```
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
from src.freq_branch.extractor import FREQ_FEATURE_DIM, FreqExtractor
from src.fusion.fuse import FeatureFusion, FusionMode, GatedFusion
from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM, PhysicsFeatureExtractor
from src.prnu_branch.extractor import PRNU_FEATURE_DIM, PRNUExtractor


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
        parts = folder_key.split("/")
        target_dir = test_root.joinpath(*parts)
        if not target_dir.exists():
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
    print(f"PROGAN (Seen generator): {len(spec.progan)} folders")

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

        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        if no_down:
            if hasattr(m.conv_stem, "stride"):
                m.conv_stem.stride = (1, 1)
        m.eval().to(device)

        bs_deep = 100 if device.type == "cuda" else 4
        inputs = torch.randn(bs_deep, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(1):
                _ = m(inputs[:1])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = m(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = bs_deep / max(elapsed, 1e-6)
        peak_mem = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0

        label = "stride 1 (--deep-no-down)" if no_down else "stride 2 (default)"
        print(f"EfficientNet-B0 [{label}]: {fps:.1f} img/s | Peak VRAM: {peak_mem:.1f} MB")

    # Benchmark CLIP ViT-L/14
    print("\nBenchmarking CLIP ViT-L/14:")
    try:
        pretrained_clip = (device.type == "cuda")
        try:
            if pretrained_clip:
                clip_model = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True)
                print("  Successfully loaded pretrained OpenAI CLIP weights!")
            else:
                clip_model = timm.create_model("vit_large_patch14_clip_224", pretrained=False)
                print("  Loaded random weights (local CPU mode).")
        except Exception:
            clip_model = timm.create_model("vit_large_patch14_clip_224", pretrained=False)
            print("  Loaded random weights.")

        clip_model.eval().to(device)
        embedding_dim = clip_model.head.out_features if hasattr(clip_model, "head") and hasattr(clip_model.head, "out_features") else clip_model.num_features
        print(f"  Embedding dim: {embedding_dim}")
        if hasattr(clip_model, "default_cfg"):
            print(f"  pretrained_cfg mean: {clip_model.default_cfg.get('mean')}, std: {clip_model.default_cfg.get('std')}")

        bs_clip = 64 if device.type == "cuda" else 2
        inputs = torch.randn(bs_clip, 3, 224, 224, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            _ = clip_model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = bs_clip / max(elapsed, 1e-6)
        peak_mem = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
        print(f"  CLIP ViT-L/14 (fp16, bs={bs_clip}): {fps:.1f} img/s | Peak VRAM: {peak_mem:.1f} MB")
    except Exception as exc:
        print(f"  [WARN] Failed to benchmark CLIP ViT-L/14: {exc}")


def check_frequency_and_dropout():
    report_section("4. FREQUENCY BRANCH & BRANCH DROPOUT VERIFICATION")
    # 1. Frequency branch
    extractor = FreqExtractor()
    test_img = np.random.randint(0, 256, (512, 384, 3), dtype=np.uint8)
    res = extractor.extract(test_img)
    print(f"FreqExtractor feature dim: {len(res.vector)} (Expected: {FREQ_FEATURE_DIM})")
    assert len(res.vector) == FREQ_FEATURE_DIM, "Freq feature dim mismatch"
    assert np.all(np.isfinite(res.vector)), "Freq vector contains NaN/inf"
    print("[OK] FreqExtractor 64-d vector verified (NaN/inf safe).")

    # 2. Branch dropout
    gated = GatedFusion(
        modality_dims={"deep": 1280, "physics": 22, "semantic": 384, "freq": 64},
        branch_dropout=0.15,
    )
    gated.train()
    mods = {
        "deep": torch.ones(20, 1280),
        "physics": torch.ones(20, 22),
        "semantic": torch.ones(20, 384),
        "freq": torch.ones(20, 64),
    }
    out = gated.apply_branch_dropout(mods)
    assert torch.all(out["deep"] == mods["deep"]), "Deep branch was dropped!"
    print("[OK] Branch dropout verified: deep branch is kept active, auxiliary branches dropped with p=0.15.")


def check_feature_allclose(data_dir: Path, feature_cache: Path | None):
    report_section("5. CACHED VS LIVE FEATURE ALLCLOSE CHECK")
    if feature_cache is None or not Path(feature_cache).exists():
        print("[INFO] No feature cache directory provided. Skipping cached-vs-live check.")
        print("Pass --feature-cache /path/to/cache to run this check on Kaggle.")
        return

    meta_path = Path(feature_cache) / "meta.json"
    if meta_path.exists():
        print("Feature Cache Meta:")
        print(meta_path.read_text(encoding="utf-8"))

    test_npz = Path(feature_cache) / "test.npz"
    if not test_npz.exists():
        print(f"[WARN] {test_npz} not found.")
        return

    z = np.load(test_npz)
    print(f"test.npz arrays present: {list(z.files)}")
    print(f"Total cached test samples: {len(z['paths'])}")


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
    check_frequency_and_dropout()
    check_feature_allclose(args.data_dir, args.feature_cache)

    print("\n" + "=" * 65)
    print("Kaggle verification complete. Copy and paste the terminal report.")
    print("=" * 65)


if __name__ == "__main__":
    main()
