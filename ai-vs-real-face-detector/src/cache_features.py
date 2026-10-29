"""
Compute the frozen-branch features ONCE and save them to disk.

Physics, PRNU and ViT vectors are computed from the raw image (no augmentation),
so they are identical every epoch. Recomputing them per image per epoch is what
makes training slow. This script does that work a single time:

  * physics + PRNU  -> CPU DataLoader workers
  * ViT embedding   -> batched on the GPU (same preprocessing as SemanticEncoder)

Output (in --cache-dir):  train.npz  val.npz  test.npz  meta.json
Resumable: work is saved in shards; re-running skips finished shards/splits.

    python src/cache_features.py --data-dir /mnt/data --cache-dir /mnt/data/feature_cache \
        --max-train-per-class 100000 --workers 30
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

cv2.setNumThreads(1)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import collect_labeled_images, subsample_balanced
from src.physics_branch.feature_vector import PhysicsFeatureExtractor, PHYSICS_FEATURE_DIM
from src.prnu_branch.extractor import PRNUExtractor, PRNU_FEATURE_DIM
from src.semantic_branch.encoder import (
    DEFAULT_VIT_MODEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SemanticEncoder,
)
import src.train


def _prnu_view(rgb, m=256):
    h, w = rgb.shape[:2]
    y, x = (h - min(h, m)) // 2, (w - min(w, m)) // 2
    return rgb[y:y + min(h, m), x:x + min(w, m)]


class FeatureDataset(Dataset):
    def __init__(
        self,
        samples,
        preprocess_mode: str = "crop",
        crop_size: int = 224,
        enable_augment: bool = False,
        seed: int = 42,
        p_blur: float = 0.5,
        p_jpeg: float = 0.8,
        p_downscale: float = 0.3,
        max_sigma: float = 3.0,
        min_quality: int = 30,
        downscale_range: Tuple[float, float] = (0.5, 0.9),
    ):
        self.samples = samples
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.enable_augment = enable_augment
        self.seed = seed
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg
        self.p_downscale = p_downscale
        self.max_sigma = max_sigma
        self.min_quality = min_quality
        self.downscale_range = downscale_range
        self.physics = None
        self.prnu = None
        self.freq = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from src.augment import augment_raw_image
        from src.freq_branch.extractor import FREQ_FEATURE_DIM, FreqExtractor
        from src.preprocess import PreprocessPipeline

        if self.physics is None:  # built lazily so each worker owns its extractors
            self.physics = PhysicsFeatureExtractor()
            self.prnu = PRNUExtractor()
            self.freq = FreqExtractor()
        path = self.samples[i][0]
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                raise IOError("unreadable image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Apply training augmentation at raw resolution if enabled (with deterministic path seeding)
            if self.enable_augment:
                rgb, crop_rng = augment_raw_image(
                    rgb,
                    path=path,
                    epoch=0,
                    base_seed=self.seed,
                    enable_augment=True,
                    p_blur=self.p_blur,
                    p_jpeg=self.p_jpeg,
                    p_downscale=self.p_downscale,
                    max_sigma=self.max_sigma,
                    min_quality=self.min_quality,
                    downscale_range=self.downscale_range,
                )
            else:
                crop_rng = None

            # Shared preprocessing: crop or resize
            pipeline = PreprocessPipeline(
                mode=self.preprocess_mode,
                crop_size=self.crop_size,
                is_training=self.enable_augment,
            )
            processed_rgb = pipeline.preprocess_image(rgb, rng=crop_rng)

            p = np.asarray(self.physics.extract(processed_rgb).vector, dtype=np.float32)
            q = np.asarray(self.prnu.extract(processed_rgb).vector, dtype=np.float32)
            from src.preprocess import preprocess_for_freq
            f = np.asarray(self.freq.extract(preprocess_for_freq(rgb, crop_size=self.crop_size, mode=self.preprocess_mode)).vector, dtype=np.float32)
            v = processed_rgb  # 224x224x3 uint8 passed to ViT/CLIP
            ok = 1
        except Exception as exc:  # keep a long job alive; bad images are dropped later
            print(f"\n[skip] {path}: {exc}", flush=True)
            p = np.zeros(PHYSICS_FEATURE_DIM, dtype=np.float32)
            q = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)
            f = np.zeros(FREQ_FEATURE_DIM, dtype=np.float32)
            v = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            ok = 0
        return torch.from_numpy(p), torch.from_numpy(q), torch.from_numpy(f), torch.from_numpy(v), ok




def process_split(split, samples, encoder, device, cache_dir, args):
    final = cache_dir / f"{split}.npz"
    if final.exists() and not args.force:
        print(f"[{split}] already cached -> {final}")
        return
    shard_dir = cache_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    batch_size = getattr(args, "batch_size", 64)
    mean = torch.tensor(encoder.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(encoder.std, device=device).view(1, 3, 1, 1)

    n_shards = math.ceil(len(samples) / args.shard_size)
    for k in range(n_shards):
        shard = shard_dir / f"{k:05d}.npz"
        if shard.exists():
            continue
        chunk = samples[k * args.shard_size:(k + 1) * args.shard_size]
        enable_aug = getattr(args, "augment", False) and split == "train"
        downscale_min, downscale_max = map(float, args.downscale_range.split(","))
        loader = DataLoader(
            FeatureDataset(
                chunk,
                preprocess_mode=getattr(args, "preprocess", "crop"),
                enable_augment=enable_aug,
                seed=args.seed,
                p_blur=args.p_blur,
                p_jpeg=args.p_jpeg,
                p_downscale=args.p_downscale,
                max_sigma=args.max_sigma,
                min_quality=args.min_quality,
                downscale_range=(downscale_min, downscale_max),
            ),
            batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            prefetch_factor=4 if args.workers > 0 else None,
        )

        phys, prnu, freqs, sem, oks = [], [], [], [], []
        for p, q, f, v, ok in tqdm(loader, desc=f"{split} shard {k + 1}/{n_shards}"):
            x = v.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                s = encoder(x).float().cpu().numpy()
            phys.append(p.numpy()); prnu.append(q.numpy()); freqs.append(f.numpy()); sem.append(s); oks.append(ok.numpy())

        max_attempts = 3
        saved = False
        last_exc = None
        for attempt in range(max_attempts):
            tmp = shard_dir / f"{k:05d}.tmp.npz"
            try:
                np.savez(
                    tmp,
                    paths=np.array([s[0] for s in chunk]),
                    labels=np.array([s[1] for s in chunk], dtype=np.int8),
                    sources=np.array([s[2] for s in chunk]),
                    physics=np.concatenate(phys), prnu=np.concatenate(prnu),
                    freq=np.concatenate(freqs),
                    semantic=np.concatenate(sem), ok=np.concatenate(oks).astype(bool),
                )
                # Check that we can load it
                with np.load(tmp) as data:
                    # Just opening to check
                    pass
                saved = True
                break
            except Exception as e:
                last_exc = e
                # If we want to wait, we could time.sleep, but not specified.
                continue

        if not saved:
            # Try rescue
            rescue_dir = Path("/kaggle/working")
            rescue_dir.mkdir(parents=True, exist_ok=True)
            rescue_path = rescue_dir / f"shard_rescue_{split}_{k:05d}.npz"
            try:
                np.savez(
                    rescue_path,
                    paths=np.array([s[0] for s in chunk]),
                    labels=np.array([s[1] for s in chunk], dtype=np.int8),
                    sources=np.array([s[2] for s in chunk]),
                    physics=np.concatenate(phys), prnu=np.concatenate(prnu),
                    freq=np.concatenate(freqs),
                    semantic=np.concatenate(sem), ok=np.concatenate(oks).astype(bool),
                )
                with np.load(rescue_path) as data:
                    pass
                # Rescue succeeded, now move it to the shard directory as the tmp file
                shard_dir.mkdir(parents=True, exist_ok=True)
                os.replace(rescue_path, tmp)
                saved = True
            except Exception as e:
                print(f"[{split}] ERROR: Failed to save shard {k} even to rescue path: {e}")
                raise last_exc

        # Now we have tmp in shard_dir (either from direct or rescue)
        os.replace(tmp, shard)

    parts = [np.load(shard_dir / f"{k:05d}.npz") for k in range(n_shards)]
    merged = {key: np.concatenate([z[key] for z in parts]) for key in parts[0].files}
    keep = merged.pop("ok")
    dropped = int((~keep).sum())
    merged = {key: val[keep] for key, val in merged.items()}
    for key in ("physics", "prnu", "freq", "semantic"):
        if key in merged:
            bad = ~np.isfinite(merged[key])
            if bad.any():
                rows = int(bad.any(axis=1).sum())
                print(f"[{split}] WARNING: {rows} rows had NaN/inf in '{key}'; replaced with 0")
                merged[key] = np.nan_to_num(merged[key], nan=0.0, posinf=0.0, neginf=0.0)
    tmp_final = cache_dir / f"{split}.tmp.npz"
    np.savez(tmp_final, **merged)
    os.replace(tmp_final, final)

    print(f"[{split}] saved {len(merged['labels'])} rows (dropped {dropped} unreadable) -> {final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-train-per-class", type=int, default=0,
                    help="Same meaning as train.py; use the SAME value so subsets match. 0 = all.")
    ap.add_argument("--max-train-images", type=int, default=0,
                    help="Cap total train images, balanced per class (N//2 real, N//2 fake), seeded. 0 = all.")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--semantic-model", default=DEFAULT_VIT_MODEL)
    ap.add_argument("--semantic-layers", default=None,
                    help="Optional comma-separated transformer block indices to extract CLS tokens from (e.g. 12,16,20,24)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for GPU semantic encoder forward pass")
    ap.add_argument("--no-semantic-pretrained", action="store_true")
    ap.add_argument("--preprocess", choices=["crop", "resize", "center_crop", "no_resize", "aspect_crop"], default="crop",
                    help="crop = random crop on train, center crop on test; center_crop = always center crop; no_resize = preserve original resolution; aspect_crop = aspect-preserving center crop; resize = legacy")
    ap.add_argument("--augment", "--augment-train", dest="augment", action="store_true",
                    help="Apply training augmentation (Blur+JPEG+Downscale) at raw resolution before crop")
    ap.add_argument("--p-blur", type=float, default=0.5, help="Probability of applying blur augmentation")
    ap.add_argument("--p-jpeg", type=float, default=0.8, help="Probability of applying JPEG compression augmentation")
    ap.add_argument("--p-downscale", type=float, default=0.3, help="Probability of applying downscale/upscale augmentation")
    ap.add_argument("--max-sigma", type=float, default=3.0, help="Maximum sigma for Gaussian blur")
    ap.add_argument("--min-quality", type=int, default=30, help="Minimum JPEG quality")
    ap.add_argument("--downscale-range", type=str, default="0.5,0.9", help="Downscale range as min,max (e.g., 0.5,0.9)")
    ap.add_argument("--branches", default="deep,physics,semantic,freq",
                    help="Comma-separated branches cached (default: deep,physics,semantic,freq; prnu is off)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-seed", type=int, default=42,
                    help="Random seed used only for balanced subset selection; training seed remains for init/order.")
    ap.add_argument("--exclude-sources", type=str, default="ffhq",
                    help="Comma-separated list of source folders to exclude from all splits. "
                         "Recorded in cache meta.json; trainers refuse a cache with a different list.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    src.train.EXCLUDE_SOURCES = tuple(s.strip() for s in args.exclude_sources.split(",") if s.strip())

    if not torch.cuda.is_available():
        raise RuntimeError("Need a GPU for the ViT pass.")
    device = torch.device("cuda")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = SemanticEncoder(
        model_name=args.semantic_model,
        pretrained=not args.no_semantic_pretrained,
        device=device,
        intermediate_layers=args.semantic_layers,
    )
    encoder.eval()

    from src.freq_branch.extractor import FREQ_FEATURE_DIM

    effective_per_class = args.max_train_per_class
    if args.max_train_images > 0:
        effective_per_class = max(1, args.max_train_images // 2)

    # Parse downscale range
    downscale_min, downscale_max = map(float, args.downscale_range.split(","))

    meta = {
        "semantic_model": args.semantic_model,
        "semantic_pretrained": not args.no_semantic_pretrained,
        "semantic_layers": args.semantic_layers or "",
        "physics_dim": PHYSICS_FEATURE_DIM,
        "prnu_dim": PRNU_FEATURE_DIM,
        "freq_dim": FREQ_FEATURE_DIM,
        "semantic_dim": int(encoder.embedding_dim),
        "branches": sorted({branch.strip() for branch in args.branches.split(",") if branch.strip()}),
        "max_train_images": args.max_train_images,
        "max_train_per_class": effective_per_class,
        "preprocess": args.preprocess,
        "crop_size": 224,
        "augment": bool(args.augment),
        "p_blur": args.p_blur,
        "p_jpeg": args.p_jpeg,
        "p_downscale": args.p_downscale,
        "max_sigma": args.max_sigma,
        "min_quality": args.min_quality,
        "downscale_min": downscale_min,
        "downscale_max": downscale_max,
        "seed": args.seed,
        "subset_seed": getattr(args, "subset_seed", args.seed),
        "exclude_sources": tuple(s.strip() for s in args.exclude_sources.split(",") if s.strip()),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))




    for split in args.splits:
        file_list_path = cache_dir / f"{split}_file_list.json"
        if file_list_path.exists():
            print(f"Loading cached file list for {split}...", flush=True)
            samples = json.loads(file_list_path.read_text(encoding="utf-8"))
            samples = [tuple(sample) for sample in samples]
        else:
            print(f"Enumerating files for {split} (first time only)...", flush=True)
            samples = collect_labeled_images(Path(args.data_dir) / split)
            file_list_path.write_text(json.dumps(samples), encoding="utf-8")
            print(
                f"Enumeration complete: {len(samples)} files, cached to {file_list_path}",
                flush=True,
            )
        if not samples:
            raise FileNotFoundError(f"No images under {Path(args.data_dir) / split}")
        if split == "train" and (args.max_train_images > 0 or args.max_train_per_class > 0):
            before = len(samples)
            subset_seed = getattr(args, "subset_seed", args.seed)
            samples = subsample_balanced(samples, effective_per_class, subset_seed)
            print(f"[train] subset {before} -> {len(samples)} images (max_train_images={args.max_train_images}, per_class={effective_per_class}, subset_seed={subset_seed})")
        process_split(split, samples, encoder, device, cache_dir, args)

    print("Feature cache complete:", cache_dir)


if __name__ == "__main__":
    main()
