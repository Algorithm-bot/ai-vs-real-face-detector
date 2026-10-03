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


class FeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.physics = None
        self.prnu = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.physics is None:  # built lazily so each worker owns its extractors
            self.physics = PhysicsFeatureExtractor()
            self.prnu = PRNUExtractor()
        path = self.samples[i][0]
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                raise IOError("unreadable image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            p = np.asarray(self.physics.extract(rgb).vector, dtype=np.float32)
            q = np.asarray(self.prnu.extract(rgb).vector, dtype=np.float32)
            v = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)  # same as SemanticEncoder
            ok = 1
        except Exception as exc:  # keep a long job alive; bad images are dropped later
            print(f"\n[skip] {path}: {exc}", flush=True)
            p = np.zeros(PHYSICS_FEATURE_DIM, dtype=np.float32)
            q = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)
            v = np.zeros((224, 224, 3), dtype=np.uint8)
            ok = 0
        return torch.from_numpy(p), torch.from_numpy(q), torch.from_numpy(v), ok


def process_split(split, samples, encoder, device, cache_dir, args):
    final = cache_dir / f"{split}.npz"
    if final.exists() and not args.force:
        print(f"[{split}] already cached -> {final}")
        return
    shard_dir = cache_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    n_shards = math.ceil(len(samples) / args.shard_size)
    for k in range(n_shards):
        shard = shard_dir / f"{k:05d}.npz"
        if shard.exists():
            continue
        chunk = samples[k * args.shard_size:(k + 1) * args.shard_size]
        loader = DataLoader(
            FeatureDataset(chunk), batch_size=64, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        phys, prnu, sem, oks = [], [], [], []
        for p, q, v, ok in tqdm(loader, desc=f"{split} shard {k + 1}/{n_shards}"):
            x = v.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            with torch.no_grad():
                s = encoder(x).float().cpu().numpy()
            phys.append(p.numpy()); prnu.append(q.numpy()); sem.append(s); oks.append(ok.numpy())
        tmp = shard_dir / f"{k:05d}.tmp.npz"
        np.savez(
            tmp,
            paths=np.array([s[0] for s in chunk]),
            labels=np.array([s[1] for s in chunk], dtype=np.int8),
            sources=np.array([s[2] for s in chunk]),
            physics=np.concatenate(phys), prnu=np.concatenate(prnu),
            semantic=np.concatenate(sem), ok=np.concatenate(oks).astype(bool),
        )
        os.replace(tmp, shard)

    parts = [np.load(shard_dir / f"{k:05d}.npz") for k in range(n_shards)]
    merged = {key: np.concatenate([z[key] for z in parts]) for key in parts[0].files}
    keep = merged.pop("ok")
    dropped = int((~keep).sum())
    merged = {key: val[keep] for key, val in merged.items()}
    for key in ("physics", "prnu", "semantic"):
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
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--semantic-model", default=DEFAULT_VIT_MODEL)
    ap.add_argument("--no-semantic-pretrained", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Need a GPU for the ViT pass.")
    device = torch.device("cuda")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = SemanticEncoder(
        model_name=args.semantic_model,
        pretrained=not args.no_semantic_pretrained,
        device=device,
    )
    encoder.eval()

    meta = {
        "semantic_model": args.semantic_model,
        "semantic_pretrained": not args.no_semantic_pretrained,
        "physics_dim": PHYSICS_FEATURE_DIM,
        "prnu_dim": PRNU_FEATURE_DIM,
        "semantic_dim": int(encoder.embedding_dim),
        "max_train_per_class": args.max_train_per_class,
        "seed": args.seed,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for split in args.splits:
        samples = collect_labeled_images(Path(args.data_dir) / split)
        if not samples:
            raise FileNotFoundError(f"No images under {Path(args.data_dir) / split}")
        if split == "train" and args.max_train_per_class > 0:
            before = len(samples)
            samples = subsample_balanced(samples, args.max_train_per_class, args.seed)
            print(f"[train] subset {before} -> {len(samples)} images")
        process_split(split, samples, encoder, device, cache_dir, args)

    print("Feature cache complete:", cache_dir)


if __name__ == "__main__":
    main()
