"""Synthetic dataset and cache fixture generator for fast CPU testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

SYNTHETIC_FOLDERS: List[str] = [
    # ProGAN
    "progan/airplane",
    "progan/car",
    # FOLD_A
    "stylegan/bedroom",
    "stylegan/car",
    "cyclegan/apple",
    "cyclegan/horse",
    "san",
    "deepfake",
    "biggan",
    "stargan",
    "gaugan",
    # FOLD_B
    "stylegan2/car",
    "stylegan2/cat",
    "crn",
    "imle",
    "seeingdark",
    "whichfaceisreal",
]

SIZES: List[Tuple[int, int]] = [
    (256, 256),
    (512, 384),
    (384, 512),
    (1024, 1024),
    (300, 224),
]


def create_synthetic_image(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Generate a synthetic RGB image with varied patterns and color gradients."""
    rng = np.random.RandomState(seed)
    # Gradient base
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    r = (np.sin(xx * 6.28) * 127 + 128).astype(np.uint8)
    g = (np.cos(yy * 6.28) * 127 + 128).astype(np.uint8)
    b = (rng.randint(0, 255, size=(height, width))).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    # Add a blob/circle
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (200, 50, 150), -1)
    return img


def create_synthetic_dataset(root_dir: Path, images_per_class: int = 2) -> Dict[str, int]:
    """Populate a minimal fake ForenSynths tree with mixed size images."""
    root_dir = Path(root_dir)
    counts = {}
    seed_idx = 0

    for split in ["train", "val", "test"]:
        split_dir = root_dir / split
        folders = ["progan/airplane", "progan/car"] if split == "train" else SYNTHETIC_FOLDERS
        for folder in folders:
            for label_name, label_id in [("0_real", 0), ("1_fake", 1)]:
                target_dir = split_dir / folder / label_name
                target_dir.mkdir(parents=True, exist_ok=True)
                for i in range(images_per_class):
                    w, h = SIZES[(seed_idx + i) % len(SIZES)]
                    img = create_synthetic_image(w, h, seed=seed_idx)
                    # Alternate PNG and JPG
                    ext = ".jpg" if i % 2 == 0 else ".png"
                    img_path = target_dir / f"img_{i:04d}{ext}"
                    if ext == ".jpg":
                        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    else:
                        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    key = f"{folder}"
                    counts[key] = counts.get(key, 0) + 1
                    seed_idx += 1

    return counts


def create_synthetic_feature_cache(
    cache_dir: Path,
    dataset_root: Path,
    split: str = "test",
    dims: Dict[str, int] | None = None,
) -> Path:
    """Create a mock .npz feature cache file with random arrays matching dims."""
    from src.train import collect_labeled_images

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_labeled_images(dataset_root / split)

    dims = dims or {"physics": 22, "prnu": 9, "semantic": 384, "freq": 64}
    n = len(samples)
    paths = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples], dtype=np.int8)
    sources = np.array([s[2] for s in samples])

    rng = np.random.RandomState(42)
    cache_data = {
        "paths": paths,
        "labels": labels,
        "sources": sources,
    }
    for k, d in dims.items():
        cache_data[k] = rng.randn(n, d).astype(np.float32)

    np.savez(cache_dir / f"{split}.npz", **cache_data)
    meta = {
        "semantic_model": "vit_small_patch16_224",
        "semantic_pretrained": False,
        "physics_dim": dims.get("physics", 22),
        "prnu_dim": dims.get("prnu", 9),
        "semantic_dim": dims.get("semantic", 384),
        "freq_dim": dims.get("freq", 64),
        "seed": 42,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir / f"{split}.npz"
