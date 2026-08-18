"""Fit and persist feature normalizers for training checkpoints."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.physics_branch.normalization import PhysicsNormalizer
from src.prnu_branch.extractor import PRNUExtractor, PRNU_FEATURE_NAMES


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_physics_and_prnu_scalers(
    sample_paths: Sequence[str],
    max_samples: int = 300,
    seed: int = 42,
) -> Tuple[PhysicsNormalizer, PhysicsNormalizer]:
    """
    Fit z-score scalers for physics and PRNU vectors on a representative
    subsample of training paths (not the full training set -- fitting on
    every image is unnecessary for stable mean/std estimates and, at full
    dataset size, prohibitively slow). Raises if any feature extraction fails.
    """
    paths = list(sample_paths)
    if len(paths) > max_samples:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_samples)

    physics_extractor = PhysicsFeatureExtractor()
    prnu_extractor = PRNUExtractor()
    physics_vectors: List[np.ndarray] = []
    prnu_vectors: List[np.ndarray] = []

    try:
        for path in tqdm(paths, desc="Fitting feature scalers"):
            rgb = _load_rgb(path)
            physics_vectors.append(physics_extractor.extract(rgb).vector)
            prnu_vectors.append(prnu_extractor.extract(rgb).vector)
    finally:
        physics_extractor.close()

    if not physics_vectors:
        raise RuntimeError("No training samples available to fit feature scalers.")

    physics_norm = PhysicsNormalizer.fit(np.stack(physics_vectors, axis=0))
    prnu_norm = PhysicsNormalizer.fit(
        np.stack(prnu_vectors, axis=0),
        feature_names=list(PRNU_FEATURE_NAMES),
    )
    return physics_norm, prnu_norm