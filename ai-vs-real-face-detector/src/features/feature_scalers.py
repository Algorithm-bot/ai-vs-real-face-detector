"""Fit and persist feature normalizers for training checkpoints."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.physics_branch.normalization import PhysicsNormalizer
from src.prnu_branch.extractor import PRNUExtractor, PRNU_FEATURE_NAMES


# Increment this whenever the scaler-fitting or cache representation changes.
SCALER_CACHE_VERSION = 1


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_physics_and_prnu_scalers(
    sample_paths: Sequence[str],
    max_samples: int = 20_000,
    seed: int = 42,
) -> Tuple[PhysicsNormalizer, PhysicsNormalizer]:
    """
    Fit z-score scalers for physics and PRNU vectors on a deterministic
    subsample of training paths. ``max_samples=0`` uses every training path.
    Raises if any feature extraction fails.
    """
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0 (0 means all training images).")

    paths = list(sample_paths)
    if max_samples and len(paths) > max_samples:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_samples)

    physics_extractor = PhysicsFeatureExtractor()
    prnu_extractor = PRNUExtractor()
    physics_vectors: List[np.ndarray] = []
    prnu_vectors: List[np.ndarray] = []

    try:
        # tqdm's default meter includes processed/total, percent, rate,
        # elapsed time, and ETA. It wraps the actual image-processing loop.
        for path in tqdm(
            paths,
            total=len(paths),
            desc="Fitting physics/PRNU scalers",
            unit="image",
            dynamic_ncols=True,
        ):
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


def load_physics_and_prnu_scaler_cache(
    cache_path: Path,
    sample_count: int,
    seed: int,
) -> Optional[Tuple[PhysicsNormalizer, PhysicsNormalizer]]:
    """Load a compatible scaler cache, or return ``None`` to trigger a refit."""
    try:
        with cache_path.open("rb") as cache_file:
            payload = pickle.load(cache_file)
        if (
            payload.get("cache_version") != SCALER_CACHE_VERSION
            or payload.get("sample_count") != sample_count
            or payload.get("seed") != seed
        ):
            return None
        return (
            PhysicsNormalizer.from_dict(payload["physics_scaler"]),
            PhysicsNormalizer.from_dict(payload["prnu_scaler"]),
        )
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        # A stale or interrupted cache is never trusted; the caller refits it.
        return None


def save_physics_and_prnu_scaler_cache(
    cache_path: Path,
    physics_scaler: PhysicsNormalizer,
    prnu_scaler: PhysicsNormalizer,
    sample_count: int,
    seed: int,
) -> None:
    """Atomically persist scaler statistics for safe Kaggle-session reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "cache_version": SCALER_CACHE_VERSION,
        "sample_count": sample_count,
        "seed": seed,
        "physics_scaler": physics_scaler.to_dict(),
        "prnu_scaler": prnu_scaler.to_dict(),
    }

    try:
        with temporary_path.open("wb") as cache_file:
            pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        # A failed write leaves any pre-existing cache intact.
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
