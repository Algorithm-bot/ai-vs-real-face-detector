"""Normalize physics feature vectors before fusion."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .feature_vector import PHYSICS_FEATURE_NAMES


# Default statistics computed from typical ranges; can be overridden from training.
_DEFAULT_MEAN = np.array(
    [
        1.0, 0.35, 0.5, 0.0, 0.0,
        0.3, 0.3, 0.1, 0.5, 4.0, 4.0, 4.0, 0.5,
        15.0, 0.85, 0.5, 0.8, 0.8, 0.8, 0.8,
    ],
    dtype=np.float32,
)

_DEFAULT_STD = np.array(
    [
        0.1, 0.25, 0.5, 5.0, 5.0,
        0.2, 0.2, 0.15, 0.3, 1.5, 1.5, 1.0, 0.5,
        20.0, 0.2, 0.5, 0.4, 0.4, 0.4, 0.4,
    ],
    dtype=np.float32,
)


class PhysicsNormalizer:
    """Z-score normalization for physics features."""

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.feature_names = list(feature_names or PHYSICS_FEATURE_NAMES)
        self.mean = mean if mean is not None else _DEFAULT_MEAN.copy()
        self.std = std if std is not None else _DEFAULT_STD.copy()
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Return normalized copy of physics vector."""
        v = np.asarray(vector, dtype=np.float32)
        return (v - self.mean) / self.std

    def normalize_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize a batch of physics vectors (N, D)."""
        v = np.asarray(vectors, dtype=np.float32)
        return (v - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicsNormalizer":
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            feature_names=data.get("feature_names"),
        )

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "PhysicsNormalizer":
        """Fit mean/std from training data."""
        v = np.asarray(vectors, dtype=np.float32)
        mean = v.mean(axis=0)
        std = v.std(axis=0)
        return cls(
            mean=mean,
            std=std,
            feature_names=feature_names,
        )
