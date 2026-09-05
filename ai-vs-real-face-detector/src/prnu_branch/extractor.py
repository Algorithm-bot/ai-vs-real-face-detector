"""PRNU noise-residual forensic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

PRNU_FEATURE_NAMES: List[str] = [
    "noise_mean",
    "noise_std",
    "noise_skew",
    "noise_kurtosis",
    "noise_energy",
    "spatial_autocorr",
    "high_freq_ratio",
    "prnu_score",
    "reliability",
]

PRNU_FEATURE_DIM = len(PRNU_FEATURE_NAMES)


@dataclass
class PRNUResult:
    """PRNU forensic analysis result."""

    vector: np.ndarray
    names: List[str] = field(default_factory=lambda: list(PRNU_FEATURE_NAMES))
    reliability: str = "limited"
    has_reference: bool = False
    reference_correlation: float = 0.0
    noise_residual: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.names, self.vector)}


def extract_noise_residual(
    image: np.ndarray,
    denoise_strength: float = 10.0,
) -> np.ndarray:
    """
    Extract noise residual via denoising filter subtraction.
    Standard PRNU preprocessing step.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.float32)

    gray_f = gray.astype(np.float32)
    denoised = cv2.fastNlMeansDenoising(
        gray.astype(np.uint8),
        None,
        h=int(denoise_strength),
        templateWindowSize=7,
        searchWindowSize=21,
    )
    residual = gray_f - denoised.astype(np.float32)
    return residual


def _spatial_autocorrelation(residual: np.ndarray, max_lag: int = 5) -> float:
    """Mean normalized autocorrelation at small lags."""
    h, w = residual.shape
    center = residual - residual.mean()
    var = center.var()
    if var < 1e-8:
        return 0.0
    corrs = []
    for lag in range(1, max_lag + 1):
        c = np.mean(center[:, lag:] * center[:, :-lag]) / var
        corrs.append(float(c))
    return float(np.mean(corrs))


def _high_freq_ratio(residual: np.ndarray) -> float:
    """Ratio of high-frequency energy to total energy."""
    fft = np.fft.fft2(residual)
    power = np.abs(fft) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    mask_low = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_low, (cx, cy), radius, 1, -1)
    low_energy = power[mask_low > 0].sum()
    total = power.sum()
    if total < 1e-8:
        return 0.0
    return float(1.0 - low_energy / total)


def correlate_with_reference(
    residual: np.ndarray,
    reference_residual: np.ndarray,
) -> float:
    """Normalized cross-correlation with reference camera PRNU pattern."""
    r1 = residual.astype(np.float64).ravel()
    r2 = reference_residual.astype(np.float64).ravel()
    min_len = min(len(r1), len(r2))
    r1, r2 = r1[:min_len], r2[:min_len]
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(r1, r2) / (n1 * n2))


class PRNUExtractor:
    """Extract camera/noise forensic features from arbitrary RGB images."""

    def __init__(
        self,
        reference_residual: Optional[np.ndarray] = None,
        denoise_strength: float = 10.0,
    ) -> None:
        self.reference_residual = reference_residual
        self.denoise_strength = denoise_strength

    def extract(
        self,
        image: np.ndarray,
        return_residual: bool = False,
    ) -> PRNUResult:
        residual = extract_noise_residual(image, self.denoise_strength)
        flat = residual.ravel()

        mean = float(flat.mean())
        std = float(flat.std())
        if std < 1e-8:
            skew, kurt = 0.0, 0.0
        else:
            normed = (flat - mean) / std
            skew = float(np.mean(normed ** 3))
            kurt = float(np.mean(normed ** 4) - 3.0)

        energy = float(np.mean(residual ** 2))
        autocorr = _spatial_autocorrelation(residual)
        hf_ratio = _high_freq_ratio(residual)

        has_ref = self.reference_residual is not None
        if has_ref:
            ref_corr = correlate_with_reference(residual, self.reference_residual)
            reliability = "reference_available"
            prnu_score = ref_corr
        else:
            ref_corr = 0.0
            reliability = "limited"
            # Without reference, score reflects noise pattern regularity only
            prnu_score = float(np.clip(autocorr * hf_ratio, 0.0, 1.0))

        reliability_flag = 1.0 if has_ref else 0.0

        vector = np.array(
            [
                mean, std, skew, kurt, energy,
                autocorr, hf_ratio, prnu_score, reliability_flag,
            ],
            dtype=np.float32,
        )

        return PRNUResult(
            vector=vector,
            reliability=reliability,
            has_reference=has_ref,
            reference_correlation=ref_corr,
            noise_residual=residual if return_residual else None,
            metadata={
                "note": (
                    "PRNU fingerprint matching requires reference camera images. "
                    "Without reference, evidence is limited to noise statistics only."
                    if not has_ref
                    else "Reference camera PRNU pattern available."
                ),
            },
        )
