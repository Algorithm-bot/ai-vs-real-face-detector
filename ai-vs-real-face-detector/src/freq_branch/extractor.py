"""
Frequency and upsampling forensic feature extraction.

Rule: Extracted from the full-resolution uncropped raw image (before 224 crop)
to preserve native sampling artifacts, checkerboards, and periodic peaks.

Features (64-d float32 vector, documented below):
1. 1D radially averaged power spectrum of the FFT magnitude (32 bins, normalised so sum=1).
2. Periodic peak score: detect regular peaks in the 2D FFT magnitude that correspond to
   upsampling/transposed-conv grid artifacts (top-k peak energy / median energy, 20 values across radii).
3. Spatial co-occurrence / neighbor pixel relationship (NPR) residual statistics:
   compute difference between an image and its 3x3 median-filtered version;
   collect mean, std, skewness, kurtosis for each of R, G, B (12 dims).

Total freq feature dim = 32 + 20 + 12 = 64 dims. NaN/inf safe, vectorized NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

FREQ_FEATURE_DIM = 64

FEATURE_NAMES = (
    [f"radial_spectrum_bin_{i:02d}" for i in range(32)]
    + [f"periodic_peak_ratio_band_{i:02d}" for i in range(20)]
    + [f"npr_{c}_{stat}" for c in ("r", "g", "b") for stat in ("mean", "std", "skew", "kurt")]
)


@dataclass
class FreqResult:
    vector: np.ndarray
    names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.names, self.vector)}


def _radially_averaged_magnitude_spectrum(mag: np.ndarray, r: np.ndarray, max_r: float, n_bins: int = 32) -> np.ndarray:
    """1D radially averaged power spectrum of the FFT magnitude (32 bins, normalised so sum=1)."""
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    radial_profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = float(np.mean(mag[mask]))
        else:
            radial_profile[i] = 0.0

    s = float(np.sum(radial_profile))
    if s > 1e-12:
        radial_profile /= s
    return radial_profile


def _periodic_peak_scores(mag: np.ndarray, r: np.ndarray, max_r: float, n_bands: int = 20) -> np.ndarray:
    """Periodic peak score: top-k peak energy / median energy across 20 radial bands."""
    band_edges = np.linspace(0, max_r, n_bands + 1)
    peaks = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        if np.any(mask):
            vals = mag[mask]
            med = float(np.median(vals)) + 1e-8
            k = max(1, int(len(vals) * 0.01))
            top_k = float(np.mean(np.sort(vals)[-k:]))
            peaks[i] = top_k / med
        else:
            peaks[i] = 1.0
    return peaks


def _npr_residual_statistics(image_rgb: np.ndarray) -> np.ndarray:
    """
    Difference between an image and its 3x3 median-filtered version.
    Collect mean, std, skewness, kurtosis for each of R, G, B (12 dims).
    """
    med_filtered = cv2.medianBlur(image_rgb, 3)
    diff = image_rgb.astype(np.float32) - med_filtered.astype(np.float32)
    stats = []

    for c in range(3):
        ch_diff = diff[:, :, c].flatten()
        mean = float(np.mean(ch_diff))
        std = float(np.std(ch_diff))
        if std > 1e-8:
            skew = float(np.mean(((ch_diff - mean) / std) ** 3))
            kurt = float(np.mean(((ch_diff - mean) / std) ** 4) - 3.0)
        else:
            skew = 0.0
            kurt = 0.0
        stats.extend([mean, std, skew, kurt])

    return np.asarray(stats, dtype=np.float32)


class FreqExtractor:
    """Frequency and upsampling artifact feature extractor."""

    def __init__(self) -> None:
        pass

    def extract(self, image: np.ndarray) -> FreqResult:
        """
        Extract 64-d frequency artifact vector from unresized center 224x224 crop.
        NaN/inf safe, float32 output.
        """
        from src.preprocess import center_crop_numpy, DEFAULT_CROP_SIZE

        # Ensure 3-channel RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        h, w = image.shape[:2]
        if h != DEFAULT_CROP_SIZE or w != DEFAULT_CROP_SIZE:
            image = center_crop_numpy(image, DEFAULT_CROP_SIZE)

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.abs(fft)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy)
        max_r = min(cy, cx)
        if max_r <= 0:
            max_r = 1.0

        # 1. 32 radial bins normalized sum=1
        rad_spec = _radially_averaged_magnitude_spectrum(mag, r, max_r, n_bins=32)

        # 2. 20 periodic peak scores across radii
        peaks = _periodic_peak_scores(mag, r, max_r, n_bands=20)

        # 3. 12 NPR neighbor statistics
        npr_stats = _npr_residual_statistics(image)

        vec = np.concatenate([rad_spec, peaks, npr_stats]).astype(np.float32)
        # NaN / inf safety
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        assert len(vec) == FREQ_FEATURE_DIM, f"Expected {FREQ_FEATURE_DIM}, got {len(vec)}"
        return FreqResult(vector=vec)
