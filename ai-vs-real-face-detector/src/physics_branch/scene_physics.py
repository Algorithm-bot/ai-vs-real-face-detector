"""Scene-level optical/forensic cues that do not require a face.

These measurements are cheap OpenCV/NumPy statistics intended to catch
illumination, sampling, and compression inconsistencies that generative
models often fail to reproduce on general images (objects, rooms, people).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

SCENE_PHYSICS_FEATURE_NAMES: List[str] = [
    "illumination_uniformity",
    "illumination_gradient_magnitude",
    "left_right_brightness_diff",
    "top_bottom_brightness_diff",
    "shadow_coverage",
    "highlight_coverage",
    "highlight_spatial_entropy",
    "specular_cluster_count",
    "chromaticity_std",
    "rg_channel_correlation",
    "radial_spectrum_slope",
    "high_freq_energy_ratio",
    "jpeg_blockiness",
    "edge_density",
    "gradient_orientation_entropy",
    "defocus_variance",
    "noise_residual_std",
    "color_constancy_error",
    "horizontal_symmetry",
    "lab_chroma_std",
    "chromatic_edge_mismatch",
    "block_texture_regularity",
]


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}.")
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] >= 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        # Accept both conventional [0, 255] arrays and normalized [0, 1]
        # arrays.  The latter is common when the extractor is used outside
        # the OpenCV dataset path.
        upper = 1.0 if np.nanmax(image) <= 1.0 else 255.0
        image = np.clip(image, 0, upper)
        image = (image * 255.0 if upper == 1.0 else image).astype(np.uint8)
    return image


def _gray01(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _radial_spectrum_stats(gray: np.ndarray) -> Tuple[float, float]:
    """Log-log slope of radial FFT power and high-frequency energy ratio."""
    h, w = gray.shape
    size = min(h, w, 256)
    crop = gray[(h - size) // 2 : (h - size) // 2 + size, (w - size) // 2 : (w - size) // 2 + size]
    windowed = crop - crop.mean()
    spec = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(spec) ** 2
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = int(min(cy, cx))
    if max_r < 8:
        return 0.0, 0.0
    radii = np.arange(1, max_r)
    radial = np.array([power[(r >= k) & (r < k + 1)].mean() if np.any((r >= k) & (r < k + 1)) else 0.0 for k in radii])
    radial = np.maximum(radial, 1e-12)
    log_r = np.log(radii.astype(np.float64))
    log_p = np.log(radial)
    slope = float(np.polyfit(log_r, log_p, 1)[0])
    slope_norm = float(np.clip((-slope) / 8.0, 0.0, 1.0))
    nyquist = max_r
    high = radial[int(nyquist * 0.5) :].sum()
    total = radial.sum()
    high_ratio = float(high / (total + 1e-12))
    return slope_norm, high_ratio


def extract_scene_physics(image: np.ndarray) -> np.ndarray:
    """Return a fixed-length scene physics vector (faces not required)."""
    rgb = _to_rgb_uint8(image)
    if min(rgb.shape[:2]) < 8:
        return np.zeros(len(SCENE_PHYSICS_FEATURE_NAMES), dtype=np.float32)

    small = rgb
    h, w = rgb.shape[:2]
    if max(h, w) > 384:
        scale = 384.0 / max(h, w)
        small = cv2.resize(rgb, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)

    gray = _gray01(small)
    mean_l = float(gray.mean())
    std_l = float(gray.std())
    illumination_uniformity = float(np.clip(1.0 - std_l / (mean_l + 1e-6), 0.0, 1.0))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Mean signed derivatives cancel at edges with opposing orientations;
    # average local magnitude measures scene illumination variation instead.
    grad_mag = float(np.mean(np.hypot(gx, gy)))
    illumination_gradient_magnitude = float(np.clip(grad_mag * 10.0, 0.0, 1.0))

    mid_x, mid_y = gray.shape[1] // 2, gray.shape[0] // 2
    left_right = float(np.clip(abs(gray[:, :mid_x].mean() - gray[:, mid_x:].mean()), 0.0, 1.0))
    top_bottom = float(np.clip(abs(gray[:mid_y, :].mean() - gray[mid_y:, :].mean()), 0.0, 1.0))

    shadow_coverage = float((gray < 0.25).mean())
    highlight = gray > 0.90
    highlight_coverage = float(highlight.mean())

    if highlight.any():
        ys, xs = np.nonzero(highlight)
        hist_x, _ = np.histogram(xs, bins=8, range=(0, gray.shape[1]))
        hist_y, _ = np.histogram(ys, bins=8, range=(0, gray.shape[0]))
        highlight_spatial_entropy = float(np.clip((_entropy(hist_x) + _entropy(hist_y)) / 6.0, 0.0, 1.0))
        mask_u8 = highlight.astype(np.uint8)
        n_cc, _ = cv2.connectedComponents(mask_u8)
        specular_cluster_count = float(np.clip((max(n_cc - 1, 0)) / 20.0, 0.0, 1.0))
    else:
        highlight_spatial_entropy = 0.0
        specular_cluster_count = 0.0

    rgb_f = small.astype(np.float32) / 255.0
    sums = rgb_f.sum(axis=2, keepdims=True) + 1e-6
    chroma = rgb_f / sums
    chromaticity_std = float(np.clip(chroma.std() * 8.0, 0.0, 1.0))
    r = rgb_f[:, :, 0].ravel()
    g = rgb_f[:, :, 1].ravel()
    if r.std() < 1e-8 or g.std() < 1e-8:
        rg_corr = 0.0
    else:
        rg_corr = float(np.clip(abs(np.corrcoef(r, g)[0, 1]), 0.0, 1.0))

    slope_norm, high_ratio = _radial_spectrum_stats(gray)

    gray_u8 = (gray * 255).astype(np.uint8)
    block_diffs = []
    for origin in (8,):
        if gray_u8.shape[1] > origin:
            horizontal = gray_u8[:, origin:].astype(np.int16) - gray_u8[:, :-origin].astype(np.int16)
            block_diffs.append(np.mean(np.abs(horizontal)))
        if gray_u8.shape[0] > origin:
            vertical = gray_u8[origin:, :].astype(np.int16) - gray_u8[:-origin, :].astype(np.int16)
            block_diffs.append(np.mean(np.abs(vertical)))
    jpeg_blockiness = float(np.clip(np.mean(block_diffs) / 40.0, 0.0, 1.0)) if block_diffs else 0.0

    edges = cv2.Canny(gray_u8, 60, 140)
    edge_density = float(edges.mean() / 255.0)

    angles = np.arctan2(gy, gx).ravel()
    hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
    gradient_orientation_entropy = float(np.clip(_entropy(hist) / 4.0, 0.0, 1.0))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    defocus_variance = float(np.clip(lap.var() / 0.05, 0.0, 1.0))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual_std = float(np.clip((gray - blur).std() * 12.0, 0.0, 1.0))

    channel_means = rgb_f.reshape(-1, 3).mean(axis=0)
    gray_world = channel_means / (channel_means.mean() + 1e-6)
    color_constancy_error = float(np.clip(np.abs(gray_world - 1.0).mean(), 0.0, 1.0))

    flipped = np.fliplr(gray)
    horizontal_symmetry = float(np.clip(1.0 - np.mean(np.abs(gray - flipped)) * 2.0, 0.0, 1.0))

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_chroma_std = float(np.clip(lab[:, :, 1:].std() / 40.0, 0.0, 1.0))

    r_edge = cv2.Sobel(rgb_f[:, :, 0], cv2.CV_32F, 1, 1, ksize=3)
    g_edge = cv2.Sobel(rgb_f[:, :, 1], cv2.CV_32F, 1, 1, ksize=3)
    chromatic_edge_mismatch = float(np.clip(np.mean(np.abs(r_edge - g_edge)) * 4.0, 0.0, 1.0))

    block = 16
    hh, ww = gray.shape
    vars_ = []
    for y0 in range(0, hh - block + 1, block):
        for x0 in range(0, ww - block + 1, block):
            vars_.append(float(gray[y0 : y0 + block, x0 : x0 + block].var()))
    if vars_:
        block_texture_regularity = float(np.clip(1.0 - np.std(vars_) / (np.mean(vars_) + 1e-6), 0.0, 1.0))
    else:
        block_texture_regularity = 0.0

    values = [
        illumination_uniformity,
        illumination_gradient_magnitude,
        left_right,
        top_bottom,
        shadow_coverage,
        highlight_coverage,
        highlight_spatial_entropy,
        specular_cluster_count,
        chromaticity_std,
        rg_corr,
        slope_norm,
        high_ratio,
        jpeg_blockiness,
        edge_density,
        gradient_orientation_entropy,
        defocus_variance,
        noise_residual_std,
        color_constancy_error,
        horizontal_symmetry,
        lab_chroma_std,
        chromatic_edge_mismatch,
        block_texture_regularity,
    ]
    return np.asarray(values, dtype=np.float32)


def scene_physics_debug(vector: np.ndarray) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(SCENE_PHYSICS_FEATURE_NAMES, vector)}
