"""Patch/region-level AI scoring and suspicious region heatmaps."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class SuspiciousRegion:
    """A detected suspicious region."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float
    label: str


@dataclass
class LocalizationResult:
    """Patch-level localization output."""

    heatmap: np.ndarray
    patch_scores: np.ndarray
    grid_size: Tuple[int, int]
    suspicious_regions: List[SuspiciousRegion]
    base64_png: Optional[str] = None
    max_patch_score: float = 0.0
    mean_patch_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class PatchScorer:
    """Score image patches for AI-generation suspicion."""

    def __init__(
        self,
        grid_rows: int = 7,
        grid_cols: int = 7,
        ai_threshold: float = 0.6,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.ai_threshold = ai_threshold

    def score_patches(
        self,
        image: np.ndarray,
        model: nn.Module,
        device: torch.device,
        physics_features: Optional[torch.Tensor] = None,
        mode: str = "hybrid",
    ) -> LocalizationResult:
        """
        Slide a grid over the image, score each patch independently.
        Returns heatmap of per-patch AI probabilities.
        """
        h, w = image.shape[:2]
        patch_h = h // self.grid_rows
        patch_w = w // self.grid_cols

        scores = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for gy in range(self.grid_rows):
            for gx in range(self.grid_cols):
                y0 = gy * patch_h
                x0 = gx * patch_w
                y1 = min(y0 + patch_h, h)
                x1 = min(x0 + patch_w, w)
                patch = image[y0:y1, x0:x1]

                if patch.size == 0:
                    continue

                from src.deep_branch.preprocessing import preprocess_for_model

                tensor = preprocess_for_model(patch).to(device)

                with torch.no_grad():
                    if mode == "hybrid" and physics_features is not None:
                        probs = model.predict_proba(tensor, physics_features)[0].cpu().numpy()
                    elif hasattr(model, "predict_proba"):
                        probs = model.predict_proba(tensor)[0].cpu().numpy()
                    else:
                        logits, _ = model(tensor)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                scores[gy, gx] = float(probs[1])  # AI probability

        heatmap = cv2.resize(
            scores,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        suspicious = self._find_suspicious_regions(scores, patch_h, patch_w)
        base64_png = self._to_base64(heatmap, image)

        return LocalizationResult(
            heatmap=heatmap,
            patch_scores=scores,
            grid_size=(self.grid_rows, self.grid_cols),
            suspicious_regions=suspicious,
            base64_png=base64_png,
            max_patch_score=float(scores.max()),
            mean_patch_score=float(scores.mean()),
        )

    def _find_suspicious_regions(
        self,
        scores: np.ndarray,
        patch_h: int,
        patch_w: int,
    ) -> List[SuspiciousRegion]:
        regions = []
        for gy in range(self.grid_rows):
            for gx in range(self.grid_cols):
                score = scores[gy, gx]
                if score >= self.ai_threshold:
                    regions.append(
                        SuspiciousRegion(
                            bbox=(gx * patch_w, gy * patch_h, patch_w, patch_h),
                            score=score,
                            label="suspicious",
                        )
                    )
        return sorted(regions, key=lambda r: r.score, reverse=True)[:5]

    def _to_base64(self, heatmap: np.ndarray, image: np.ndarray) -> str:
        colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_HOT,
        )
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = (0.4 * image.astype(np.float32) + 0.6 * colored).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
