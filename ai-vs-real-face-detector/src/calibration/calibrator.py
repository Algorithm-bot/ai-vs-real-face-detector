"""Probability calibration, uncertainty, and rejection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class LabelDecision(str, Enum):
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class CalibratedPrediction:
    """Calibrated prediction with uncertainty."""

    label: LabelDecision
    real_probability: float
    ai_probability: float
    confidence: float
    uncertainty: float
    calibrated: bool
    rejection_reason: Optional[str] = None


class ProbabilityCalibrator:
    """Temperature scaling for probability calibration."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(1e-6, temperature)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits, return probabilities."""
        scaled = logits / self.temperature
        exp = np.exp(scaled - scaled.max())
        return exp / exp.sum()

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Recalibrate already-softmaxed probabilities via log-space temperature."""
        probs = np.clip(probs, 1e-8, 1.0)
        logits = np.log(probs)
        return self.calibrate(logits)

    @classmethod
    def fit_temperature(
        cls,
        logits_list: Sequence[np.ndarray],
        labels: Sequence[int],
        init_temp: float = 1.0,
    ) -> "ProbabilityCalibrator":
        """Fit temperature on validation logits via simple grid search."""
        best_temp = init_temp
        best_nll = float("inf")

        for temp in np.linspace(0.5, 3.0, 26):
            nll = 0.0
            for logits, label in zip(logits_list, labels):
                probs = cls(temp).calibrate(logits)
                nll -= np.log(probs[label] + 1e-8)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        return cls(temperature=best_temp)


class UncertaintyEstimator:
    """Estimate prediction uncertainty and decide label with rejection."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        uncertainty_threshold: float = 0.35,
        margin_threshold: float = 0.15,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.margin_threshold = margin_threshold

    def predict_entropy(self, probs: np.ndarray) -> float:
        """Normalized entropy as uncertainty measure [0, 1]."""
        probs = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        return float(entropy / max_entropy)

    def decide(
        self,
        probs: np.ndarray,
        calibrator: Optional[ProbabilityCalibrator] = None,
    ) -> CalibratedPrediction:
        """Decide REAL / AI_GENERATED / UNCERTAIN from probabilities."""
        calibrated = False
        if calibrator is not None:
            probs = calibrator.calibrate_probs(probs)
            calibrated = True

        real_prob = float(probs[0])
        ai_prob = float(probs[1])
        confidence = max(real_prob, ai_prob)
        uncertainty = self.predict_entropy(probs)
        margin = abs(real_prob - ai_prob)

        rejection_reason = None
        if uncertainty > self.uncertainty_threshold:
            label = LabelDecision.UNCERTAIN
            rejection_reason = "high_entropy"
        elif margin < self.margin_threshold:
            label = LabelDecision.UNCERTAIN
            rejection_reason = "low_margin"
        elif confidence < self.confidence_threshold:
            label = LabelDecision.UNCERTAIN
            rejection_reason = "low_confidence"
        elif ai_prob > real_prob:
            label = LabelDecision.AI_GENERATED
        else:
            label = LabelDecision.REAL

        return CalibratedPrediction(
            label=label,
            real_probability=real_prob,
            ai_probability=ai_prob,
            confidence=confidence,
            uncertainty=uncertainty,
            calibrated=calibrated,
            rejection_reason=rejection_reason,
        )

    def calibration_metrics(
        self,
        probs_list: Sequence[np.ndarray],
        labels: Sequence[int],
        n_bins: int = 10,
    ) -> dict:
        """Compute expected calibration error (ECE) and max calibration error."""
        all_probs = np.array([p[1] for p in probs_list])
        all_labels = np.array(labels)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        total = len(all_labels)

        for i in range(n_bins):
            mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = all_labels[mask].mean()
            bin_conf = all_probs[mask].mean()
            gap = abs(bin_acc - bin_conf)
            ece += gap * mask.sum() / total
            mce = max(mce, gap)

        return {
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "n_bins": n_bins,
        }
