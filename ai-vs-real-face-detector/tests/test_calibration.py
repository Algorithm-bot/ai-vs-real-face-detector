"""Tests for probability calibration."""

import numpy as np

from src.calibration.calibrator import (
    LabelDecision,
    ProbabilityCalibrator,
    UncertaintyEstimator,
)


def test_temperature_scaling():
    cal = ProbabilityCalibrator(temperature=2.0)
    logits = np.array([2.0, 0.5])
    probs = cal.calibrate(logits)
    assert probs.sum() == 1.0
    assert probs[0] > probs[1]


def test_uncertain_on_low_margin():
    est = UncertaintyEstimator(margin_threshold=0.2)
    probs = np.array([0.51, 0.49])
    result = est.decide(probs)
    assert result.label == LabelDecision.UNCERTAIN


def test_ai_decision():
    est = UncertaintyEstimator(confidence_threshold=0.5, margin_threshold=0.1, uncertainty_threshold=0.9)
    probs = np.array([0.05, 0.95])
    result = est.decide(probs)
    assert result.label == LabelDecision.AI_GENERATED


def test_calibration_metrics():
    est = UncertaintyEstimator()
    probs_list = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    labels = [0, 1]
    metrics = est.calibration_metrics(probs_list, labels, n_bins=2)
    assert "expected_calibration_error" in metrics
