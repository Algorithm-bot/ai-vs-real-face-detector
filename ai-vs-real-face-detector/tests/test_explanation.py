"""Tests for human-readable explanations."""

from src.calibration.calibrator import CalibratedPrediction, LabelDecision
from src.explain.explanation import generate_explanation


def test_explanation_from_physics():
    calibrated = CalibratedPrediction(
        label=LabelDecision.REAL,
        real_probability=0.85,
        ai_probability=0.15,
        confidence=0.85,
        uncertainty=0.2,
        calibrated=False,
    )
    physics = {
        "face_detected": 1.0,
        "highlight_iou": 0.6,
        "highlight_consistent": 1.0,
        "light_consistent": 1.0,
        "iris_entropy_mean": 4.2,
    }
    text = generate_explanation(calibrated, physics_features=physics)
    assert "REAL" in text
    assert "Physics" in text or "physics" in text.lower() or "iris" in text


def test_explanation_prnu_limited():
    calibrated = CalibratedPrediction(
        label=LabelDecision.AI_GENERATED,
        real_probability=0.1,
        ai_probability=0.9,
        confidence=0.9,
        uncertainty=0.1,
        calibrated=False,
    )
    text = generate_explanation(
        calibrated,
        prnu_result={"reliability": "limited"},
    )
    assert "limited" in text.lower()
