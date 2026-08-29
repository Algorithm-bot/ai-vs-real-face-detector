"""Human-readable explanations from actual model outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.calibration.calibrator import CalibratedPrediction, LabelDecision


def generate_explanation(
    calibrated: CalibratedPrediction,
    physics_features: Optional[Dict[str, float]] = None,
    prnu_result: Optional[Dict[str, Any]] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    face_status: Optional[str] = None,
) -> str:
    """
    Build human-readable explanation based ONLY on actual model outputs.
    No fabricated evidence.
    """
    parts: List[str] = []

    label_str = calibrated.label.value
    parts.append(
        f"Prediction: {label_str} "
        f"(real: {calibrated.real_probability:.1%}, "
        f"AI: {calibrated.ai_probability:.1%}, "
        f"confidence: {calibrated.confidence:.1%})."
    )

    if calibrated.label == LabelDecision.UNCERTAIN:
        reason = calibrated.rejection_reason or "uncertainty"
        parts.append(f"Marked uncertain due to {reason}.")

    if face_status and face_status != "ok":
        parts.append(f"Face detection status: {face_status}.")

    if physics_features:
        physics_lines = _explain_physics(physics_features)
        if physics_lines:
            parts.append("Physics analysis: " + "; ".join(physics_lines) + ".")

    if prnu_result:
        reliability = prnu_result.get("reliability", "limited")
        if reliability == "limited":
            parts.append(
                "PRNU camera fingerprint: evidence limited (no reference camera images available)."
            )
        else:
            score = prnu_result.get("reference_correlation", 0.0)
            parts.append(f"PRNU reference correlation: {score:.3f}.")

    if fusion_weights:
        weight_str = ", ".join(f"{k}: {v:.2f}" for k, v in fusion_weights.items())
        parts.append(f"Modality fusion weights: {weight_str}.")

    return " ".join(parts)


def _explain_physics(features: Dict[str, float]) -> List[str]:
    lines = []

    if features.get("face_detected", 0) < 0.5:
        lines.append("no face detected for physics analysis")
        return lines

    iou = features.get("highlight_iou", 0)
    if features.get("highlight_consistent", 0) >= 0.5:
        lines.append(f"bilateral corneal highlights consistent (IoU={iou:.2f})")
    else:
        lines.append(f"bilateral corneal highlights inconsistent (IoU={iou:.2f})")

    light_consistent = features.get("light_consistent", 0)
    if light_consistent >= 0.5:
        lines.append("light direction consistent across eyes")
    else:
        angle = features.get("light_angle_diff_deg", 0)
        lines.append(f"light direction mismatch ({angle:.1f}° difference)")

    entropy = features.get("iris_entropy_mean", 0)
    lines.append(f"mean iris entropy {entropy:.2f}")

    if features.get("fresnel_plausible", 0) >= 0.5:
        lines.append("Fresnel reflectance physically plausible")
    elif "fresnel_plausible" in features:
        lines.append("Fresnel reflectance deviation detected")

    if features.get("shadow_consistent", 0) >= 0.5:
        lines.append("shadow/illumination geometry consistent")
    elif "shadow_consistent" in features:
        lines.append("shadow asymmetry detected")

    return lines
