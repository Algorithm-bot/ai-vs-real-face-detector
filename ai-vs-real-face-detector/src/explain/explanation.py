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
        parts.append(f"Optional face alignment status: {face_status}.")

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

    uniformity = features.get("illumination_uniformity")
    if uniformity is not None:
        lines.append(f"illumination uniformity {uniformity:.2f}")

    high_freq = features.get("high_freq_energy_ratio")
    if high_freq is not None:
        lines.append(f"high-frequency energy ratio {high_freq:.2f}")

    blockiness = features.get("jpeg_blockiness")
    if blockiness is not None:
        lines.append(f"block/grid artifact score {blockiness:.2f}")

    shadow = features.get("shadow_coverage")
    if shadow is not None:
        lines.append(f"shadow coverage {shadow:.2f}")

    chroma = features.get("lab_chroma_std")
    if chroma is not None:
        lines.append(f"chroma variation {chroma:.2f}")

    texture = features.get("block_texture_regularity")
    if texture is not None:
        lines.append(f"block texture regularity {texture:.2f}")

    return lines
