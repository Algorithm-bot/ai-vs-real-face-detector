"""
Local single-image inference using a pre-trained checkpoint (from Colab).

Supports legacy hybrid checkpoints and full hybrid architecture with
PRNU, ViT, calibration, localization, and explainability.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.calibrator import ProbabilityCalibrator, UncertaintyEstimator
from src.classifier.head import BranchOnlyClassifier, FullHybridClassifier, HybridClassifier
from src.config import get_config
from src.deep_branch.feature_extractor import DeepClassifier
from src.deep_branch.preprocessing import FacePreprocessor, PreprocessConfig, preprocess_for_model
from src.explain.explanation import generate_explanation
from src.explain.gradcam import GradCAMExplainer
from src.localization.patch_scorer import PatchScorer
from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.prnu_branch.extractor import PRNUExtractor
from src.semantic_branch.encoder import SemanticEncoder


_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, str]] = {}
_SEMANTIC_CACHE: dict[str, SemanticEncoder] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "hybrid")
    args = ckpt.get("args", {})

    if mode == "stage1":
        model = DeepClassifier(
            model_name=args.get("backbone", "efficientnet_b0"),
            freeze_blocks=args.get("freeze_blocks", 5),
            pretrained=False,
        )
    elif mode in {"physics_only", "prnu_only", "semantic_only"}:
        from src.prnu_branch.extractor import PRNU_FEATURE_DIM
        from src.physics_branch.feature_vector import PHYSICS_FEATURE_DIM
        dims = {
            "physics_only": args.get("physics_dim", PHYSICS_FEATURE_DIM),
            "prnu_only": args.get("prnu_dim", PRNU_FEATURE_DIM),
            "semantic_only": args.get("semantic_dim", 384),
        }
        model = BranchOnlyClassifier(
            input_dim=int(dims[mode]),
            normalize=mode in {"physics_only", "prnu_only"},
        )
    elif mode == "full_hybrid":
        from src.fusion.fuse import FusionMode
        fusion_mode = FusionMode(args.get("fusion_mode", "concat"))
        model = FullHybridClassifier(
            model_name=args.get("backbone", "efficientnet_b0"),
            semantic_model=args.get("semantic_model", "vit_small_patch16_224"),
            semantic_pretrained=args.get(
                "semantic_pretrained", not args.get("no_semantic_pretrained", False)
            ),
            freeze_blocks=args.get("freeze_blocks", 5),
            pretrained=False,
            fusion_mode=fusion_mode,
            semantic_dim=args.get("semantic_dim", 384),
            normalize_physics=args.get("normalize_physics", True),
            normalize_prnu=args.get("normalize_prnu", True),
        )
    else:
        model = HybridClassifier(
            model_name=args.get("backbone", "efficientnet_b0"),
            freeze_blocks=args.get("freeze_blocks", 5),
            pretrained=False,
            normalize_physics=args.get("normalize_physics", False),
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    from src.physics_branch.normalization import PhysicsNormalizer
    if mode == "full_hybrid" and isinstance(model, FullHybridClassifier):
        if ckpt.get("physics_normalizer"):
            model.set_feature_scalers(
                PhysicsNormalizer.from_dict(ckpt["physics_normalizer"]),
                PhysicsNormalizer.from_dict(ckpt["prnu_normalizer"]) if ckpt.get("prnu_normalizer") else None,
            )
    elif mode in {"physics_only", "prnu_only"} and isinstance(model, BranchOnlyClassifier):
        key = "physics_normalizer" if mode == "physics_only" else "prnu_normalizer"
        if ckpt.get(key):
            model.set_normalizer(PhysicsNormalizer.from_dict(ckpt[key]))
    model.to(device)
    model.eval()
    return model, mode


def get_model(checkpoint_path: str, device: torch.device):
    """Return one cached model instance per checkpoint/device pair."""
    key = (str(Path(checkpoint_path).resolve()), str(device))
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = load_model(key[0], device)
        return _MODEL_CACHE[key]


def _get_semantic_encoder(
    model_name: str, device: torch.device, pretrained: bool = True
) -> SemanticEncoder:
    cache_key = f"{model_name}:{pretrained}:{device}"
    with _MODEL_CACHE_LOCK:
        if cache_key not in _SEMANTIC_CACHE:
            _SEMANTIC_CACHE[cache_key] = SemanticEncoder(
                model_name=model_name, pretrained=pretrained, device=device
            )
        return _SEMANTIC_CACHE[cache_key]


def _enrich_physics_dict(physics_features: Dict[str, float], physics_result) -> Dict[str, float]:
    enriched = dict(physics_features)
    if physics_result.shadow and physics_result.shadow.shadow_map:
        enriched["shadow_coverage"] = physics_result.shadow.shadow_map.shadow_coverage
        enriched["shadow_asymmetry"] = physics_result.shadow.shadow_map.shadow_asymmetry
        enriched["illumination_uniformity"] = physics_result.shadow.shadow_map.illumination_uniformity
        enriched["shadow_consistent"] = float(physics_result.shadow.consistent)
    fresnel = physics_result.metadata.get("fresnel", {})
    if fresnel:
        enriched["fresnel_plausible"] = float(
            fresnel.get("left_plausible", False) and fresnel.get("right_plausible", False)
        )
        enriched["fresnel_consistency"] = float(fresnel.get("bilateral_consistency", 0.0))
    return enriched


def predict(
    image_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    include_heatmap: bool = True,
    include_full_explainability: bool = True,
    align_face: bool = False,
) -> Dict[str, Any]:
    cfg = get_config()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mode = get_model(checkpoint_path, device_t)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pre_cfg = PreprocessConfig(align_face=align_face, is_bgr=False)
    preprocessor = FacePreprocessor(pre_cfg)
    tensor, align_meta = preprocess_for_model(rgb, preprocessor=preprocessor, return_meta=True)

    tensor = tensor.to(device_t)

    physics_features: Optional[Dict[str, float]] = None
    physics_t: Optional[torch.Tensor] = None
    prnu_t: Optional[torch.Tensor] = None
    semantic_t: Optional[torch.Tensor] = None
    physics_result = None
    prnu_result = None
    semantic_result = None

    with torch.no_grad():
        if mode in {"physics_only", "prnu_only", "semantic_only"}:
            with PhysicsFeatureExtractor() as physics_ext:
                physics_result = physics_ext.extract(rgb)
                physics_features = _enrich_physics_dict(physics_result.to_dict(), physics_result)
                physics_t = torch.from_numpy(physics_result.vector).unsqueeze(0).to(device_t)
            if mode == "physics_only":
                feats = physics_t
            elif mode == "prnu_only":
                prnu_result = PRNUExtractor().extract(rgb)
                prnu_t = torch.from_numpy(prnu_result.vector).unsqueeze(0).to(device_t)
                feats = prnu_t
            else:
                semantic_enc = _get_semantic_encoder("vit_small_patch16_224", device_t)
                semantic_result = semantic_enc.extract(rgb, return_attention=True)
                semantic_t = torch.from_numpy(semantic_result.features).unsqueeze(0).to(device_t)
                feats = semantic_t
            logits, _ = model(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            fusion_weights = {mode.replace("_only", ""): 1.0}
        elif mode in ("hybrid", "full_hybrid"):
            with PhysicsFeatureExtractor() as physics_ext:
                physics_result = physics_ext.extract(rgb)
                physics = physics_result.vector
                physics_features = _enrich_physics_dict(physics_result.to_dict(), physics_result)
            physics_t = torch.from_numpy(physics).unsqueeze(0).to(device_t)

            if mode == "full_hybrid" and isinstance(model, FullHybridClassifier):
                prnu_ext = PRNUExtractor()
                prnu_result = prnu_ext.extract(rgb)
                semantic_model = model.semantic_model
                semantic_enc = _get_semantic_encoder(
                    semantic_model, device_t, model.semantic_pretrained
                )
                semantic_result = semantic_enc.extract(rgb, return_attention=True)
                prnu_t = torch.from_numpy(prnu_result.vector).unsqueeze(0).to(device_t)
                semantic_t = torch.from_numpy(semantic_result.features).unsqueeze(0).to(device_t)
                probs = model.predict_proba(tensor, physics_t, prnu_t, semantic_t)[0].cpu().numpy()
                fusion_weights = model.get_fusion_weights(tensor, physics_t, prnu_t, semantic_t)
            else:
                probs = model.predict_proba(tensor, physics_t)[0].cpu().numpy()
                fusion_weights = {"deep": 0.5, "physics": 0.5}
                prnu_result = PRNUExtractor().extract(rgb) if include_full_explainability else None
                semantic_result = None
        else:
            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            fusion_weights = {"deep": 1.0}
            prnu_result = PRNUExtractor().extract(rgb) if include_full_explainability else None
            semantic_result = None

    # Extract semantic/ViT for explainability even if not in classifier
    if include_full_explainability and semantic_result is None:
        try:
            semantic_enc = _get_semantic_encoder("vit_small_patch16_224", device_t)
            semantic_result = semantic_enc.extract(rgb, return_attention=True)
        except Exception:
            semantic_result = None

    if include_full_explainability and prnu_result is None:
        prnu_result = PRNUExtractor().extract(rgb)

    calibrator = ProbabilityCalibrator(temperature=cfg.inference.temperature)
    uncertainty_est = UncertaintyEstimator(
        confidence_threshold=cfg.inference.confidence_threshold,
        uncertainty_threshold=cfg.inference.uncertainty_threshold,
        margin_threshold=cfg.inference.margin_threshold,
    )
    calibrated = uncertainty_est.decide(probs, calibrator=calibrator)

    # Face alignment is optional and disabled for general-image inference.
    # Do not report a fabricated face when no detector was requested.
    face_status = align_meta.status.value if align_meta else "not_requested"

    result: Dict[str, Any] = {
        "label": calibrated.label.value.lower().replace("_generated", ""),
        "decision": calibrated.label.value,
        "confidence_score": round(calibrated.confidence * 100.0, 2),
        "uncertainty": round(calibrated.uncertainty, 4),
        "probability_distribution": {
            "real": round(calibrated.real_probability * 100, 2),
            "ai": round(calibrated.ai_probability * 100, 2),
        },
        "calibrated": calibrated.calibrated,
        "rejection_reason": calibrated.rejection_reason,
        "heatmap": None,
        "physics_features": physics_features,
        "face_detection": {
            "status": face_status,
            "face_count": align_meta.face_count if align_meta else 0,
        },
        "fusion_weights": fusion_weights,
        "model_mode": mode,
        "features_used_in_classification": {
            "deep": mode in {"stage1", "hybrid", "full_hybrid"},
            "physics": mode in {"hybrid", "full_hybrid", "physics_only"},
            "prnu": mode in {"full_hybrid", "prnu_only"},
            "semantic": mode in {"full_hybrid", "semantic_only"},
        },
    }

    if prnu_result:
        result["prnu"] = {
            "score": float(prnu_result.vector[7]),
            "reliability": prnu_result.reliability,
            "has_reference": prnu_result.has_reference,
            "reference_correlation": prnu_result.reference_correlation,
            "features": prnu_result.to_dict(),
            "note": prnu_result.metadata.get("note", ""),
        }

    if semantic_result and semantic_result.attention_map:
        result["vit_attention"] = semantic_result.attention_map.base64_png

    if include_full_explainability:
        result["explanation"] = generate_explanation(
            calibrated,
            physics_features=physics_features,
            prnu_result=result.get("prnu"),
            fusion_weights=fusion_weights,
            face_status=face_status if align_face else None,
        )

    if include_heatmap:
        try:
            explainer = GradCAMExplainer(
                model,
                use_cuda=device_t.type == "cuda",
                physics_features=physics_t,
                prnu_features=prnu_t,
                semantic_features=semantic_t,
            )
            try:
                class_idx = 1 if calibrated.ai_probability > calibrated.real_probability else 0
                result["heatmap"] = explainer.generate_base64_png(tensor, target_class=class_idx)
            finally:
                explainer.close()
        except Exception as exc:
            result["heatmap_error"] = f"{type(exc).__name__}: {exc}"

    if include_full_explainability:
        try:
            patch_scorer = PatchScorer()
            loc_result = patch_scorer.score_patches(
                rgb, model, device_t, physics_features=physics_t, mode=mode,
            )
            result["suspicious_regions"] = [
                {"bbox": r.bbox, "score": r.score, "label": r.label}
                for r in loc_result.suspicious_regions
            ]
            result["localization_heatmap"] = loc_result.base64_png
            result["patch_scores"] = {
                "max": loc_result.max_patch_score,
                "mean": loc_result.mean_patch_score,
            }
        except Exception as exc:
            result["localization_error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="models/hybrid_best.pt")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-align", action="store_true", help="Deprecated; alignment is off by default.")
    parser.add_argument("--align-face", action="store_true", help="Crop/align a face before scoring (optional).")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = predict(
        args.image,
        args.checkpoint,
        include_heatmap=not args.no_heatmap,
        align_face=args.align_face,
    )
    output = json.dumps(result, indent=2, default=str)
    print(output)
    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
