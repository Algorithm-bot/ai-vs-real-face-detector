"""
Local single-image inference using a pre-trained checkpoint (from Colab).

Does NOT train models. Safe to run on 8GB RAM laptop for one image at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.head import HybridClassifier
from src.deep_branch.feature_extractor import DeepClassifier
from src.deep_branch.preprocessing import preprocess_for_model
from src.explain.gradcam import GradCAMExplainer
from src.physics_branch.feature_vector import PhysicsFeatureExtractor


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    mode = ckpt.get("mode", "hybrid")
    args = ckpt.get("args", {})

    if mode == "stage1":
        model = DeepClassifier(
            model_name=args.get("backbone", "efficientnet_b0"),
            freeze_blocks=args.get("freeze_blocks", 5),
            pretrained=False,
        )
    else:
        model = HybridClassifier(
            model_name=args.get("backbone", "efficientnet_b0"),
            freeze_blocks=args.get("freeze_blocks", 5),
            pretrained=False,
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, mode


def predict(
    image_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    include_heatmap: bool = True,
) -> Dict[str, Any]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mode = load_model(checkpoint_path, device_t)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    tensor = preprocess_for_model(rgb).to(device_t)

    with torch.no_grad():
        if mode == "hybrid" and isinstance(model, HybridClassifier):
            with PhysicsFeatureExtractor() as physics_ext:
                physics = physics_ext.extract(rgb).vector
            physics_t = torch.from_numpy(physics).unsqueeze(0).to(device_t)
            probs = model.predict_proba(tensor, physics_t)[0].cpu().numpy()
        else:
            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_idx = int(np.argmax(probs))
    label = "real" if class_idx == 0 else "ai"
    confidence = float(probs[class_idx] * 100.0)

    result: Dict[str, Any] = {
        "label": label,
        "confidence_score": round(confidence, 2),
        "probability_distribution": {
            "real": round(float(probs[0]) * 100, 2),
            "ai": round(float(probs[1]) * 100, 2),
        },
        "heatmap": None,
    }

    if include_heatmap:
        try:
            explainer = GradCAMExplainer(model, use_cuda=device_t.type == "cuda")
            result["heatmap"] = explainer.generate_base64_png(tensor, target_class=class_idx)
        except Exception as exc:
            result["heatmap_error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single face image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="models/hybrid_best.pt")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = predict(
        args.image,
        args.checkpoint,
        include_heatmap=not args.no_heatmap,
    )
    output = json.dumps(result, indent=2)
    print(output)
    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
