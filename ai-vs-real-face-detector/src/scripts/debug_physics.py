"""
Debug script for physics branch — visualize intermediate outputs on a single image.

Safe to run locally (no GPU, no training).

Usage:
  python src/scripts/debug_physics.py path/to/portrait.jpg --output-dir debug_out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics_branch.corneal_reflection import analyze_corneal_reflections
from src.physics_branch.feature_vector import PhysicsFeatureExtractor
from src.physics_branch.iris_pupil import analyze_iris_pupil
from src.physics_branch.region_detection import FaceRegionDetector, draw_landmarks_debug
from src.physics_branch.shadow_geometry import analyze_shadow_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug physics branch on one image")
    parser.add_argument("image", type=str)
    parser.add_argument("--output-dir", type=str, default="debug_out")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read: {args.image}")

    with FaceRegionDetector() as detector:
        landmarks = detector.detect(bgr)

    if not landmarks.detected:
        print("No face detected.")
        return

    corneal = analyze_corneal_reflections(landmarks)
    iris = analyze_iris_pupil(landmarks)
    shadow = analyze_shadow_geometry(landmarks, corneal)

    with PhysicsFeatureExtractor() as ext:
        features = ext.extract_from_landmarks(landmarks, corneal, iris, shadow)

    cv2.imwrite(str(out_dir / "landmarks.jpg"), draw_landmarks_debug(bgr, landmarks))

    for side, eye, det in (
        ("left", landmarks.left_eye, corneal.left),
        ("right", landmarks.right_eye, corneal.right),
    ):
        if eye is not None:
            cv2.imwrite(
                str(out_dir / f"{side}_eye.jpg"),
                cv2.cvtColor(eye.crop, cv2.COLOR_RGB2BGR),
            )
        if det.edges is not None:
            cv2.imwrite(str(out_dir / f"{side}_edges.jpg"), det.edges)
        if det.mask is not None:
            cv2.imwrite(str(out_dir / f"{side}_highlight.jpg"), det.mask)

    report = {
        "feature_vector": features.to_dict(),
        "corneal": {
            "highlight_iou": corneal.highlight_iou,
            "consistent": corneal.consistent,
        },
        "iris_pupil": {
            "left_entropy": iris.left_iris_entropy,
            "right_entropy": iris.right_iris_entropy,
            "pupil_regularity_mean": iris.pupil_regularity_mean,
        },
        "shadow": {
            "angle_difference_deg": shadow.angle_difference_deg,
            "cosine_similarity": shadow.vector_cosine_similarity,
            "consistent": shadow.consistent,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Debug artifacts saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
