"""Read/write the sidecar calibration file next to a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def calibration_path_for_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_suffix(path.suffix + ".calibration.json")


def write_calibration(checkpoint: str | Path, payload: Dict[str, Any]) -> Path:
    dest = calibration_path_for_checkpoint(checkpoint)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def load_calibration(checkpoint: str | Path) -> Optional[Dict[str, Any]]:
    """Prefer the sidecar JSON; fall back to keys stored inside the checkpoint."""
    dest = calibration_path_for_checkpoint(checkpoint)
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    if "calibration" in ckpt and isinstance(ckpt["calibration"], dict):
        return ckpt["calibration"]
    args = ckpt.get("args") or {}
    if "calibrated_threshold" in args or "temperature" in args:
        return {
            "threshold": args.get("calibrated_threshold", 0.5),
            "temperature": args.get("temperature", 1.0),
        }
    return None
