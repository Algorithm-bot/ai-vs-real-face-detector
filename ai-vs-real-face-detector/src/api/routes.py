"""FastAPI routes."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from src.api.schemas import (
    FaceDetectionInfo,
    FeaturesUsed,
    HealthResponse,
    PRNUInfo,
    PredictionRecordResponse,
    PredictionResponse,
    ProbabilityDistribution,
    SuspiciousRegion,
)
from src.config import get_config
from src.db.crud import create_prediction, get_prediction, list_predictions
from src.inference import get_model, predict

router = APIRouter()

_model_loaded = False
_checkpoint_path: Optional[str] = None
_model_mode: Optional[str] = None
_db_session_factory = None

MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 32
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}


def configure(checkpoint_path: str, db_session_factory) -> None:
    global _model_loaded, _checkpoint_path, _db_session_factory, _model_mode
    _checkpoint_path = checkpoint_path
    _db_session_factory = db_session_factory
    _model_loaded = Path(checkpoint_path).exists()
    if _model_loaded:
        import torch
        _, _model_mode = get_model(
            checkpoint_path,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )


def get_db():
    if _db_session_factory is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    db = _db_session_factory()
    try:
        yield db
    finally:
        db.close()


def _validate_image(contents: bytes, max_bytes: int) -> None:
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = Image.open(io.BytesIO(contents))
    if img.format and img.format.upper() not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {img.format}")

    w, h = img.size
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({w}x{h}). Minimum {MIN_IMAGE_DIMENSION}px.",
        )
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({w}x{h}). Maximum {MAX_IMAGE_DIMENSION}px.",
        )


@router.get("/health", response_model=HealthResponse)
def health():
    cfg = get_config()
    return HealthResponse(
        model_loaded=_model_loaded,
        model_version=cfg.model.version,
        architecture=_model_mode or "hybrid",
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not _model_loaded or not _checkpoint_path:
        raise HTTPException(status_code=503, detail="Model checkpoint not found")

    cfg = get_config()
    max_bytes = cfg.api.max_upload_size_mb * 1024 * 1024
    contents = await file.read()
    _validate_image(contents, max_bytes)

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict(
            tmp_path,
            _checkpoint_path,
            include_heatmap=True,
            include_full_explainability=True,
            align_face=cfg.inference.align_face,
        )
        result["filename"] = file.filename
        result["model_version"] = cfg.model.version

        record = create_prediction(db, result)
        return PredictionResponse(
            label=result["label"],
            decision=result.get("decision", result["label"]),
            confidence_score=result["confidence_score"],
            uncertainty=result.get("uncertainty", 0.0),
            probability_distribution=ProbabilityDistribution(**result["probability_distribution"]),
            calibrated=result.get("calibrated", False),
            rejection_reason=result.get("rejection_reason"),
            heatmap=result.get("heatmap"),
            heatmap_error=result.get("heatmap_error"),
            vit_attention=result.get("vit_attention"),
            localization_heatmap=result.get("localization_heatmap"),
            physics_features=result.get("physics_features"),
            prnu=PRNUInfo(**result["prnu"]) if result.get("prnu") else None,
            fusion_weights=result.get("fusion_weights"),
            face_detection=FaceDetectionInfo(**result["face_detection"]) if result.get("face_detection") else None,
            suspicious_regions=[
                SuspiciousRegion(**r) for r in result.get("suspicious_regions", [])
            ] or None,
            patch_scores=result.get("patch_scores"),
            explanation=result.get("explanation"),
            features_used_in_classification=FeaturesUsed(**result.get("features_used_in_classification", {})),
            model_mode=result.get("model_mode"),
            prediction_id=record.id,
            model_version=cfg.model.version,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/predictions", response_model=list[PredictionRecordResponse])
def get_predictions(limit: int = 50, db: Session = Depends(get_db)):
    records = list_predictions(db, limit=limit)
    return [
        PredictionRecordResponse(
            id=r.id,
            created_at=r.created_at,
            filename=r.filename,
            label=r.label,
            confidence_score=r.confidence_score,
            prob_real=r.prob_real,
            prob_ai=r.prob_ai,
            model_version=r.model_version,
        )
        for r in records
    ]


@router.get("/predictions/{prediction_id}", response_model=PredictionRecordResponse)
def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    record = get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionRecordResponse(
        id=record.id,
        created_at=record.created_at,
        filename=record.filename,
        label=record.label,
        confidence_score=record.confidence_score,
        prob_real=record.prob_real,
        prob_ai=record.prob_ai,
        model_version=record.model_version,
    )
