"""FastAPI routes."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse, PredictionRecordResponse, PredictionResponse
from src.config import get_config
from src.db.crud import create_prediction, get_prediction, list_predictions
from src.inference import predict

router = APIRouter()

# Set by main.py at startup
_model_loaded = False
_checkpoint_path: Optional[str] = None
_db_session_factory = None


def configure(checkpoint_path: str, db_session_factory) -> None:
    global _model_loaded, _checkpoint_path, _db_session_factory
    _checkpoint_path = checkpoint_path
    _db_session_factory = db_session_factory
    _model_loaded = Path(checkpoint_path).exists()


def get_db():
    if _db_session_factory is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    db = _db_session_factory()
    try:
        yield db
    finally:
        db.close()


@router.get("/health", response_model=HealthResponse)
def health():
    cfg = get_config()
    return HealthResponse(
        model_loaded=_model_loaded,
        model_version=cfg.model.version,
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
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, _checkpoint_path, include_heatmap=True)
        result["filename"] = file.filename
        result["model_version"] = cfg.model.version

        record = create_prediction(db, result)
        return PredictionResponse(
            label=result["label"],
            confidence_score=result["confidence_score"],
            probability_distribution=result["probability_distribution"],
            heatmap=result.get("heatmap"),
            heatmap_error=result.get("heatmap_error"),
            prediction_id=record.id,
            model_version=cfg.model.version,
        )
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
