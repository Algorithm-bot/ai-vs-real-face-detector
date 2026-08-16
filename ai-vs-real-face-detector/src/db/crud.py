"""CRUD operations for predictions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import PredictionRecord


def create_prediction(session: Session, data: Dict[str, Any]) -> PredictionRecord:
    probs = data.get("probability_distribution", {})
    record = PredictionRecord(
        filename=data.get("filename"),
        label=data["label"],
        confidence_score=data["confidence_score"],
        prob_real=probs.get("real", 0.0),
        prob_ai=probs.get("ai", 0.0),
        model_version=data.get("model_version", "0.1.0"),
        heatmap_base64=data.get("heatmap"),
        error_message=data.get("error_message"),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_prediction(session: Session, prediction_id: int) -> Optional[PredictionRecord]:
    return session.get(PredictionRecord, prediction_id)


def list_predictions(session: Session, limit: int = 50) -> List[PredictionRecord]:
    stmt = (
        select(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())
