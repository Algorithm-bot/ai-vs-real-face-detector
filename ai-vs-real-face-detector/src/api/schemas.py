"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ProbabilityDistribution(BaseModel):
    real: float
    ai: float


class PredictionResponse(BaseModel):
    label: str
    confidence_score: float
    probability_distribution: ProbabilityDistribution
    heatmap: Optional[str] = None
    heatmap_error: Optional[str] = None
    prediction_id: Optional[int] = None
    model_version: str = "0.1.0"


class PredictionRecordResponse(BaseModel):
    id: int
    created_at: datetime
    filename: Optional[str]
    label: str
    confidence_score: float
    prob_real: float
    prob_ai: float
    model_version: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str
