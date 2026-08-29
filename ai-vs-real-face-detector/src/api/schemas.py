"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProbabilityDistribution(BaseModel):
    real: float
    ai: float


class FaceDetectionInfo(BaseModel):
    status: str
    face_count: int = 0


class PRNUInfo(BaseModel):
    score: float
    reliability: str
    has_reference: bool = False
    reference_correlation: float = 0.0
    note: Optional[str] = None


class FeaturesUsed(BaseModel):
    deep: bool = True
    physics: bool = False
    prnu: bool = False
    semantic: bool = False


class SuspiciousRegion(BaseModel):
    bbox: List[int]
    score: float
    label: str


class PredictionResponse(BaseModel):
    label: str
    decision: str = "REAL"
    confidence_score: float
    uncertainty: float = 0.0
    probability_distribution: ProbabilityDistribution
    calibrated: bool = False
    rejection_reason: Optional[str] = None
    heatmap: Optional[str] = None
    heatmap_error: Optional[str] = None
    vit_attention: Optional[str] = None
    localization_heatmap: Optional[str] = None
    physics_features: Optional[Dict[str, float]] = None
    prnu: Optional[PRNUInfo] = None
    fusion_weights: Optional[Dict[str, float]] = None
    face_detection: Optional[FaceDetectionInfo] = None
    suspicious_regions: Optional[List[SuspiciousRegion]] = None
    patch_scores: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    features_used_in_classification: Optional[FeaturesUsed] = None
    model_mode: Optional[str] = None
    prediction_id: Optional[int] = None
    model_version: str = "0.2.0"


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
    architecture: str = "hybrid"
