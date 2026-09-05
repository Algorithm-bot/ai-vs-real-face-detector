"""Load YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ModelConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    freeze_blocks: int = 5
    checkpoint_path: str = "models/hybrid_best.pt"
    stage1_checkpoint_path: str = "models/stage1_best.pt"
    version: str = "0.1.0"


class InferenceConfig(BaseModel):
    confidence_threshold: float = 0.55
    uncertainty_threshold: float = 0.35
    margin_threshold: float = 0.15
    temperature: float = 1.0
    align_face: bool = False
    default_label_on_low_confidence: str = "real"


class PhysicsConfig(BaseModel):
    highlight_iou_threshold: float = 0.35
    light_angle_threshold_deg: float = 25.0
    light_cosine_threshold: float = 0.85


class DatabaseConfig(BaseModel):
    url: str = "postgresql://postgres:postgres@localhost:5432/face_detector"
    echo: bool = False


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 10
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "logs/app.log"
    log_predictions: bool = True


class AlertsConfig(BaseModel):
    enabled: bool = False
    min_confidence_for_alert: float = 0.9
    alert_on_label: str = "ai"


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


def load_config(path: Optional[Path] = None) -> AppConfig:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return AppConfig()
    with open(config_path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig(**raw)


@lru_cache
def get_config() -> AppConfig:
    return load_config()
