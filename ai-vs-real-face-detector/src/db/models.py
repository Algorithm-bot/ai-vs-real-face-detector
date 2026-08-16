"""SQLAlchemy models for prediction storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    filename: Mapped[str] = mapped_column(String(512), nullable=True)
    label: Mapped[str] = mapped_column(String(16))
    confidence_score: Mapped[float] = mapped_column(Float)
    prob_real: Mapped[float] = mapped_column(Float)
    prob_ai: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), default="0.1.0")
    heatmap_base64: Mapped[str] = mapped_column(Text, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)


def get_engine(database_url: str):
    return create_engine(database_url, echo=False)


def get_session_factory(database_url: str):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
