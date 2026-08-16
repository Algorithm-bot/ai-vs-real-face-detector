"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api import routes
from src.config import get_config
from src.db.models import get_session_factory
from src.monitoring.logger import logger, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def create_app() -> FastAPI:
    setup_logger()
    cfg = get_config()

    app = FastAPI(
        title="AI vs Real Face Detector",
        description="Hybrid deep learning + physics-based face authenticity classifier",
        version=cfg.model.version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    checkpoint = PROJECT_ROOT / cfg.model.checkpoint_path
    session_factory = get_session_factory(cfg.database.url)
    routes.configure(str(checkpoint), session_factory)
    app.include_router(routes.router, prefix="/api/v1")

    if DASHBOARD_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

    @app.get("/", response_class=HTMLResponse)
    def root():
        return """
        <html><body style="font-family:sans-serif;padding:2rem">
        <h1>AI vs Real Face Detector</h1>
        <p>API docs: <a href="/docs">/docs</a></p>
        <p>Dashboard: <a href="/dashboard/">/dashboard/</a></p>
        </body></html>
        """

    logger.info("App started. Checkpoint exists: %s", checkpoint.exists())
    return app


app = create_app()
