"""Application entry point.

Run with::

    python -m uvicorn fastApi.main:app --reload
"""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from fastApi.core.config import get_frontend_origin
from fastApi.routers import (
    diff,
    health,
    metrics,
    models,
    outputs,
    runs,
    text,
    transcription,
)

ROUTERS = (
    health.router,
    models.router,
    metrics.router,
    text.router,
    diff.router,
    runs.router,
    outputs.router,
    transcription.router,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Transcription API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[get_frontend_origin()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in ROUTERS:
        app.include_router(router)

    return app


app = create_app()
