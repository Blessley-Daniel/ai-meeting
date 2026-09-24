"""FastAPI application entry point.

Run in development with::

    cd backend
    .venv/bin/uvicorn app.main:app --reload --port 8000

The application is intentionally thin: it wires together configuration,
CORS, static file serving and the versioned API router. All real work lives
in ``app/services``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.database.session import init_db

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare on-disk storage and database tables before serving requests."""
    settings.ensure_directories()
    init_db()
    logger.info("Storage directories and database ready")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Turn a recorded meeting (Google Meet, Zoom, plain audio) into "
        "structured Minutes of Meeting using local, open-source models."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Serve the plain HTML/CSS/JS frontend at "/" when it is present.
if settings.frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(settings.frontend_dir), html=True),
        name="frontend",
    )
