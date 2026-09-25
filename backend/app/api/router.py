"""Aggregate router for the versioned API.

Feature routers are added here as they are implemented in later steps, so
``main.py`` only ever has to include this single object.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, history, meetings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meetings.router)
api_router.include_router(history.router)
