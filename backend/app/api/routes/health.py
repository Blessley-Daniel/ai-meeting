"""Health / system-status endpoints.

Also acts as a live environment report: it tells you whether ffmpeg is
installed and which optional ML libraries are importable, which is the
fastest way to diagnose a broken setup during a demo.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["system"])

# Optional-at-import-time dependencies. The API works without them; only the
# pipeline stages that need them require their presence at call time.
_OPTIONAL_PACKAGES = (
    "faster_whisper",
    "torch",
    "transformers",
    "peft",
    "sentence_transformers",
    "docx",
    "reportlab",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@router.get("/health", summary="Liveness and environment report")
def health() -> dict[str, Any]:
    """Return service status plus a snapshot of the runtime environment."""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ffprobe": shutil.which("ffprobe") is not None,
            "packages": {name: _module_available(name) for name in _OPTIONAL_PACKAGES},
        },
        "config": {
            "whisper_model_size": settings.whisper_model_size,
            "whisper_device": settings.whisper_device,
            "extraction_backend": settings.extraction_backend,
            "extraction_model_name": settings.extraction_model_name,
        },
    }
