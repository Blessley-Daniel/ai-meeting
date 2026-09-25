"""Tests for the Step 1 application skeleton."""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from tests.conftest import requires_ffmpeg


def test_settings_use_project_paths(app_env) -> None:
    settings = get_settings()
    assert settings.upload_dir.name == "uploads"
    assert settings.generated_dir.name == "generated"
    assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024
    assert ".mp4" in settings.allowed_extensions
    assert ".wav" in settings.allowed_extensions


def test_health_endpoint_reports_environment(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    # ffmpeg is a system dependency; the endpoint must report its presence as
    # a boolean rather than crashing when it is missing.
    assert isinstance(body["environment"]["ffmpeg"], bool)
    assert isinstance(body["environment"]["ffprobe"], bool)
    assert "packages" in body["environment"]
    assert "whisper_model_size" in body["config"]


@requires_ffmpeg
def test_health_endpoint_detects_ffmpeg_when_installed(client) -> None:
    """When ffmpeg *is* installed, the health report must say so."""
    body = client.get("/api/health").json()
    assert body["environment"]["ffmpeg"] is True
    assert body["environment"]["ffprobe"] is True


def test_ensure_directories_is_idempotent(app_env) -> None:
    settings = get_settings()
    settings.ensure_directories()
    settings.ensure_directories()
    assert settings.upload_dir.is_dir()
    assert settings.generated_dir.is_dir()


def test_storage_directories_are_gitignored() -> None:
    """Uploads and generated output must never be committed to version control."""
    project_root = Path(__file__).resolve().parents[2]
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert "uploads/*" in gitignore
    assert "generated/*" in gitignore
    assert "*.db" in gitignore
    assert ".env" in gitignore
