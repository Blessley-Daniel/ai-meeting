"""Tests for the Step 1 application skeleton."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)


def test_settings_use_project_paths() -> None:
    settings = get_settings()
    assert settings.upload_dir.name == "uploads"
    assert settings.generated_dir.name == "generated"
    assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024
    assert ".mp4" in settings.allowed_extensions
    assert ".wav" in settings.allowed_extensions


def test_health_endpoint_reports_environment() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    # ffmpeg was installed as a system dependency, so it must be detected.
    assert body["environment"]["ffmpeg"] is True
    assert "packages" in body["environment"]
    assert "whisper_model_size" in body["config"]


def test_ensure_directories_is_idempotent() -> None:
    settings = get_settings()
    settings.ensure_directories()
    settings.ensure_directories()
    assert settings.upload_dir.is_dir()
    assert settings.generated_dir.is_dir()


def test_storage_directories_are_gitignored() -> None:
    """Uploads/generated output must never be committed to version control."""
    project_root = get_settings().upload_dir.parent
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert "uploads/*" in gitignore
    assert "generated/*" in gitignore
