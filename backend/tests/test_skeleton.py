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


def test_health_reports_upload_limit(client, app_env) -> None:
    """The app's own upload ceiling must be discoverable.

    A Codespaces forwarded port imposes a much smaller body cap in front of the
    app, so the two numbers need to be distinguishable when diagnosing a 413.
    """
    body = client.get("/api/health").json()
    assert body["config"]["max_upload_size_mb"] == get_settings().max_upload_size_mb


def test_frontend_assets_are_served(client) -> None:
    """index.html plus the assets it references must all resolve.

    A missing asset is what produced the stray browser-console 404: the page
    referenced nothing, so the browser requested /favicon.ico and the static
    mount correctly answered 404 for a file that did not exist.
    """
    index = client.get("/")
    assert index.status_code == 200

    # Every local href/src in the page must be fetchable.
    body = index.text
    for asset in ("style.css", "script.js", "favicon.svg"):
        assert asset in body, f"index.html should reference {asset}"
        response = client.get(f"/{asset}")
        assert response.status_code == 200, f"{asset} returned {response.status_code}"


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
