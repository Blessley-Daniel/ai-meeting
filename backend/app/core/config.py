"""Central application configuration.

All settings are loaded from environment variables (or a `.env` file) using
pydantic-settings. Every variable is prefixed with ``MMA_`` to avoid clashing
with unrelated system variables such as ``PORT`` or ``DEBUG``.

Import the shared instance with::

    from app.core.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> core -> app -> backend -> <project root>
BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="MMA_",
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- Application ----------
    app_name: str = "AI-Based Meeting Minutes Generator"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---------- Storage locations ----------
    upload_dir: Path = PROJECT_ROOT / "uploads"
    generated_dir: Path = PROJECT_ROOT / "generated"
    models_dir: Path = PROJECT_ROOT / "models"
    dataset_dir: Path = PROJECT_ROOT / "datasets" / "processed"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'meeting_minutes.db'}"

    # ---------- Upload validation ----------
    max_upload_size_mb: int = 500
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"]
    )
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"]
    )

    # ---------- Speech recognition ----------
    whisper_model_size: str = "base"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None
    whisper_beam_size: int = 5

    # ---------- Meeting information extraction ----------
    extraction_backend: Literal["transformers", "rules", "auto"] = "auto"
    extraction_model_name: str = "google/flan-t5-base"
    extraction_adapter_path: str | None = None
    extraction_max_input_tokens: int = 1024
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------- Derived helpers ----------
    @property
    def allowed_extensions(self) -> set[str]:
        """Every extension accepted by the upload endpoint."""
        return set(self.allowed_video_extensions) | set(self.allowed_audio_extensions)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create every directory the application writes to."""
        for directory in (
            self.upload_dir,
            self.generated_dir,
            self.models_dir,
            self.dataset_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance."""
    return Settings()