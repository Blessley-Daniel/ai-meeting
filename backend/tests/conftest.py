"""Shared pytest fixtures.

The application reads its storage locations from a cached ``Settings`` object.
To keep tests hermetic we clear that cache and point every path at a temporary
directory *before* importing application modules that capture settings at
import time (notably the SQLAlchemy engine).
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ffmpeg is a *system* dependency and does not live inside the virtual
# environment, so it can disappear when a container is reset. Tests that need
# it skip with a clear message instead of failing with a confusing error.
_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(
    not _HAS_FFMPEG,
    reason="ffmpeg/ffprobe not on PATH - install with: sudo apt-get install -y ffmpeg",
)


@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the app against a throwaway database and upload directory."""
    monkeypatch.setenv("MMA_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MMA_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MMA_GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("MMA_MAX_UPLOAD_SIZE_MB", "1")

    from app.core.config import get_settings

    get_settings.cache_clear()
    if "app.main" in sys.modules:
        importlib.reload(sys.modules["app.main"])
    if "app.database.session" in sys.modules:
        importlib.reload(sys.modules["app.database.session"])

    yield

    get_settings.cache_clear()


@pytest.fixture()
def client(app_env: None) -> Iterator["TestClient"]:  # noqa: F821
    """A ``TestClient`` bound to the freshly configured application."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Path:
    """A tiny but genuinely valid WAV file (1 s of silence at 16 kHz)."""
    import wave

    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return path


@pytest.fixture()
def sample_mp4(tmp_path: Path) -> Path:
    """A 2 s MP4 containing one video and one audio stream, built with ffmpeg."""
    import subprocess

    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture()
def silent_mp4(tmp_path: Path) -> Path:
    """A 2 s video-only MP4: no audio stream at all.

    Used to prove the pipeline reports a clear error instead of crashing when
    someone uploads a screen recording made without a microphone.
    """
    import subprocess

    path = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture()
def m4a_audio(tmp_path: Path) -> Path:
    """A 2 s audio-only file in a compressed container (AAC in M4A)."""
    import subprocess

    path = tmp_path / "voice.m4a"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:a", "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path