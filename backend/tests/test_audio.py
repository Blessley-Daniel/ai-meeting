"""Tests for audio extraction (Step 4).

These run real ffmpeg/ffprobe subprocesses on real generated media files. No
mocks: a passing test means the pipeline genuinely produced a 16 kHz mono WAV.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services import audio
from app.utils.errors import MediaProcessingError, NoAudioStreamError
from tests.conftest import requires_ffmpeg


def _upload(client: TestClient, path: Path, content_type: str) -> dict:
    response = client.post(
        "/api/meetings",
        files={"file": (path.name, path.read_bytes(), content_type)},
    )
    assert response.status_code == 201
    return response.json()


def _probe_stream(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)["streams"][0]


# --------------------------------------------------------------------------
# probe_media
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_probe_video_reports_both_streams(sample_mp4: Path, app_env) -> None:
    info = audio.probe_media(sample_mp4)

    assert info.has_video is True
    assert info.has_audio is True
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"


@requires_ffmpeg
def test_probe_video_without_audio_detects_missing_track(silent_mp4: Path, app_env) -> None:
    info = audio.probe_media(silent_mp4)

    assert info.has_video is True
    assert info.has_audio is False
    assert info.audio_codec is None


def test_probe_rejects_non_media_file(tmp_path: Path, app_env) -> None:
    fake = tmp_path / "not_media.wav"
    fake.write_bytes(b"this is plain text pretending to be audio")

    with pytest.raises(MediaProcessingError):
        audio.probe_media(fake)


def test_probe_rejects_missing_file(tmp_path: Path, app_env) -> None:
    with pytest.raises(MediaProcessingError):
        audio.probe_media(tmp_path / "does_not_exist.mp4")


# --------------------------------------------------------------------------
# extract_audio
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_extract_produces_16khz_mono_wav(sample_mp4: Path, tmp_path: Path, app_env) -> None:
    destination = tmp_path / "out.wav"

    info = audio.extract_audio(sample_mp4, destination)

    assert destination.is_file()
    assert info.duration_seconds == pytest.approx(2.0, abs=0.3)

    with wave.open(str(destination), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2  # 16-bit PCM


@requires_ffmpeg
def test_extract_handles_audio_only_input(m4a_audio: Path, tmp_path: Path, app_env) -> None:
    destination = tmp_path / "out.wav"

    audio.extract_audio(m4a_audio, destination)

    stream = _probe_stream(destination)
    assert stream["codec_name"] == "pcm_s16le"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1


@requires_ffmpeg
def test_extract_raises_on_video_without_audio(silent_mp4: Path, tmp_path: Path, app_env) -> None:
    with pytest.raises(NoAudioStreamError):
        audio.extract_audio(silent_mp4, tmp_path / "out.wav")

    # A failed extraction must not leave a file behind.
    assert not (tmp_path / "out.wav").exists()


# --------------------------------------------------------------------------
# endpoint integration
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_extract_endpoint_advances_status(client: TestClient, sample_mp4: Path) -> None:
    meeting = _upload(client, sample_mp4, "video/mp4")

    response = client.post(f"/api/meetings/{meeting['id']}/extract-audio")

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] is True
    assert body["meeting"]["status"] == "audio_extracted"
    assert body["meeting"]["has_audio"] is True
    assert body["meeting"]["duration_seconds"] == pytest.approx(2.0, abs=0.3)


@requires_ffmpeg
def test_extracted_audio_lands_in_audio_dir(client: TestClient, sample_mp4: Path) -> None:
    meeting = _upload(client, sample_mp4, "video/mp4")
    client.post(f"/api/meetings/{meeting['id']}/extract-audio")

    produced = list(get_settings().audio_dir.glob("*.wav"))
    assert len(produced) == 1
    assert produced[0].name == f"meeting_{meeting['id']}.wav"


@requires_ffmpeg
def test_extract_endpoint_fails_gracefully_without_audio(
    client: TestClient, silent_mp4: Path
) -> None:
    """The request succeeds; the *job* is marked failed with a clear reason."""
    meeting = _upload(client, silent_mp4, "video/mp4")

    response = client.post(f"/api/meetings/{meeting['id']}/extract-audio")

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] is False
    assert body["meeting"]["status"] == "failed"
    assert "no audio stream" in body["error_message"].lower()
    # A failed job must not claim to have audio.
    assert body["meeting"]["has_audio"] is False


@requires_ffmpeg
def test_extract_endpoint_is_not_repeatable(client: TestClient, sample_mp4: Path) -> None:
    meeting = _upload(client, sample_mp4, "video/mp4")
    client.post(f"/api/meetings/{meeting['id']}/extract-audio")

    second = client.post(f"/api/meetings/{meeting['id']}/extract-audio")

    assert second.status_code == 409


def test_extract_endpoint_404_for_unknown_meeting(client: TestClient) -> None:
    assert client.post("/api/meetings/424242/extract-audio").status_code == 404