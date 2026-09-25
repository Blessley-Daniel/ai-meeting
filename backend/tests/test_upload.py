"""Tests for the meeting upload module (Step 2).

These exercise the real code path: multipart upload -> validation ->
streaming to disk -> SQLite persistence. Only the uploaded bytes and the
storage directory are controlled; no mocks are used.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from tests.conftest import requires_ffmpeg


def _upload(client: TestClient, name: str, data: bytes, content_type: str):
    return client.post(
        "/api/meetings",
        files={"file": (name, data, content_type)},
    )


def test_upload_audio_creates_job(client: TestClient, sample_wav: Path) -> None:
    response = _upload(client, "standup.wav", sample_wav.read_bytes(), "audio/wav")

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "standup.wav"
    assert body["media_type"] == "audio"
    assert body["status"] == "uploaded"
    assert body["file_size_bytes"] == sample_wav.stat().st_size
    assert body["has_audio"] is False
    # Internal absolute path must not leak to the client.
    assert "stored_path" not in body


@requires_ffmpeg
def test_upload_video_is_classified_as_video(client: TestClient, sample_mp4: Path) -> None:
    response = _upload(client, "meet.mp4", sample_mp4.read_bytes(), "video/mp4")

    assert response.status_code == 201
    assert response.json()["media_type"] == "video"


def test_uploaded_file_is_written_to_disk(client: TestClient, sample_wav: Path) -> None:
    client.post("/api/meetings", files={"file": ("a.wav", sample_wav.read_bytes(), "audio/wav")})

    stored = list(get_settings().upload_dir.glob("*.wav"))
    assert len(stored) == 1
    assert stored[0].read_bytes() == sample_wav.read_bytes()


def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = _upload(client, "notes.txt", b"hello", "text/plain")

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert list(get_settings().upload_dir.iterdir()) == []


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = _upload(client, "silent.wav", b"", "audio/wav")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_oversized_file_is_rejected_and_cleaned_up(client: TestClient) -> None:
    settings = get_settings()
    # Limit is 1 MiB for tests; send 1.5 MiB.
    oversized = b"\x00" * int(settings.max_upload_size_bytes * 1.5)

    response = _upload(client, "huge.wav", oversized, "audio/wav")

    assert response.status_code == 413
    # The partial file must not be left behind.
    assert list(settings.upload_dir.iterdir()) == []


def test_malicious_filename_cannot_escape_upload_dir(client: TestClient) -> None:
    """A traversal-style filename must not influence the stored path."""
    uploads = get_settings().upload_dir
    response = _upload(client, "../../../../tmp/evil.wav", b"data", "audio/wav")

    assert response.status_code == 201
    stored = list(uploads.glob("*.wav"))
    assert len(stored) == 1
    # The UUID-named file lives directly inside uploads/, not three levels up.
    assert stored[0].parent == uploads
    assert not Path("/tmp/evil.wav").exists()


def test_list_meetings_returns_newest_first(client: TestClient, sample_wav: Path) -> None:
    for name in ("first.wav", "second.wav"):
        client.post("/api/meetings", files={"file": (name, b"x" * 32, "audio/wav")})

    response = client.get("/api/meetings")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 2
    assert [item["original_filename"] for item in body["items"]] == [
        "second.wav",
        "first.wav",
    ]


def test_get_meeting_returns_404_for_unknown_id(client: TestClient) -> None:
    assert client.get("/api/meetings/9999").status_code == 404


def test_delete_meeting_removes_row_and_file(client: TestClient) -> None:
    created = client.post(
        "/api/meetings", files={"file": ("bye.wav", b"x" * 64, "audio/wav")}
    ).json()

    stored_file = next(get_settings().upload_dir.glob("*.wav"))

    assert client.delete(f"/api/meetings/{created['id']}").status_code == 204
    assert client.get(f"/api/meetings/{created['id']}").status_code == 404
    assert not stored_file.exists()
    assert client.delete(f"/api/meetings/{created['id']}").status_code == 404