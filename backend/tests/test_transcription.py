"""Tests for speech recognition (Steps 5-6).

The unit tests use a *fake* model object so they run fast and deterministically;
they verify our logic around the model (error handling, segment assembly,
serialisation), not Whisper's accuracy. Accuracy is measured separately in the
evaluation step against real audio.

One integration test runs real Whisper on real speech; it is marked ``slow``
and skipped by default. Run it with::

    .venv/bin/python -m pytest -m slow tests/test_transcription.py
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import transcription
from app.utils.errors import EmptyTranscriptError, TranscriptionError


# --------------------------------------------------------------------------
# Fakes: stand in for faster-whisper's generator output
# --------------------------------------------------------------------------


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str, avg_logprob: float = -0.2):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeInfo:
    def __init__(self, language: str = "en", probability: float = 0.99, duration: float = 12.0):
        self.language = language
        self.language_probability = probability
        self.duration = duration


class _FakeModel:
    """Mimics WhisperModel.transcribe: returns (generator, info)."""

    def __init__(self, segments: list[_FakeSegment], info: _FakeInfo | None = None):
        self._segments = segments
        self._info = info or _FakeInfo()

    def transcribe(self, _path, **_kwargs):
        return iter(self._segments), self._info


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    """A small real WAV file so the existence check passes."""
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)
    return path


# --------------------------------------------------------------------------
# transcribe_audio logic
# --------------------------------------------------------------------------


def test_transcribe_assembles_text_and_segments(wav_file: Path, monkeypatch) -> None:
    fake = _FakeModel(
        [
            _FakeSegment(0.0, 4.0, " Good morning everyone."),
            _FakeSegment(4.0, 8.0, " Let's start the project sync."),
        ]
    )
    monkeypatch.setattr(transcription, "load_model", lambda *_: fake)

    result = transcription.transcribe_audio(wav_file)

    assert result.text == "Good morning everyone. Let's start the project sync."
    assert len(result.segments) == 2
    assert result.segments[0].index == 0
    assert result.segments[1].start == 4.0
    assert result.language == "en"
    assert result.model_size


def test_transcribe_skips_whitespace_only_segments(wav_file: Path, monkeypatch) -> None:
    """Whisper emits blank segments over silence; they must not create gaps."""
    fake = _FakeModel(
        [
            _FakeSegment(0.0, 2.0, "First"),
            _FakeSegment(2.0, 4.0, "   "),
            _FakeSegment(4.0, 6.0, "Second"),
        ]
    )
    monkeypatch.setattr(transcription, "load_model", lambda *_: fake)

    result = transcription.transcribe_audio(wav_file)

    assert result.text == "First Second"
    # Indices must be contiguous after dropping the blank.
    assert [s.index for s in result.segments] == [0, 1]


def test_transcribe_raises_on_empty_result(wav_file: Path, monkeypatch) -> None:
    """A silent recording yields no segments and must be a clear error."""
    monkeypatch.setattr(transcription, "load_model", lambda *_: _FakeModel([]))

    with pytest.raises(EmptyTranscriptError):
        transcription.transcribe_audio(wav_file)


def test_transcribe_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="not found"):
        transcription.transcribe_audio(tmp_path / "absent.wav")


def test_load_model_failure_becomes_domain_error(wav_file: Path, monkeypatch) -> None:
    class _Boom:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("ctranslate2 exploded")

    monkeypatch.setattr(transcription, "load_model", lambda *_: _Boom())

    with pytest.raises(TranscriptionError, match="Could not decode"):
        transcription.transcribe_audio(wav_file)


def test_realtime_factor_is_computed(wav_file: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel([_FakeSegment(0.0, 1.0, "Hi")], _FakeInfo(duration=10.0)),
    )

    result = transcription.transcribe_audio(wav_file)

    assert result.realtime_factor is not None and result.realtime_factor > 0


def test_progress_callback_receives_each_segment(wav_file: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel([_FakeSegment(i, i + 1, f"s{i}") for i in range(3)]),
    )

    seen: list[int] = []
    transcription.transcribe_audio(wav_file, progress=lambda count, _seg: seen.append(count))

    assert seen == [1, 2, 3]


def test_segment_round_trip_through_json() -> None:
    original = [
        transcription.TranscriptSegment(0, 0.0, 1.5, "hello", 0.9),
        transcription.TranscriptSegment(1, 1.5, 3.0, "world", None),
    ]

    restored = transcription.dicts_to_segments(transcription.segments_to_dicts(original))

    assert restored == original


def test_average_logprob_converts_to_probability() -> None:
    assert transcription._average_logprob(_FakeSegment(0, 1, "x", 0.0)) == 0.0

    value = transcription._average_logprob(_FakeSegment(0, 1, "x", -1.0))
    assert value == pytest.approx(0.3679, abs=1e-3)

    assert transcription._average_logprob(type("S", (), {})()) is None


# --------------------------------------------------------------------------
# Endpoint behaviour (uses the fake model via monkeypatch)
# --------------------------------------------------------------------------


def _upload_and_extract(client: TestClient, sample_mp4: Path) -> int:
    meeting = client.post(
        "/api/meetings",
        files={"file": (sample_mp4.name, sample_mp4.read_bytes(), "video/mp4")},
    ).json()
    client.post(f"/api/meetings/{meeting['id']}/extract-audio")
    return meeting["id"]


def test_transcribe_endpoint_rejects_wrong_state(client: TestClient, sample_mp4: Path) -> None:
    """Transcribing before extraction must be a 409, not a crash."""
    meeting = client.post(
        "/api/meetings",
        files={"file": (sample_mp4.name, sample_mp4.read_bytes(), "video/mp4")},
    ).json()

    response = client.post(f"/api/meetings/{meeting['id']}/transcribe")

    assert response.status_code == 409


def test_transcribe_endpoint_404_for_unknown_meeting(client: TestClient) -> None:
    assert client.post("/api/meetings/999999/transcribe").status_code == 404


def test_forcing_empty_transcript_marks_job_failed(
    client: TestClient, sample_mp4: Path, monkeypatch
) -> None:
    """An empty transcript is recorded as a failure with a readable reason."""
    meeting_id = _upload_and_extract(client, sample_mp4)
    monkeypatch.setattr(
        transcription,
        "transcribe_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            EmptyTranscriptError("meeting.wav")
        ),
    )

    body = client.post(f"/api/meetings/{meeting_id}/transcribe").json()

    assert body["succeeded"] is False
    assert body["meeting"]["status"] == "failed"
    assert "no speech was recognised" in body["error_message"].lower()
    assert body["transcript"] is None


def test_transcript_endpoint_404_before_transcription(
    client: TestClient, sample_mp4: Path
) -> None:
    meeting_id = _upload_and_extract(client, sample_mp4)
    assert client.get(f"/api/meetings/{meeting_id}/transcript").status_code == 404


def test_transcript_endpoint_returns_stored_transcript(
    client: TestClient, sample_mp4: Path, monkeypatch
) -> None:
    """The success path: transcribe with a fake model, then read it back."""
    meeting_id = _upload_and_extract(client, sample_mp4)
    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel(
            [
                _FakeSegment(0.0, 3.0, " The team decided to use Python."),
                _FakeSegment(3.0, 6.0, " Rahul will update the docs."),
            ]
        ),
    )

    posted = client.post(f"/api/meetings/{meeting_id}/transcribe").json()
    assert posted["succeeded"] is True
    assert posted["meeting"]["status"] == "transcribed"

    fetched = client.get(f"/api/meetings/{meeting_id}/transcript").json()

    assert fetched["meeting_id"] == meeting_id
    assert len(fetched["segments"]) == 2
    assert fetched["segment_count"] == 2
    assert fetched["word_count"] == len(fetched["text"].split())
    assert fetched["model_name"].startswith("faster-whisper:")
    assert "Python" in fetched["text"]
    # Timestamps must survive the round trip through SQLite JSON.
    assert fetched["segments"][0]["start"] == 0.0
    assert fetched["segments"][1]["end"] == 6.0


# --------------------------------------------------------------------------
# Row lifecycle (regression)
# --------------------------------------------------------------------------


def _child_rows(meeting_id: int) -> tuple[Transcript | None, MinutesDocument | None]:
    """Fetch the transcript and minutes for a meeting, or None if absent."""
    from sqlalchemy import select

    from app.database.session import SessionLocal
    from app.models.minutes import MinutesDocument
    from app.models.transcript import Transcript

    with SessionLocal() as db:
        transcript = db.scalar(
            select(Transcript).where(Transcript.meeting_id == meeting_id)
        )
        minutes = db.scalar(
            select(MinutesDocument).where(MinutesDocument.meeting_id == meeting_id)
        )
    return transcript, minutes


def test_deleting_meeting_removes_child_rows(
    client: TestClient, sample_mp4: Path, monkeypatch
) -> None:
    """Deleting a meeting must take its transcript and minutes with it.

    Regression: SQLite leaves foreign-key enforcement off unless
    ``PRAGMA foreign_keys=ON`` is set per connection, so ``ON DELETE CASCADE``
    never fired and every delete left orphaned child rows behind.
    """
    from app.database.session import SessionLocal
    from app.models.minutes import MinutesDocument

    meeting_id = _upload_and_extract(client, sample_mp4)
    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel([_FakeSegment(0.0, 3.0, " Some text.")]),
    )
    assert client.post(f"/api/meetings/{meeting_id}/transcribe").json()["succeeded"]

    with SessionLocal() as db:
        db.add(MinutesDocument(meeting_id=meeting_id, extraction={"title": "T"}))
        db.commit()

    transcript, minutes = _child_rows(meeting_id)
    assert transcript is not None
    assert minutes is not None

    assert client.delete(f"/api/meetings/{meeting_id}").status_code == 204

    transcript, minutes = _child_rows(meeting_id)
    assert transcript is None
    assert minutes is None


def test_retranscribing_existing_meeting_succeeds(
    client: TestClient, sample_mp4: Path, monkeypatch
) -> None:
    """Re-running the transcription stage must replace, not corrupt, the row.

    Regression: the replacement path deleted the old Transcript row while
    meeting.transcript still referenced it, so the status write cascaded to a
    deleted instance and sqlalchemy raised InvalidRequestError.
    """
    from app.database.session import SessionLocal
    from app.models.meeting import Meeting
    from app.services import pipeline

    meeting_id = _upload_and_extract(client, sample_mp4)
    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel([_FakeSegment(0.0, 3.0, " First pass.")]),
    )
    with SessionLocal() as db:
        meeting = pipeline.transcribe_meeting(db, db.get(Meeting, meeting_id))
        assert meeting.status.value == "transcribed"

    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel(
            [
                _FakeSegment(0.0, 3.0, " Second pass, revised."),
                _FakeSegment(3.0, 5.0, " Extra detail."),
            ]
        ),
    )
    with SessionLocal() as db:
        meeting = pipeline.transcribe_meeting(db, db.get(Meeting, meeting_id))
        assert meeting.status.value == "transcribed"

    transcript, _ = _child_rows(meeting_id)
    assert len(transcript.segments) == 2
    assert "Second pass" in transcript.text
    # The old text must be replaced, not appended to or left behind.
    assert "First pass" not in transcript.text


def test_retranscribing_invalidates_previous_minutes(
    client: TestClient, sample_mp4: Path, monkeypatch
) -> None:
    """Minutes describe a transcript, so a new transcript makes them stale.

    The stale minutes are cleared in place rather than deleted, because the
    relationship holds a live reference to them while the status is written.
    """
    from app.database.session import SessionLocal
    from app.models.meeting import Meeting
    from app.models.minutes import MinutesDocument
    from app.services import pipeline

    meeting_id = _upload_and_extract(client, sample_mp4)
    with SessionLocal() as db:
        db.add(
            MinutesDocument(
                meeting_id=meeting_id,
                extraction={"title": "Old"},
                text="OLD MINUTES",
            )
        )
        db.commit()

    monkeypatch.setattr(
        transcription,
        "load_model",
        lambda *_: _FakeModel([_FakeSegment(0.0, 3.0, " Fresh transcript.")]),
    )
    with SessionLocal() as db:
        pipeline.transcribe_meeting(db, db.get(Meeting, meeting_id))

    _, minutes = _child_rows(meeting_id)
    assert minutes is not None
    assert minutes.extraction == {}
    assert minutes.text == ""


# --------------------------------------------------------------------------
# Real Whisper integration test (opt-in)
# --------------------------------------------------------------------------


def _make_speech_wav(tmp_path: Path) -> Path:
    """Synthesise real speech with espeak-ng, if available."""
    import shutil

    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not installed; cannot synthesise speech")

    raw = tmp_path / "raw.wav"
    out = tmp_path / "speech.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(raw),
         "The team decided to use Python for the backend."],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )
    return out


@pytest.mark.slow
def test_real_whisper_transcribes_synthesised_speech(tmp_path: Path) -> None:
    """End-to-end check that real Whisper weights load and decode audio."""
    pytest.importorskip("faster_whisper")
    audio_path = _make_speech_wav(tmp_path)

    result = transcription.transcribe_audio(audio_path)

    assert result.segments, "Whisper returned no segments"
    assert result.language == "en"
    lowered = result.text.lower()
    # Real ASR on synthetic speech: assert on robust keywords only.
    assert "python" in lowered or "backend" in lowered
    assert result.realtime_factor is not None