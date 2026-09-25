"""Pydantic schemas for the meetings API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.meeting import MediaType, MeetingStatus


class MeetingRead(BaseModel):
    """A meeting job as returned to the client.

    Deliberately excludes ``stored_path``: the client has no business knowing
    server-side absolute paths.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    media_type: MediaType
    file_size_bytes: int
    content_type: str | None
    status: MeetingStatus
    error_message: str | None
    duration_seconds: float | None
    has_audio: bool
    created_at: datetime
    updated_at: datetime


class MeetingList(BaseModel):
    """Paged collection of meeting jobs."""

    total: int
    limit: int
    offset: int
    items: list[MeetingRead]


class AudioExtractionResult(BaseModel):
    """Outcome of running the audio-extraction stage for one meeting.

    Returned with HTTP 200 even when ``status`` is ``failed``: the request
    itself succeeded, it is the *media* that could not be processed. The
    reason is carried in ``error_message``.
    """

    meeting: MeetingRead
    succeeded: bool
    error_message: str | None = None


class ErrorResponse(BaseModel):
    """Consistent error body for all 4xx/5xx responses."""

    detail: str
    code: str


class TranscriptSegmentRead(BaseModel):
    """One timestamped span of recognised speech."""

    index: int
    start: float
    end: float
    text: str
    confidence: float | None = None


class TranscriptRead(BaseModel):
    """The speech-recognition output for a meeting."""

    model_config = ConfigDict(from_attributes=True)

    meeting_id: int
    text: str
    segments: list[TranscriptSegmentRead]
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    model_name: str | None
    processing_seconds: float | None
    segment_count: int
    word_count: int
    created_at: datetime


class TranscriptionResultOut(BaseModel):
    """Outcome of running the transcription stage.

    Mirrors :class:`AudioExtractionResult`: HTTP 200 even when the *job*
    failed, with the reason in ``error_message``.
    """

    meeting: MeetingRead
    succeeded: bool
    error_message: str | None = None
    transcript: TranscriptRead | None = None