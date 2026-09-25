"""Pydantic schemas for the meetings API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.meeting import MediaType, MeetingStatus
from app.schemas.extraction import NOT_SPECIFIED


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


# ---------------------------------------------------------------------------
# Information extraction
# ---------------------------------------------------------------------------


class ActionItemRead(BaseModel):
    """A single action item."""

    task: str
    responsible_person: str = NOT_SPECIFIED
    deadline: str = NOT_SPECIFIED


class ExtractionRead(BaseModel):
    """Structured information extracted from a meeting transcript."""

    title: str = NOT_SPECIFIED
    date: str = NOT_SPECIFIED
    time: str = NOT_SPECIFIED
    participants: list[str] = Field(default_factory=list)
    overview: str = NOT_SPECIFIED
    discussion_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItemRead] = Field(default_factory=list)
    conclusion: str = NOT_SPECIFIED


class ExtractionDiagnosticsRead(BaseModel):
    """How an extraction was produced, for transparency in the interface."""

    model_name: str = ""
    rules_used: list[str] = Field(default_factory=list)
    dropped_ungrounded: dict[str, list[str]] = Field(default_factory=dict)
    extraction_seconds: float | None = None


class ExtractionResultOut(BaseModel):
    """Outcome of running the analysis stage.

    As with the other stages, HTTP 200 even when the job failed; the reason is
    in ``error_message``.
    """

    meeting: MeetingRead
    succeeded: bool
    error_message: str | None = None
    extraction: ExtractionRead | None = None
    diagnostics: ExtractionDiagnosticsRead | None = None


# ---------------------------------------------------------------------------
# Minutes of Meeting
# ---------------------------------------------------------------------------


class MinutesRead(BaseModel):
    """Generated Minutes of Meeting."""

    meeting_id: int
    title: str = NOT_SPECIFIED
    extraction: ExtractionRead
    text: str = ""
    has_pdf: bool = False
    has_docx: bool = False
    generated_at: datetime | None = None


class MinutesResultOut(BaseModel):
    """Outcome of running the minutes-generation stage."""

    meeting: MeetingRead
    succeeded: bool
    error_message: str | None = None
    minutes: MinutesRead | None = None


class ProcessResultOut(BaseModel):
    """Outcome of running the whole pipeline in one request."""

    meeting: MeetingRead
    succeeded: bool
    error_message: str | None = None
    minutes: MinutesRead | None = None