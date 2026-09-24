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


class ErrorResponse(BaseModel):
    """Consistent error body for all 4xx/5xx responses."""

    detail: str
    code: str