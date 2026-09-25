"""ORM model for an uploaded meeting and its processing lifecycle."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum as SAEnum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.session import utcnow


class MeetingStatus(str, enum.Enum):
    """Lifecycle states of a meeting job, surfaced to the UI as progress."""

    UPLOADED = "uploaded"
    EXTRACTING_AUDIO = "extracting_audio"
    AUDIO_EXTRACTED = "audio_extracted"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    ANALYSING = "analysing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class MediaType(str, enum.Enum):
    """Whether the uploaded recording contains video or audio only."""

    VIDEO = "video"
    AUDIO = "audio"


def _enum_column(enum_cls: type[enum.Enum]) -> SAEnum:
    """Store enum *values* (e.g. "uploaded") instead of member names."""
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Meeting(Base):
    """A single uploaded recording together with its processing state."""

    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Name as supplied by the client, kept for display only. It is never used
    # to build a filesystem path (that would allow path traversal).
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    media_type: Mapped[MediaType] = mapped_column(_enum_column(MediaType))
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[MeetingStatus] = mapped_column(
        _enum_column(MeetingStatus),
        default=MeetingStatus.UPLOADED,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Filled in by the audio extraction step.
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    # One-to-one with the transcript produced by the speech-recognition stage.
    # cascade="all, delete-orphan" means deleting a meeting removes its
    # transcript rather than leaving an orphan row.
    transcript = relationship(
        "Transcript",
        back_populates="meeting",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def has_audio(self) -> bool:
        """True once the audio-extraction step has produced a track."""
        return bool(self.audio_path)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Meeting id={self.id} file={self.original_filename!r} "
            f"status={self.status.value!r}>"
        )