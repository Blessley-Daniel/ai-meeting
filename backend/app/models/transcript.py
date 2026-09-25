"""ORM model holding the transcript produced for a meeting.

Kept in its own table rather than as columns on ``Meeting`` for two reasons:

1. A transcript is large (a one-hour meeting is tens of thousands of
   characters) and is only needed by the later stages. Listing meeting history
   should not drag that payload along.
2. It keeps ``meetings`` as a compact job/status table, and gives a clean place
   to record ASR metadata such as the model used and the detected language.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.session import utcnow


class Transcript(Base):
    """The speech-recognition output for one meeting."""

    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, index=True
    )

    # Full transcript as one string, for display and for feeding the NLP stage.
    text: Mapped[str] = mapped_column(Text)

    # Timestamped segments: [{index, start, end, text, confidence}, ...]
    # Stored as JSON so no extra table is needed for a read-mostly payload.
    segments: Mapped[list] = mapped_column(JSON, default=list)

    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    language_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Provenance, so results can be reproduced and reported honestly.
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    meeting = relationship("Meeting", back_populates="transcript")

    @property
    def segment_count(self) -> int:
        return len(self.segments or [])

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Transcript meeting={self.meeting_id} "
            f"segments={self.segment_count} words={self.word_count}>"
        )