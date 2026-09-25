"""ORM model for the Minutes of Meeting produced from a meeting.

Stores the structured extraction (as JSON), the rendered text, and the paths of
any exported documents. The structured JSON is the source of truth: the PDF and
DOCX are derived artefacts and can be regenerated from it at any time, which is
why the export files are treated as a cache rather than the record.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.database.session import utcnow


class MinutesDocument(Base):
    """Generated minutes for one meeting."""

    __tablename__ = "minutes_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, index=True
    )

    # The structured extraction - the authoritative record.
    extraction: Mapped[dict] = mapped_column(JSON, default=dict)

    # Rendered plain-text minutes, stored so history can be browsed without
    # re-running any model.
    text: Mapped[str] = mapped_column(Text, default="")

    # Populated lazily, on first export, rather than during generation: most
    # users view the minutes on screen and a PDF is only worth building if asked
    # for.
    pdf_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    docx_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Diagnostics describing how the extraction was produced. Kept because the
    # project is assessed partly on how well its accuracy can be explained.
    extraction_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rules_used: Mapped[list] = mapped_column(JSON, default=list)
    dropped_ungrounded: Mapped[dict] = mapped_column(JSON, default=dict)
    extraction_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    meeting = relationship("Meeting", back_populates="minutes")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MinutesDocument meeting={self.meeting_id}>"