"""Meeting history: browsing previously processed recordings.

Separate from ``meetings.py`` because history is a different concern: it reads
across meetings to build a list for the sidebar, rather than acting on one
meeting. The list is deliberately lighter than the detail view - it returns
summaries, not transcripts or full extractions - so the interface stays
responsive as the number of stored meetings grows.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.meeting import Meeting, MeetingStatus
from app.models.minutes import MinutesDocument
from app.models.transcript import Transcript
from app.schemas.extraction import NOT_SPECIFIED
from app.schemas.meeting import MeetingRead

router = APIRouter(prefix="/history", tags=["history"])


class HistoryEntry(BaseModel):
    """One row in the history list."""

    id: int
    original_filename: str
    status: MeetingStatus
    title: str = NOT_SPECIFIED
    participants: list[str] = Field(default_factory=list)
    action_item_count: int = 0
    decision_count: int = 0
    duration_seconds: float | None = None
    word_count: int = 0
    has_pdf: bool = False
    has_docx: bool = False
    created_at: datetime
    error_message: str | None = None


class HistoryList(BaseModel):
    """A page of history plus the total number of stored meetings."""

    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class HistoryStats(BaseModel):
    """Aggregate figures, useful for the project evaluation report."""

    total_meetings: int
    completed: int
    failed: int
    total_audio_seconds: float
    total_action_items: int
    total_decisions: int


def _entry(meeting: Meeting) -> HistoryEntry:
    """Build a history row from a meeting and whatever it produced."""
    record = meeting.minutes
    extraction = (record.extraction or {}) if record else {}
    transcript = meeting.transcript

    return HistoryEntry(
        id=meeting.id,
        original_filename=meeting.original_filename,
        status=meeting.status,
        title=extraction.get("title", NOT_SPECIFIED),
        participants=extraction.get("participants", []) or [],
        action_item_count=len(extraction.get("action_items", []) or []),
        decision_count=len(extraction.get("decisions", []) or []),
        duration_seconds=meeting.duration_seconds,
        word_count=len(transcript.text.split()) if transcript else 0,
        has_pdf=bool(record and record.pdf_path),
        has_docx=bool(record and record.docx_path),
        created_at=meeting.created_at,
        error_message=meeting.error_message,
    )


@router.get("", response_model=HistoryList, summary="List processed meetings")
def list_history(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: MeetingStatus | None = Query(None, alias="status"),
) -> HistoryList:
    """Return stored meetings, most recent first.

    Filtering and paging happen in SQL rather than in Python so the query cost
    does not grow with the number of stored meetings.
    """
    query = select(Meeting)
    count_query = select(func.count()).select_from(Meeting)

    if status_filter is not None:
        query = query.where(Meeting.status == status_filter)
        count_query = count_query.where(Meeting.status == status_filter)

    total = db.scalar(count_query) or 0
    meetings = (
        db.scalars(
            query.order_by(Meeting.created_at.desc(), Meeting.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .all()
    )

    return HistoryList(
        items=[_entry(m) for m in meetings],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/stats",
    response_model=HistoryStats,
    summary="Aggregate statistics over all processed meetings",
)
def history_stats(db: Session = Depends(get_db)) -> HistoryStats:
    """Aggregate counts used when writing up the project results."""
    meetings = db.scalars(select(Meeting)).all()

    completed = sum(1 for m in meetings if m.status is MeetingStatus.COMPLETED)
    failed = sum(1 for m in meetings if m.status is MeetingStatus.FAILED)
    audio_seconds = sum(m.duration_seconds or 0.0 for m in meetings)

    documents = db.scalars(select(MinutesDocument)).all()
    action_items = 0
    decisions = 0
    for document in documents:
        extraction = document.extraction or {}
        action_items += len(extraction.get("action_items", []) or [])
        decisions += len(extraction.get("decisions", []) or [])

    return HistoryStats(
        total_meetings=len(meetings),
        completed=completed,
        failed=failed,
        total_audio_seconds=audio_seconds,
        total_action_items=action_items,
        total_decisions=decisions,
    )


@router.get(
    "/{meeting_id}/full",
    response_model=dict,
    summary="Full record for one meeting: transcript, extraction and minutes",
)
def get_full_record(meeting_id: int, db: Session = Depends(get_db)) -> dict:
    """Return everything stored about one meeting in a single response.

    Used by the interface when a history entry is opened, so the page does not
    need four separate requests.
    """
    meeting = db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No meeting with id {meeting_id}."
        )

    transcript = meeting.transcript
    record = meeting.minutes

    return {
        "meeting": MeetingRead.model_validate(meeting).model_dump(mode="json"),
        "transcript": (
            {
                "text": transcript.text,
                "segments": transcript.segments,
                "language": transcript.language,
                "model_name": transcript.model_name,
                "duration_seconds": transcript.duration_seconds,
                "processing_seconds": transcript.processing_seconds,
                "segment_count": len(transcript.segments or []),
                "word_count": len(transcript.text.split()),
            }
            if transcript
            else None
        ),
        "extraction": record.extraction if record else None,
        "minutes": (
            {
                "text": record.text,
                "has_pdf": bool(record.pdf_path),
                "has_docx": bool(record.docx_path),
                "extraction_model": record.extraction_model,
                "rules_used": record.rules_used,
                "dropped_ungrounded": record.dropped_ungrounded,
                "extraction_seconds": record.extraction_seconds,
            }
            if record
            else None
        ),
    }