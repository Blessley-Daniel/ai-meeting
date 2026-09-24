"""Meeting upload, retrieval, listing and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.meeting import Meeting
from app.schemas.meeting import MeetingList, MeetingRead
from app.services import storage
from app.utils.errors import (
    EmptyFileError,
    FileTooLargeError,
    MediaNotFoundError,
    UnsupportedFileTypeError,
)

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post(
    "",
    response_model=MeetingRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a meeting recording",
    responses={
        400: {"description": "Empty file or unsupported file type"},
        413: {"description": "File exceeds the configured size limit"},
    },
)
async def upload_meeting(
    file: UploadFile = File(..., description="Video or audio recording"),
    db: Session = Depends(get_db),
) -> Meeting:
    """Persist a recording and create its processing job."""
    try:
        stored = await storage.save_upload(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except EmptyFileError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc

    meeting = Meeting(
        original_filename=stored.original_filename,
        stored_path=str(stored.stored_path),
        file_size_bytes=stored.size_bytes,
        media_type=stored.media_type,
        content_type=file.content_type,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


@router.get("", response_model=MeetingList, summary="List uploaded meetings")
def list_meetings(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> MeetingList:
    """Return meeting jobs newest first."""
    total = db.scalar(select(func.count()).select_from(Meeting)) or 0
    rows = db.scalars(
        select(Meeting).order_by(Meeting.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return MeetingList(
        total=total,
        limit=limit,
        offset=offset,
        items=[MeetingRead.model_validate(row) for row in rows],
    )


def _get_or_404(db: Session, meeting_id: int) -> Meeting:
    meeting = db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No meeting found for id {meeting_id}"
        )
    return meeting


@router.get("/{meeting_id}", response_model=MeetingRead, summary="Get one meeting")
def get_meeting(meeting_id: int, db: Session = Depends(get_db)) -> Meeting:
    return _get_or_404(db, meeting_id)


@router.delete(
    "/{meeting_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a meeting and its stored recording",
)
def delete_meeting(meeting_id: int, db: Session = Depends(get_db)) -> None:
    """Remove the database row and the file on disk."""
    meeting = _get_or_404(db, meeting_id)

    for path_str in (meeting.stored_path, meeting.audio_path):
        if path_str:
            Path(path_str).unlink(missing_ok=True)

    db.delete(meeting)
    db.commit()