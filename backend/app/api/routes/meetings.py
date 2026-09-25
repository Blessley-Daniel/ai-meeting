"""Meeting upload, retrieval, listing and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.meeting import Meeting, MeetingStatus
from app.schemas.meeting import (
    AudioExtractionResult,
    MeetingList,
    MeetingRead,
    TranscriptRead,
    TranscriptionResultOut,
)
from app.services import pipeline, storage
from app.utils.errors import (
    EmptyFileError,
    FFmpegNotFoundError,
    FileTooLargeError,
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


@router.post(
    "/{meeting_id}/extract-audio",
    response_model=AudioExtractionResult,
    summary="Extract a normalised audio track from the recording",
    responses={
        404: {"description": "Unknown meeting id"},
        409: {"description": "Recording has already advanced past this stage"},
        503: {"description": "ffmpeg is not installed"},
    },
)
async def extract_audio_endpoint(
    meeting_id: int,
    db: Session = Depends(get_db),
) -> AudioExtractionResult:
    """Run the audio-extraction stage.

    ffmpeg is synchronous and CPU-bound, so it is dispatched to a worker
    thread; running it inline would block the event loop for every other
    request.
    """
    meeting = _get_or_404(db, meeting_id)

    if meeting.status is not MeetingStatus.UPLOADED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Meeting {meeting_id} is in state {meeting.status.value!r}; "
            "audio extraction runs once, from the 'uploaded' state.",
        )

    try:
        meeting = await run_in_threadpool(pipeline.extract_audio_for_meeting, db, meeting)
    except FFmpegNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return AudioExtractionResult(
        meeting=MeetingRead.model_validate(meeting),
        succeeded=meeting.status is MeetingStatus.AUDIO_EXTRACTED,
        error_message=meeting.error_message,
    )


@router.post(
    "/{meeting_id}/transcribe",
    response_model=TranscriptionResultOut,
    summary="Transcribe the extracted audio with Whisper",
    responses={
        404: {"description": "Unknown meeting id"},
        409: {"description": "Audio has not been extracted yet"},
    },
)
async def transcribe_endpoint(
    meeting_id: int,
    db: Session = Depends(get_db),
) -> TranscriptionResultOut:
    """Run the speech-recognition stage.

    Whisper is synchronous and CPU-bound, so it runs in a worker thread to
    keep the event loop responsive.
    """
    meeting = _get_or_404(db, meeting_id)

    if meeting.status is not MeetingStatus.AUDIO_EXTRACTED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Meeting {meeting_id} is in state {meeting.status.value!r}; "
            "transcription requires the 'audio_extracted' state.",
        )

    meeting = await run_in_threadpool(pipeline.transcribe_meeting, db, meeting)

    transcript = TranscriptRead.model_validate(meeting.transcript) if meeting.transcript else None
    return TranscriptionResultOut(
        meeting=MeetingRead.model_validate(meeting),
        succeeded=meeting.status is MeetingStatus.TRANSCRIBED,
        error_message=meeting.error_message,
        transcript=transcript,
    )


@router.get(
    "/{meeting_id}/transcript",
    response_model=TranscriptRead,
    summary="Get the stored transcript",
    responses={404: {"description": "No transcript exists for this meeting"}},
)
def get_transcript(meeting_id: int, db: Session = Depends(get_db)) -> TranscriptRead:
    meeting = _get_or_404(db, meeting_id)

    if meeting.transcript is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No transcript has been produced for meeting {meeting_id}.",
        )
    return TranscriptRead.model_validate(meeting.transcript)


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