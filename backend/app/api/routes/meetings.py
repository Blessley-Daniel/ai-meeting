"""Meeting upload, retrieval, listing and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.meeting import Meeting, MeetingStatus
from app.schemas.extraction import MeetingExtraction
from app.schemas.meeting import (
    AudioExtractionResult,
    ExtractionDiagnosticsRead,
    ExtractionRead,
    ExtractionResultOut,
    MeetingList,
    MeetingRead,
    MinutesRead,
    MinutesResultOut,
    ProcessResultOut,
    TranscriptRead,
    TranscriptionResultOut,
)
from app.services import exporters, pipeline, storage
from app.services import mom as mom_service
from app.utils.errors import (
    DocumentExportError,
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
    """Remove the database row and the files it produced on disk."""
    meeting = _get_or_404(db, meeting_id)

    paths = [meeting.stored_path, meeting.audio_path]
    if meeting.minutes is not None:
        paths += [meeting.minutes.pdf_path, meeting.minutes.docx_path]

    for path_str in paths:
        if path_str:
            Path(path_str).unlink(missing_ok=True)

    db.delete(meeting)
    db.commit()


# ---------------------------------------------------------------------------
# Information extraction and minutes
# ---------------------------------------------------------------------------


def _minutes_read(meeting: Meeting) -> MinutesRead | None:
    """Build the API representation of generated minutes, if any."""
    record = meeting.minutes
    if record is None or not record.extraction:
        return None
    return MinutesRead(
        meeting_id=meeting.id,
        title=record.extraction.get("title", "Not specified"),
        extraction=ExtractionRead.model_validate(record.extraction),
        text=record.text,
        has_pdf=bool(record.pdf_path),
        has_docx=bool(record.docx_path),
        generated_at=record.created_at,
    )


@router.post(
    "/{meeting_id}/analyse",
    response_model=ExtractionResultOut,
    summary="Extract structured information from the transcript with AI",
    responses={
        404: {"description": "Unknown meeting id"},
        409: {"description": "No transcript is available yet"},
    },
)
async def analyse_endpoint(
    meeting_id: int,
    db: Session = Depends(get_db),
) -> ExtractionResultOut:
    """Run the AI information-extraction stage.

    The language model is synchronous and CPU-bound, so it runs in a worker
    thread rather than blocking the event loop.
    """
    meeting = _get_or_404(db, meeting_id)

    if meeting.transcript is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Meeting {meeting_id} has no transcript yet; run transcription first.",
        )

    meeting = await run_in_threadpool(pipeline.analyse_meeting, db, meeting)

    succeeded = meeting.status is MeetingStatus.ANALYSED
    record = meeting.minutes
    diagnostics = None
    if record is not None:
        diagnostics = ExtractionDiagnosticsRead.model_validate(
            {
                "model_name": record.extraction_model or "",
                "rules_used": record.rules_used or [],
                "dropped_ungrounded": record.dropped_ungrounded or {},
                "extraction_seconds": record.extraction_seconds,
            }
        )

    return ExtractionResultOut(
        meeting=MeetingRead.model_validate(meeting),
        succeeded=succeeded,
        error_message=meeting.error_message,
        extraction=(
            ExtractionRead.model_validate(record.extraction)
            if succeeded and record and record.extraction
            else None
        ),
        diagnostics=diagnostics,
    )


@router.post(
    "/{meeting_id}/generate",
    response_model=MinutesResultOut,
    summary="Generate Minutes of Meeting from the extracted information",
    responses={
        404: {"description": "Unknown meeting id"},
        409: {"description": "The meeting has not been analysed yet"},
    },
)
async def generate_endpoint(
    meeting_id: int,
    db: Session = Depends(get_db),
) -> MinutesResultOut:
    """Render the Minutes of Meeting document."""
    meeting = _get_or_404(db, meeting_id)

    if meeting.minutes is None or not meeting.minutes.extraction:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Meeting {meeting_id} has not been analysed yet; run the analysis step first.",
        )

    meeting = await run_in_threadpool(pipeline.generate_minutes, db, meeting)

    return MinutesResultOut(
        meeting=MeetingRead.model_validate(meeting),
        succeeded=meeting.status is MeetingStatus.COMPLETED,
        error_message=meeting.error_message,
        minutes=_minutes_read(meeting),
    )


@router.post(
    "/{meeting_id}/process",
    response_model=ProcessResultOut,
    summary="Run the full pipeline: audio, transcription, analysis, minutes",
    responses={404: {"description": "Unknown meeting id"}},
)
async def process_endpoint(
    meeting_id: int,
    db: Session = Depends(get_db),
) -> ProcessResultOut:
    """Run every stage in sequence and return the finished minutes.

    This is the endpoint the web interface uses. Individual stage endpoints
    remain available for debugging and for demonstrating one module at a time.
    """
    meeting = _get_or_404(db, meeting_id)

    meeting = await run_in_threadpool(pipeline.process_meeting, db, meeting)

    return ProcessResultOut(
        meeting=MeetingRead.model_validate(meeting),
        succeeded=meeting.status is MeetingStatus.COMPLETED,
        error_message=meeting.error_message,
        minutes=_minutes_read(meeting),
    )


@router.get(
    "/{meeting_id}/minutes",
    response_model=MinutesRead,
    summary="Get the generated Minutes of Meeting",
    responses={404: {"description": "No minutes exist for this meeting"}},
)
def get_minutes(meeting_id: int, db: Session = Depends(get_db)) -> MinutesRead:
    meeting = _get_or_404(db, meeting_id)
    minutes = _minutes_read(meeting)
    if minutes is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No minutes have been generated for meeting {meeting_id}.",
        )
    return minutes


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

_EXPORT_TYPES = {"pdf", "docx"}


def _build_export(meeting: Meeting, fmt: str, db: Session) -> Path:
    """Render (or reuse) an exported report and return its path.

    Exports are cached on disk: the file is only rebuilt when it is missing or
    older than the extraction it describes. Since re-running the analysis stage
    clears the stored paths, a stale report can never be served.
    """
    settings = get_settings()
    record = meeting.minutes
    if record is None or not record.extraction:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Meeting {meeting_id_message(meeting)} has no minutes to export.",
        )

    existing = record.pdf_path if fmt == "pdf" else record.docx_path
    if existing and Path(existing).exists():
        return Path(existing)

    extraction_result = MeetingExtraction.model_validate(record.extraction)
    document = mom_service.build_mom(
        extraction_result,
        source_filename=meeting.original_filename,
        asr_model=meeting.transcript.model_name if meeting.transcript else "",
        extraction_model=record.extraction_model or "",
        fallback_title=mom_service.title_from_filename(meeting.original_filename),
    )

    stem = exporters.safe_stem(document.title, fallback=f"meeting_{meeting.id}_minutes")
    destination = settings.reports_path / f"{stem}_{meeting.id}.{fmt}"

    try:
        if fmt == "pdf":
            exporters.export_pdf(document, destination)
        else:
            exporters.export_docx(document, destination)
    except DocumentExportError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    if fmt == "pdf":
        record.pdf_path = str(destination)
    else:
        record.docx_path = str(destination)
    db.add(record)
    db.commit()
    return destination


def meeting_id_message(meeting: Meeting) -> str:
    """Readable identifier for error messages."""
    return f"{meeting.id} ({meeting.original_filename!r})"


@router.get(
    "/{meeting_id}/export/{fmt}",
    summary="Download the Minutes of Meeting as PDF or DOCX",
    responses={
        404: {"description": "Unknown meeting id"},
        409: {"description": "No minutes have been generated yet"},
    },
)
def export_minutes(
    meeting_id: int,
    fmt: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download generated minutes as a PDF or DOCX file.

    The file is rendered on first request and cached afterwards.
    """
    if fmt not in _EXPORT_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported export format {fmt!r}. Use one of: "
            + ", ".join(sorted(_EXPORT_TYPES)),
        )

    meeting = _get_or_404(db, meeting_id)
    path = _build_export(meeting, fmt, db)

    media_type = (
        "application/pdf"
        if fmt == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
    )