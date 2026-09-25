"""Pipeline orchestration for a meeting job.

Each stage lives in its own service module; this module is only responsible
for moving a :class:`Meeting` through its lifecycle states:

    uploaded -> extracting_audio -> audio_extracted -> transcribing
             -> transcribed -> analysing -> ...

Design choices worth noting:

* **Failures are recorded, not raised through.** A stage that fails sets the
  job to ``failed`` and stores the reason in ``error_message``. The API then
  returns a normal 200 response describing the failure, so one bad file never
  takes the server down.
* **Blocking work runs in a threadpool.** ffmpeg and the ML models are
  synchronous and CPU-bound. Running them directly in an ``async def`` endpoint
  would block the event loop and freeze every other request.
* **Stages are individually callable.** Keeping each stage as its own function
  makes it testable in isolation and lets the API expose them separately while
  the pipeline is still being built.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.meeting import Meeting, MeetingStatus
from app.models.minutes import MinutesDocument
from app.models.transcript import Transcript
from app.services import audio, extraction, mom as mom_service, transcription
from app.utils.errors import (
    EmptyTranscriptError,
    ExtractionError,
    MediaProcessingError,
    NoAudioStreamError,
    TranscriptionError,
)

logger = logging.getLogger(__name__)


def _set_status(
    db: Session,
    meeting: Meeting,
    status: MeetingStatus,
    *,
    error: str | None = None,
) -> None:
    """Persist a status transition.

    Committed individually so the UI can poll and observe progress while a
    long stage is still running.
    """
    meeting.status = status
    meeting.error_message = error
    db.add(meeting)
    db.commit()
    db.refresh(meeting)


def extract_audio_for_meeting(db: Session, meeting: Meeting) -> Meeting:
    """Run the audio-extraction stage for ``meeting``.

    Blocking: call via ``run_in_threadpool`` from an async endpoint.

    Returns the refreshed meeting; check ``status`` for the outcome.
    """
    settings = get_settings()

    _set_status(db, meeting, MeetingStatus.EXTRACTING_AUDIO)

    source = Path(meeting.stored_path)
    destination = settings.audio_dir / f"meeting_{meeting.id}.wav"

    try:
        info = audio.extract_audio(source, destination)
    except NoAudioStreamError:
        # Report the user's own filename, not the internal UUID we stored it
        # under, so the message is actionable.
        message = (
            f"{meeting.original_filename!r} has no audio stream. "
            "Please upload a recording that contains speech."
        )
        _set_status(db, meeting, MeetingStatus.FAILED, error=message)
        logger.warning("Meeting %s has no audio stream", meeting.id)
        return meeting
    except MediaProcessingError as exc:
        _set_status(db, meeting, MeetingStatus.FAILED, error=str(exc))
        logger.error("Audio extraction failed for meeting %s: %s", meeting.id, exc)
        return meeting

    meeting.audio_path = str(destination)
    meeting.duration_seconds = info.duration_seconds
    _set_status(db, meeting, MeetingStatus.AUDIO_EXTRACTED)

    logger.info(
        "Meeting %s: extracted %.1fs of audio to %s",
        meeting.id,
        info.duration_seconds or 0.0,
        destination.name,
    )
    return meeting


def transcribe_meeting(db: Session, meeting: Meeting) -> Meeting:
    """Run the speech-recognition stage for ``meeting``.

    Requires the audio-extraction stage to have completed. Blocking: call via
    ``run_in_threadpool``.

    Re-running replaces any existing transcript for the meeting, so a failed
    attempt can be retried without duplicating rows.
    """
    settings = get_settings()

    _set_status(db, meeting, MeetingStatus.TRANSCRIBING)

    if not meeting.audio_path:
        message = (
            "No extracted audio is available for this meeting. "
            "Run audio extraction first."
        )
        _set_status(db, meeting, MeetingStatus.FAILED, error=message)
        return meeting

    audio_file = Path(meeting.audio_path)

    try:
        result = transcription.transcribe_audio(audio_file, settings)
    except EmptyTranscriptError:
        # Report the user's original filename rather than our internal
        # meeting_<id>.wav, so the message is meaningful to them.
        message = (
            f"No speech was recognised in {meeting.original_filename!r}. "
            "The recording may be silent, or contain only music or "
            "background noise."
        )
        _set_status(db, meeting, MeetingStatus.FAILED, error=message)
        logger.warning("Meeting %s produced an empty transcript", meeting.id)
        return meeting
    except TranscriptionError as exc:
        _set_status(db, meeting, MeetingStatus.FAILED, error=str(exc))
        logger.error("Transcription failed for meeting %s: %s", meeting.id, exc)
        return meeting

    # A retry must refresh the existing row rather than delete and recreate it.
    # Deleting through the session leaves the old instance attached to
    # meeting.transcript, and the later db.add(meeting) then tries to cascade
    # to a deleted object and raises. Mutating in place keeps both sides of the
    # relationship consistent, and matches how the analysis stage upserts.
    record = meeting.transcript
    if record is None:
        record = Transcript(meeting_id=meeting.id)

    record.text = result.text
    record.segments = transcription.segments_to_dicts(result.segments)
    record.language = result.language
    record.language_probability = result.language_probability
    record.duration_seconds = result.duration_seconds
    record.model_name = f"faster-whisper:{result.model_size}"
    record.processing_seconds = result.processing_seconds
    db.add(record)

    # A new transcript invalidates the analysis and minutes derived from it.
    # Their rows are mutated, not deleted, so the relationship never holds a
    # deleted instance.
    stale_minutes = meeting.minutes
    if stale_minutes is not None:
        stale_minutes.extraction = {}
        stale_minutes.text = ""
        stale_minutes.pdf_path = None
        stale_minutes.docx_path = None
        db.add(stale_minutes)

    _set_status(db, meeting, MeetingStatus.TRANSCRIBED)

    logger.info(
        "Meeting %s: transcript with %d segments, %d words",
        meeting.id,
        len(result.segments),
        len(result.text.split()),
    )
    return meeting


def analyse_meeting(db: Session, meeting: Meeting) -> Meeting:
    """Run the AI information-extraction stage for ``meeting``.

    Requires a transcript. Blocking: call via ``run_in_threadpool``.

    The structured extraction is stored as JSON on the minutes row, which is
    the authoritative record. Rendered text is produced by the next stage, so
    that changing the presentation never requires re-running the model.
    """
    settings = get_settings()

    _set_status(db, meeting, MeetingStatus.ANALYSING)

    transcript = meeting.transcript
    if transcript is None or not transcript.text.strip():
        message = (
            "No transcript is available for this meeting. "
            "Run transcription first."
        )
        _set_status(db, meeting, MeetingStatus.FAILED, error=message)
        return meeting

    try:
        result, diagnostics = extraction.extract_meeting_information(
            transcript.text, settings
        )
    except ExtractionError as exc:
        _set_status(db, meeting, MeetingStatus.FAILED, error=str(exc))
        logger.error("Extraction failed for meeting %s: %s", meeting.id, exc)
        return meeting

    existing = meeting.minutes
    if existing is None:
        record = MinutesDocument(meeting_id=meeting.id)
    else:
        record = existing

    record.extraction = result.to_json_dict()
    record.extraction_model = diagnostics.model_name
    record.rules_used = diagnostics.rule_sources
    record.dropped_ungrounded = diagnostics.dropped_ungrounded
    record.extraction_seconds = diagnostics.processing_seconds
    # Invalidate any previously exported files: they now describe an older
    # extraction and would be misleading if downloaded.
    record.pdf_path = None
    record.docx_path = None
    db.add(record)

    _set_status(db, meeting, MeetingStatus.ANALYSED)

    logger.info(
        "Meeting %s: extracted %d discussion points, %d decisions, %d action items",
        meeting.id,
        len(result.discussion_points),
        len(result.decisions),
        len(result.action_items),
    )
    return meeting


def generate_minutes(db: Session, meeting: Meeting) -> Meeting:
    """Render Minutes of Meeting from the stored extraction.

    Requires the analysis stage. Blocking, but fast: this is pure formatting
    with no model involved, which is why it can be re-run cheaply whenever the
    presentation changes.
    """
    _set_status(db, meeting, MeetingStatus.GENERATING)

    record = meeting.minutes
    if record is None or not record.extraction:
        message = (
            "No extracted information is available for this meeting. "
            "Run the analysis step first."
        )
        _set_status(db, meeting, MeetingStatus.FAILED, error=message)
        return meeting

    extraction_result = extraction.MeetingExtraction.model_validate(record.extraction)

    transcript = meeting.transcript
    document = mom_service.build_mom(
        extraction_result,
        source_filename=meeting.original_filename,
        asr_model=transcript.model_name if transcript else "",
        extraction_model=record.extraction_model or "",
        fallback_title=mom_service.title_from_filename(meeting.original_filename),
    )

    record.text = mom_service.render_text(document)
    db.add(record)

    _set_status(db, meeting, MeetingStatus.COMPLETED)

    logger.info("Meeting %s: minutes generated", meeting.id)
    return meeting


def process_meeting(
    db: Session,
    meeting: Meeting,
    *,
    on_stage: "Callable[[str], None] | None" = None,
) -> Meeting:
    """Run every pipeline stage in order, stopping at the first failure.

    This exists so the web interface can start one job and poll a single
    status, instead of orchestrating four endpoints itself. The individual
    stage functions remain available for debugging and for demonstrating a
    single module during a project evaluation.

    Args:
        on_stage: optional callback invoked with the status name before each
            stage runs, used to stream progress.
    """

    def notify(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    stages = (
        ("extracting_audio", extract_audio_for_meeting),
        ("transcribing", transcribe_meeting),
        ("analysing", analyse_meeting),
        ("generating", generate_minutes),
    )

    for name, stage in stages:
        notify(name)
        meeting = stage(db, meeting)
        if meeting.status is MeetingStatus.FAILED:
            logger.warning(
                "Meeting %s: pipeline stopped at %s (%s)",
                meeting.id,
                name,
                meeting.error_message,
            )
            return meeting

    notify("completed")
    return meeting