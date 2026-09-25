"""Pipeline orchestration for a meeting job.

Each stage lives in its own service module; this module is only responsible
for moving a :class:`Meeting` through its lifecycle states:

    uploaded -> extracting_audio -> audio_extracted -> transcribing -> ...

Design choices worth noting:

* **Failures are recorded, not raised through.** A stage that fails sets the
  job to ``failed`` and stores the reason in ``error_message``. The API then
  returns a normal 200 response describing the failure, so one bad file never
  takes the server down.
* **Blocking work runs in a threadpool.** ffmpeg and, later, the ML models are
  synchronous and CPU-bound. Running them directly in an ``async def`` endpoint
  would block the event loop and freeze every other request.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.meeting import Meeting, MeetingStatus
from app.services import audio
from app.utils.errors import MediaProcessingError, NoAudioStreamError

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