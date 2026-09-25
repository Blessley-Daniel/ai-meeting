"""Speech recognition: turn a normalised WAV into a timestamped transcript.

Uses `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_, which runs
OpenAI's Whisper weights through CTranslate2. On CPU with ``int8`` it is
several times faster than the reference implementation, which is what makes
this project practical on an ordinary laptop.

Three things this module handles that a naive wrapper does not:

1. **Model caching.** Loading Whisper takes seconds. A module-level cache keyed
   by (size, device, compute_type) means the cost is paid once per process,
   not once per request.
2. **Long recordings.** A two-hour meeting cannot be decoded in one pass
   without a lot of memory. Transcription is therefore *streamed* segment by
   segment.
3. **Empty audio.** Recordings that contain only silence produce no segments.
   That is reported as a domain error rather than an empty transcript, so the
   UI can say something useful.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.utils.errors import EmptyTranscriptError, TranscriptionError

logger = logging.getLogger(__name__)

# Loading a model is expensive, so cache one per distinct configuration.
# The lock stops two concurrent requests from loading it simultaneously.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TranscriptSegment:
    """One recognised span of speech."""

    index: int
    start: float
    end: float
    text: str
    # Whisper's confidence for the average token in the segment, in [0, 1].
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TranscriptionResult:
    """A complete transcript plus the metadata captured while producing it."""

    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    duration_seconds: float | None = None
    processing_seconds: float = 0.0
    model_size: str = ""

    @property
    def realtime_factor(self) -> float | None:
        """Audio seconds processed per wall-clock second. Higher is better."""
        if not self.duration_seconds or self.processing_seconds <= 0:
            return None
        return self.duration_seconds / self.processing_seconds


def _resolve_device(requested: str) -> str:
    """Turn the ``auto`` setting into a concrete device."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # torch absent: ctranslate2 can still run on CPU
        return "cpu"


def load_model(settings: Settings | None = None):
    """Return a cached ``WhisperModel`` for the configured settings.

    Raises:
        TranscriptionError: faster-whisper is missing or the model cannot load
            (most often: no network access on the very first run).
    """
    settings = settings or get_settings()
    device = _resolve_device(settings.whisper_device)
    key = (settings.whisper_model_size, device, settings.whisper_compute_type)

    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        # Re-check: another thread may have loaded it while we waited.
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on install
            raise TranscriptionError(
                "faster-whisper is not installed. "
                "Install it with: pip install -r requirements-ai.txt"
            ) from exc

        threads = settings.whisper_cpu_threads or 0
        if device == "cpu" and threads > 0:
            try:
                import torch

                torch.set_num_threads(threads)
            except ImportError:  # pragma: no cover
                pass

        logger.info(
            "Loading Whisper %s on %s (%s)",
            settings.whisper_model_size,
            device,
            settings.whisper_compute_type,
        )
        started = time.perf_counter()
        try:
            model = WhisperModel(
                settings.whisper_model_size,
                device=device,
                compute_type=settings.whisper_compute_type,
                cpu_threads=threads,
                num_workers=1,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise TranscriptionError(
                f"Could not load the Whisper '{settings.whisper_model_size}' model: {exc}"
            ) from exc

        logger.info("Whisper ready in %.1fs", time.perf_counter() - started)
        _MODEL_CACHE[key] = model
        return model


def _average_logprob(segment) -> float | None:
    """Average token log-probability of a segment, converted to probability."""
    avg = getattr(segment, "avg_logprob", None)
    if avg is None:
        return None
    if avg >= 0:
        return float(avg)
    return float(math.exp(avg))


def transcribe_audio(
    audio_path: Path,
    settings: Settings | None = None,
    *,
    progress: Callable[[int, "TranscriptSegment"], None] | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV file into text and timestamped segments.

    Args:
        audio_path: 16 kHz mono WAV produced by the audio-extraction stage.
        settings: Override configuration (used by tests).
        progress: Optional callback ``(segments_so_far, latest)`` invoked as
            each segment arrives, so a caller can report progress.

    Raises:
        TranscriptionError: the model failed to load or decode the audio.
        EmptyTranscriptError: the audio decoded but contained no recognisable
            speech (silence, or a music-only track).
    """
    settings = settings or get_settings()

    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    model = load_model(settings)

    # vad_filter drops non-speech spans. Two benefits: silence-only recordings
    # are detected cheaply, and long silent stretches stop Whisper from
    # hallucinating repeated text over them.
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise TranscriptionError(f"Could not decode {audio_path.name}: {exc}") from exc

    started = time.perf_counter()
    collected: list[TranscriptSegment] = []

    # The generator returned by faster-whisper does the actual work, lazily;
    # iterating is what triggers decoding.
    try:
        for raw in segments_iter:
            text = (raw.text or "").strip()
            if not text:
                continue
            segment = TranscriptSegment(
                index=len(collected),
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                confidence=_average_logprob(raw),
            )
            collected.append(segment)
            if progress is not None:
                progress(len(collected), segment)
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Transcription failed part-way: {exc}") from exc

    elapsed = time.perf_counter() - started

    if not collected:
        raise EmptyTranscriptError(audio_path.name)

    full_text = " ".join(segment.text for segment in collected)
    detected_language = getattr(info, "language", None) or settings.whisper_language

    result = TranscriptionResult(
        text=full_text,
        segments=collected,
        language=detected_language,
        language_probability=getattr(info, "language_probability", None),
        duration_seconds=getattr(info, "duration", None),
        processing_seconds=elapsed,
        model_size=settings.whisper_model_size,
    )

    logger.info(
        "Transcribed %s: %d segments, %.1fs audio in %.1fs (%.1fx real time)",
        audio_path.name,
        len(collected),
        result.duration_seconds or 0.0,
        elapsed,
        result.realtime_factor or 0.0,
    )
    return result


def segments_to_dicts(segments: list[TranscriptSegment]) -> list[dict]:
    """Serialise segments for storage in JSON."""
    return [
        {
            "index": s.index,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text,
            "confidence": round(s.confidence, 4) if s.confidence is not None else None,
        }
        for s in segments
    ]


def dicts_to_segments(payload: list[dict]) -> list[TranscriptSegment]:
    """Rebuild segments loaded from JSON."""
    return [
        TranscriptSegment(
            index=item.get("index", position),
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            text=item.get("text", ""),
            confidence=item.get("confidence"),
        )
        for position, item in enumerate(payload)
    ]