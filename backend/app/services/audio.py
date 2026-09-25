"""Audio extraction: turn any uploaded recording into a 16 kHz mono WAV.

Speech recognition models have a fixed input expectation. Whisper was trained
on 16 kHz mono audio, so we normalise here, once, rather than letting every
later stage worry about sample rates and channel counts.

Two command-line tools from the FFmpeg suite are used:

* ``ffprobe`` - reads metadata without decoding: which streams exist, the
  duration, the codecs. We need this to detect audio-less files *before*
  spending time on extraction.
* ``ffmpeg``  - decodes and re-encodes the audio into a plain PCM WAV.

Both are invoked as subprocesses with an explicit argument list and
``shell=False``, so a filename can never be interpreted as a shell command.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.utils.errors import (
    FFmpegNotFoundError,
    MediaProcessingError,
    NoAudioStreamError,
)

# Whisper's expected input format.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"

# A 3-hour meeting extracts in well under 10 minutes on any modern CPU; this
# ceiling only exists so a corrupt file cannot hang a worker forever.
EXTRACTION_TIMEOUT_SECONDS = 3600
PROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class MediaInfo:
    """Metadata read from a media file by ffprobe."""

    duration_seconds: float | None
    has_audio: bool
    has_video: bool
    audio_codec: str | None
    video_codec: str | None
    size_bytes: int


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegNotFoundError(name)
    return path


def _sanitise_ffmpeg_message(message: str, path: Path) -> str:
    """Strip server-side paths out of an ffmpeg/ffprobe message.

    Tools like ffprobe echo the path they were given. That path is an internal
    implementation detail and leaks the deployment layout into a user-facing
    error, so replace it with the filename the user recognises. A leading
    "filename: " prefix is then dropped, because callers already name the file
    and repeating it reads as a stutter. A missing message becomes "unknown
    error" rather than an empty string, since ffmpeg can fail silently.
    """
    cleaned = (message or "").strip()
    if not cleaned:
        return "unknown error"

    cleaned = cleaned.replace(str(path), path.name)
    cleaned = re.sub(rf"^{re.escape(path.name)}:\s*", "", cleaned)
    return cleaned or "unknown error"


def probe_media(path: Path) -> MediaInfo:
    """Inspect a media file with ffprobe.

    Raises:
        FFmpegNotFoundError: ffprobe is not installed.
        MediaProcessingError: the file is missing, unreadable, or not media.
    """
    ffprobe = _require_executable("ffprobe")

    if not path.is_file():
        raise MediaProcessingError(f"File not found: {path}")

    command = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(
            f"ffprobe timed out after {PROBE_TIMEOUT_SECONDS}s on {path.name}"
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = _sanitise_ffmpeg_message(detail[-1] if detail else "", path)
        raise MediaProcessingError(f"ffprobe could not read {path.name}: {reason}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProcessingError(f"ffprobe returned unreadable output for {path.name}") from exc

    streams = payload.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]

    duration: float | None = None
    raw_duration = (payload.get("format") or {}).get("duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None
    if duration is None and audio_streams:
        try:
            duration = float(audio_streams[0]["duration"])
        except (KeyError, TypeError, ValueError):
            duration = None

    return MediaInfo(
        duration_seconds=duration,
        has_audio=bool(audio_streams),
        has_video=bool(video_streams),
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        video_codec=video_streams[0].get("codec_name") if video_streams else None,
        size_bytes=path.stat().st_size,
    )


def extract_audio(source: Path, destination: Path) -> MediaInfo:
    """Extract a normalised 16 kHz mono WAV track from ``source``.

    Returns the probed metadata of the *source* file, so the caller can store
    the duration without probing twice.

    Raises:
        FFmpegNotFoundError: ffmpeg is not installed.
        NoAudioStreamError: the source has no audio track.
        MediaProcessingError: ffmpeg failed or timed out.
    """
    info = probe_media(source)

    if not info.has_audio:
        raise NoAudioStreamError(source.name)

    ffmpeg = _require_executable("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-nostdin",          # never wait on stdin; we run unattended
        "-y",                # overwrite the destination if it exists
        "-loglevel", "error",
        "-i", str(source),
        "-vn",               # drop any video stream
        "-sn",               # drop subtitles
        "-dn",               # drop data streams
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", TARGET_CODEC,
        str(destination),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=EXTRACTION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        destination.unlink(missing_ok=True)
        raise MediaProcessingError(
            f"Audio extraction timed out after {EXTRACTION_TIMEOUT_SECONDS}s"
        ) from exc

    if completed.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        detail = (completed.stderr or "").strip().splitlines()
        reason = (
            _sanitise_ffmpeg_message(detail[-1], source)
            if detail
            else f"exit code {completed.returncode}"
        )
        raise MediaProcessingError(
            f"ffmpeg failed to extract audio from {source.name}: {reason}"
        )

    return info