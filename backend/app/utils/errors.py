"""Domain-level exceptions raised by services.

Routes translate these into HTTP responses, which keeps services free of any
web-framework imports and makes them directly unit-testable.
"""

from __future__ import annotations


class MeetingMinutesError(Exception):
    """Base class for all expected application errors."""


class UnsupportedFileTypeError(MeetingMinutesError):
    """The uploaded file extension is not on the allow-list."""

    def __init__(self, filename: str, allowed: set[str]) -> None:
        self.filename = filename
        self.allowed = allowed
        super().__init__(
            f"Unsupported file type for {filename!r}. Allowed: {sorted(allowed)}"
        )


class FileTooLargeError(MeetingMinutesError):
    """The upload exceeded the configured maximum size."""

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(f"File exceeds the {limit_bytes} byte upload limit")


class EmptyFileError(MeetingMinutesError):
    """The upload contained no data."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"Uploaded file {filename!r} is empty")


class MediaNotFoundError(MeetingMinutesError):
    """No meeting (or no stored file) matches the given identifier."""

    def __init__(self, identifier: int | str) -> None:
        self.identifier = identifier
        super().__init__(f"No meeting found for {identifier!r}")


class MediaProcessingError(MeetingMinutesError):
    """ffmpeg/ffprobe failed to process the media."""


class FFmpegNotFoundError(MeetingMinutesError):
    """The ffmpeg or ffprobe executable is not available on PATH."""

    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(
            f"{executable!r} was not found on PATH. "
            "Install it with: sudo apt-get install -y ffmpeg"
        )


class NoAudioStreamError(MeetingMinutesError):
    """The recording contains no audio track, so it cannot be transcribed."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(
            f"{filename!r} has no audio stream. "
            "Please upload a recording that contains speech."
        )