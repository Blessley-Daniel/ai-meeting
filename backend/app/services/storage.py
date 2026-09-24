"""Storing uploaded recordings safely on disk.

Two rules drive this module:

1. **Never trust the client filename.** It is used only for display and to
   read the extension. The file on disk is named from a fresh UUID, so a
   malicious name such as ``../../etc/passwd`` cannot escape ``uploads/``.
2. **Never buffer the whole upload in memory.** The stream is copied in
   fixed-size chunks and aborted as soon as the size limit is crossed, so a
   multi-gigabyte upload cannot exhaust RAM.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import get_settings
from app.models.meeting import MediaType
from app.utils.errors import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError

# 1 MiB: large enough that syscall overhead is negligible, small enough that an
# oversized upload is rejected promptly.
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class StoredUpload:
    """Result of persisting an upload."""

    original_filename: str
    stored_path: Path
    media_type: MediaType
    size_bytes: int


def classify_extension(extension: str) -> MediaType:
    """Map a lowercase extension such as ``.mp4`` to a :class:`MediaType`."""
    settings = get_settings()
    if extension in settings.allowed_video_extensions:
        return MediaType.VIDEO
    if extension in settings.allowed_audio_extensions:
        return MediaType.AUDIO
    raise UnsupportedFileTypeError(extension or "<none>", settings.allowed_extensions)


async def save_upload(upload: UploadFile) -> StoredUpload:
    """Validate and stream an uploaded recording into the uploads directory.

    Raises:
        EmptyFileError: the client sent no bytes.
        UnsupportedFileTypeError: the extension is not allowed.
        FileTooLargeError: the stream exceeded ``max_upload_size_mb``.
    """
    settings = get_settings()
    settings.ensure_directories()

    original_filename = upload.filename or "unnamed"
    extension = Path(original_filename).suffix.lower()
    media_type = classify_extension(extension)

    destination = settings.upload_dir / f"{uuid.uuid4().hex}{extension}"
    limit = settings.max_upload_size_bytes
    written = 0

    try:
        async with aiofiles.open(destination, "wb") as handle:
            while chunk := await upload.read(CHUNK_SIZE):
                written += len(chunk)
                if written > limit:
                    raise FileTooLargeError(limit)
                await handle.write(chunk)
    except FileTooLargeError:
        # Never leave a partial file behind on a rejected upload.
        destination.unlink(missing_ok=True)
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise EmptyFileError(original_filename)

    return StoredUpload(
        original_filename=original_filename,
        stored_path=destination,
        media_type=media_type,
        size_bytes=written,
    )