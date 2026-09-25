"""Generate the reproducible sample meeting recording and its ground truth.

The project needs meeting audio with *known* text so that transcription
accuracy (WER) can be measured honestly. No public meeting corpus ships with
a transcript we can redistribute, so this script synthesises one locally with
``espeak-ng``: several utterances, each spoken with a different voice to
simulate distinct participants, concatenated and wrapped in an MP4 container.

The output is **synthetic**, and must be described as such in any report. It
is useful for verifying the pipeline end to end and for measuring how ASR
handles realistic features (technical terms, proper nouns, turn changes). It
is not a substitute for a corpus of real human meetings.

Usage::

    cd backend
    .venv/bin/python scripts/make_sample_recording.py

Produces:
    datasets/raw/sample_meeting/sample_meeting.mp4
    datasets/raw/sample_meeting/ground_truth.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw" / "sample_meeting"

# Each turn: (voice, words per minute, text).
# Different voices approximate different speakers so the transcript exercises
# speaker changes. The text deliberately includes:
#   * proper nouns (Priya, Rahul)          -> hard for ASR
#   * a domain term (PostgreSQL)           -> hard for ASR
#   * explicit decisions and action items  -> for the NLP stages
#   * one action item with NO deadline     -> tests "Not specified" handling
#   * one informal deadline ("next Monday")-> tests relative date extraction
TURNS: list[tuple[str, int, str]] = [
    (
        "en-us", 145,
        "Good morning everyone. Let's start the weekly project sync. "
        "First item, the authentication module. I have finished the login and "
        "signup endpoints and they are working in the staging environment.",
    ),
    (
        "en-gb", 150,
        "Great work Priya. I will review the code and run the security tests "
        "on Wednesday. Also we need to decide on the database.",
    ),
    (
        "en-us", 140,
        "The team decided to use PostgreSQL for the main database. "
        "Rahul will update the schema documentation by next Monday.",
    ),
    (
        "en-gb", 150,
        "Agreed. I will also prepare the deployment checklist. "
        "No deadlines are needed for that one yet.",
    ),
    (
        "en-us", 145,
        "Good. So to conclude, the authentication work is on track, "
        "PostgreSQL is confirmed, and we will meet again next week.",
    ),
]


def _require(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        sys.exit(
            f"error: {name!r} not found on PATH.\n"
            "  espeak-ng : sudo apt-get install -y espeak-ng\n"
            "  ffmpeg    : sudo apt-get install -y ffmpeg"
        )
    return path


def main() -> int:
    espeak = _require("espeak-ng")
    ffmpeg = _require("ffmpeg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []

        for index, (voice, speed, text) in enumerate(TURNS):
            raw = tmpdir / f"turn{index}_raw.wav"
            normalised = tmpdir / f"turn{index}.wav"
            subprocess.run(
                [espeak, "-v", voice, "-s", str(speed), "-w", str(raw), text],
                check=True,
            )
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
                 "-ar", "16000", "-ac", "1", str(normalised)],
                check=True,
            )
            parts.append(normalised)

        # Concatenate the turns into one continuous audio track.
        combined = tmpdir / "meeting.wav"
        inputs: list[str] = []
        for part in parts:
            inputs += ["-i", str(part)]
        filter_graph = (
            "".join(f"[{i}:a]" for i in range(len(parts)))
            + f"concat=n={len(parts)}:v=0:a=1[out]"
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", *inputs,
             "-filter_complex", filter_graph, "-map", "[out]",
             "-ar", "16000", "-ac", "1", str(combined)],
            check=True,
        )

        # Wrap in a video container so the sample mimics a real Meet export.
        video_out = OUTPUT_DIR / "sample_meeting.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=5",
             "-i", str(combined),
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(video_out)],
            check=True,
        )

        # Also keep the audio-only version, which is what the pipeline extracts.
        audio_out = OUTPUT_DIR / "sample_meeting_16k.wav"
        shutil.copy(combined, audio_out)

    ground_truth = {
        "source": "synthetic",
        "generator": "espeak-ng 1.52.0 + ffmpeg",
        "language": "en",
        "note": (
            "Synthetic audio generated locally by scripts/make_sample_recording.py. "
            "Not a real human meeting recording. Text below is what was actually "
            "spoken, so it is a valid reference for computing WER."
        ),
        "turns": [
            {"voice": voice, "speed_wpm": speed, "text": text}
            for voice, speed, text in TURNS
        ],
        "transcript": " ".join(text for _voice, _speed, text in TURNS),
    }
    truth_path = OUTPUT_DIR / "ground_truth.json"
    truth_path.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")

    size_kb = video_out.stat().st_size // 1024
    print(f"wrote {video_out.relative_to(PROJECT_ROOT)} ({size_kb} KB)")
    print(f"wrote {audio_out.relative_to(PROJECT_ROOT)}")
    print(f"wrote {truth_path.relative_to(PROJECT_ROOT)}")
    print(f"reference words: {len(ground_truth['transcript'].split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())