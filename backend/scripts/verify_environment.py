"""Throwaway diagnostic: verify the installed AI stack actually works.

Run with the project virtualenv::

    cd backend
    .venv/bin/python scripts/verify_environment.py

This is not part of the application. It reports which pretrained components
are usable and how fast they run on this machine, so model sizes can be
chosen for the demo.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time

CHECKS = (
    ("torch", "PyTorch"),
    ("transformers", "Hugging Face Transformers"),
    ("faster_whisper", "faster-whisper (ASR)"),
    ("peft", "PEFT / LoRA"),
    ("sentence_transformers", "Sentence Transformers"),
    ("docx", "python-docx"),
    ("reportlab", "ReportLab"),
    ("jiwer", "jiwer (WER)"),
    ("rouge_score", "ROUGE"),
)


def main() -> int:
    print("=" * 62)
    print("Environment verification")
    print("=" * 62)
    print(f"Python          : {sys.version.split()[0]}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg          : {'found ' + ffmpeg if ffmpeg else 'MISSING'}")
    print()

    missing: list[str] = []
    for module, label in CHECKS:
        ok = importlib.util.find_spec(module) is not None
        print(f"[{'ok ' if ok else 'MISS'}] {label}")
        if not ok:
            missing.append(module)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: .venv/bin/pip install -r requirements-ai.txt")
        return 1

    import torch

    print(f"\ntorch           : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CPU threads     : {torch.get_num_threads()}")

    # Confirm Whisper can load its weights (cached after the first run).
    print("\nLoading Whisper (downloads weights on first run)...")
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    WhisperModel("base", device="cpu", compute_type="int8")
    print(f"  whisper base ready in {time.perf_counter() - started:.1f}s")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())