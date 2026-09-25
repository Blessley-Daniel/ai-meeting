# AI-Based Meeting Minutes Generator

Convert a recorded meeting (Google Meet, Zoom, or plain audio) into structured
Minutes of Meeting using local, open-source AI models.

**Status:** Step 3 of 16 complete — project skeleton, upload module with SQLite
job tracking, and the full AI stack installed and measured on CPU.

## Architecture at a glance

```
Meeting recording (video/audio)
        ↓  ffmpeg
Audio (16 kHz mono WAV)
        ↓  faster-whisper (pretrained ASR)
Transcript
        ↓  our extraction layer (HF seq2seq model + optional LoRA adapter)
Structured JSON  {discussion_points, decisions, action_items, ...}
        ↓  our MoM renderer
Minutes of Meeting  →  viewer / PDF / DOCX / SQLite history
```

* **Pretrained:** Whisper (speech recognition), a small instruction-tuned
  seq2seq model (structured extraction), MiniLM (evaluation embeddings).
* **Ours:** the dataset, the extraction prompt/schema and validators, the
  MoM renderer, the web application, and the evaluation harness.

## Pipeline progress

Each stage is a separate service module behind its own endpoint, so the
pipeline can be exercised stage by stage while it is being built. Status
transitions are persisted on the meeting row, which is what lets the UI show
progress:

| Stage | Endpoint | Status before → after |
|---|---|---|
| Upload | `POST /api/meetings` | — → `uploaded` |
| Audio extraction | `POST /api/meetings/{id}/extract-audio` | `uploaded` → `audio_extracted` |
| Transcription | `POST /api/meetings/{id}/transcribe` | `audio_extracted` → `transcribed` |
| Extraction (NLP) | _not yet implemented_ | — |
| MoM generation | _not yet implemented_ | — |

Any stage that fails sets the meeting to `failed` and stores a human-readable
reason in `error_message`. The response is still HTTP 200, because the request
succeeded even though the media could not be processed.

Storage is split by purpose: `uploads/` holds raw input, `derived/` holds
intermediate artefacts such as extracted audio, and `generated/` holds
finished PDF/DOCX deliverables.

## Requirements

* Python 3.11–3.13 (tested on 3.13)
* `ffmpeg` / `ffprobe` on `PATH` — `sudo apt-get install -y ffmpeg`
* No API keys needed; everything runs locally.

## Quick start

```bash
# 1. System dependency
sudo apt-get install -y ffmpeg

# 2. Python environment
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. AI stack - torch MUST come from the CPU index first, otherwise pip
#    downloads ~2GB of CUDA libraries that are useless without a GPU.
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
.venv/bin/pip install -r requirements-ai.txt

# 4. Confirm everything works (downloads Whisper weights on first run)
.venv/bin/python scripts/verify_environment.py

# 5. Run
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/> for the page and
<http://localhost:8000/api/health> for the environment report.
Interactive API docs: <http://localhost:8000/docs>.

> **On a CUDA machine:** install `torch` normally and set
> `MMA_WHISPER_DEVICE=cuda`, `MMA_WHISPER_COMPUTE_TYPE=float16`.

## Tests

```bash
cd backend
.venv/bin/pip install pytest httpx
.venv/bin/python -m pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` at the project root and edit as needed. Every
setting is prefixed with `MMA_`. Defaults work out of the box.