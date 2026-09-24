# AI-Based Meeting Minutes Generator

Convert a recorded meeting (Google Meet, Zoom, or plain audio) into structured
Minutes of Meeting using local, open-source AI models.

**Status:** Step 1 of 16 complete — project skeleton, configuration, and a
runnable FastAPI service with a health/environment endpoint.

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

## Requirements

* Python 3.11–3.13 (tested on 3.13)
* `ffmpeg` / `ffprobe` on `PATH` — `sudo apt-get install -y ffmpeg`
* No API keys needed; everything runs locally.

## Quick start (Step 1 scope)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/> for the page and
<http://localhost:8000/api/health> for the environment report.
Interactive API docs: <http://localhost:8000/docs>.

## Tests

```bash
cd backend
.venv/bin/pip install pytest httpx
.venv/bin/python -m pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` at the project root and edit as needed. Every
setting is prefixed with `MMA_`. Defaults work out of the box.