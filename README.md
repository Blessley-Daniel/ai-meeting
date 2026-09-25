# AI-Based Meeting Minutes Generator

Convert a recorded meeting (Google Meet, Zoom, or plain audio) into structured
Minutes of Meeting using local, open-source AI models.

**Status:** Steps 1–13 complete. The end-to-end pipeline runs: upload →
audio extraction → transcription → AI extraction → Minutes of Meeting →
PDF/DOCX export → SQLite history. Step 14 (optional LoRA fine-tuning) and
Step 15 (evaluation harness) are implemented and runnable; see
`training/README.md` and the evaluation results in `docs/`.

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
| Extraction (NLP) | `POST /api/meetings/{id}/analyse` | `transcribed` → `analysed` |
| MoM generation | `POST /api/meetings/{id}/generate` | `analysed` → `completed` |
| Export | `GET /api/meetings/{id}/export?format=pdf\|docx` | — |
| One-shot run | `POST /api/meetings/{id}/process` | any → `completed` |
| History | `GET /api/meetings` | — |

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

The frontend and the API are served by the same process, and the frontend
calls `/api` as a relative path. It therefore talks to whatever port you
started the server on, so the `--port` flag is the only thing to change.

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