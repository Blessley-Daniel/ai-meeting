# Repository notes

AI-Based Meeting Minutes Generator. FastAPI backend + vanilla-JS frontend that
turns a meeting recording into structured Minutes of Meeting using local,
CPU-only open-source models.

## Layout

* `backend/` — FastAPI app. `app/services/` holds the pipeline; `app/api/routes/`
  is thin HTTP only.
* `frontend/` — static HTML/CSS/JS served by the backend.
* `training/` — dataset prep, LoRA fine-tuning, evaluation.
* `datasets/` — synthetic transcript→minutes JSONL under `processed/`.
* `docs/` — academic write-up and saved evaluation results.

## Commands

```bash
cd backend
./.venv/bin/python -m pytest tests/ -q          # full suite (~6 s)
./.venv/bin/python -m pytest -m slow tests/     # real-model tests, opt-in
./.venv/bin/uvicorn app.main:app --port 12000   # run the server
```

Tests use fakes for the models, so they are fast and deterministic. Accuracy is
measured separately by `training/evaluate.py` against real audio and the
held-out split.

## Conventions

* Routes validate and delegate; logic lives in `services/`.
* A failed stage sets `status="failed"` and `error_message`, and still returns
  HTTP 200 — the request succeeded even though the media could not be
  processed. HTTP 4xx is reserved for bad requests (unknown id, wrong stage,
  unsupported format).
* Heavy model work runs via `run_in_threadpool` so progress polling stays
  responsive.
* Optional AI/document dependencies are imported lazily inside the functions
  that use them, so `/api/health` works without the ML stack installed.

## Gotchas

* **SQLite needs `PRAGMA foreign_keys=ON` per connection.** It is off by
  default, so `ON DELETE CASCADE` silently does nothing. `database/session.py`
  sets it on connect. If you add another engine, do the same.
* **Do not `db.delete()` a child row and then write to the parent.** The parent
  relationship still references the deleted instance, and `db.add(parent)`
  raises `InvalidRequestError`. Mutate the child in place instead — see
  `pipeline.transcribe_meeting`.
* **User-facing errors must not contain server paths.** ffmpeg/ffprobe echo the
  absolute path they were given; sanitise with
  `audio._sanitise_ffmpeg_message`.
* `MinutesDocument.text` and `extraction` are non-nullable. Clear them with
  `""` / `{}`, not `None`.
* Dataset splits are grouped by scenario, not by example. Variants paraphrase
  the same meeting, so a random split would leak near-duplicates into test.
