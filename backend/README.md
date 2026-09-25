# Backend

FastAPI application implementing the meeting-minutes pipeline.

## Layout

```
app/
├── main.py               # app factory: CORS, static frontend, router
├── api/
│   ├── router.py         # aggregates the route modules
│   └── routes/
│       ├── health.py     # liveness + environment report
│       ├── meetings.py   # upload and the per-stage pipeline endpoints
│       └── history.py    # browsing previous meetings
├── core/
│   └── config.py         # pydantic-settings; every value is MMA_-prefixed
├── database/
│   └── session.py        # engine, session factory, get_db dependency
├── models/               # SQLAlchemy tables
├── schemas/              # Pydantic request/response and extraction models
├── services/             # the actual pipeline work
│   ├── storage.py        # saving uploads, validating size and type
│   ├── audio.py          # ffmpeg/ffprobe audio extraction
│   ├── transcription.py  # faster-whisper speech recognition
│   ├── extraction.py     # language-model extraction, JSON repair, grounding
│   ├── rules.py          # deterministic rule layer
│   ├── pipeline.py       # stage orchestration and status transitions
│   ├── mom.py            # assembling and rendering Minutes of Meeting
│   └── exporters.py      # PDF (reportlab) and DOCX (python-docx)
└── utils/                # shared helpers and error types
```

## Design rules

**Routes stay thin.** A route validates input, delegates to a service, and maps
the result to a response schema. The pipeline logic lives in `services/`, which
is what makes it testable without HTTP.

**Status is persisted, not inferred.** Every stage moves the meeting row to a
new status. This is what lets the interface report real progress by polling,
rather than guessing from a timer.

**Failures are data, not crashes.** A stage that cannot complete sets
`status="failed"` and writes a human-readable reason to `error_message`. The
response is still HTTP 200, because the request was handled successfully even
though the media could not be processed. HTTP error codes are reserved for
genuine request errors: unknown ids, wrong stages, unsupported formats.

**Heavy work runs in a worker thread.** Whisper and the language model are
synchronous and CPU-bound, so endpoints that run them use
`run_in_threadpool` and stay async at the boundary. This keeps the event loop
free to serve the progress-polling requests while a job is running.

**Optional dependencies are imported lazily.** `faster_whisper`,
`transformers`, `reportlab` and `python-docx` are only imported inside the
functions that need them. The API therefore starts and serves `/api/health`
even if the AI stack is missing, and reports what is unavailable.

**SQLite foreign keys are enabled per connection.** SQLite ignores
`ON DELETE CASCADE` unless `PRAGMA foreign_keys=ON` is set, and the default is
off. `database/session.py` sets it on every connection. Without it, deleting a
meeting leaves its transcript and minutes behind as unreferenced rows.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness plus which optional packages are importable |
| POST | `/api/meetings` | Upload a recording |
| GET | `/api/meetings` | List uploaded meetings |
| POST | `/api/meetings/{id}/extract-audio` | Extract audio |
| POST | `/api/meetings/{id}/transcribe` | Transcribe |
| GET | `/api/meetings/{id}/transcript` | Read the transcript |
| POST | `/api/meetings/{id}/analyse` | Extract structured information |
| POST | `/api/meetings/{id}/generate` | Render the minutes |
| GET | `/api/meetings/{id}/minutes` | Read the minutes |
| GET | `/api/meetings/{id}/export/{pdf\|docx}` | Download the report |
| POST | `/api/meetings/{id}/process` | Run the whole pipeline |
| GET | `/api/history` | Previous meetings, most recent first |
| GET | `/api/history/stats` | Aggregate counts |
| GET | `/api/history/{id}/full` | Transcript + extraction + minutes in one response |
| DELETE | `/api/meetings/{id}` | Delete a meeting and its derived files |

The per-stage endpoints exist so each module can be demonstrated and debugged
in isolation; the interface uses `/process`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Tests that need ffmpeg skip with a clear message when it is absent rather than
failing. Tests that need the language model are not in this suite: the model is
exercised by `training/evaluate.py`, which is the right place to measure it.
