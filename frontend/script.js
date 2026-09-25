/**
 * Interface logic for the AI-Based Meeting Minutes Generator.
 *
 * Design notes, because the choices here are deliberate:
 *
 * 1. Progress is *derived from the job's persisted status*, not from a timer.
 *    The pipeline runs in a single server request, so the page cannot observe
 *    each stage directly. Instead it polls GET /api/meetings/{id} and maps the
 *    stored status onto the five labelled steps. The progress shown therefore
 *    reflects real server state, and does not advance on its own if the server
 *    is stuck.
 *
 * 2. All text from the API is inserted with textContent, never innerHTML.
 *    Transcripts and model output are untrusted input; using innerHTML would
 *    make a transcript containing markup execute in the page.
 *
 * 3. Failures are surfaced from the API's own error_message rather than being
 *    replaced with a generic message, so the user sees the actual reason.
 */

"use strict";

const API = "/api";

/**
 * Hosts whose traffic passes through the GitHub Codespaces port-forwarding
 * tunnel rather than reaching uvicorn directly.
 */
const FORWARDED_PORT_HOST_SUFFIX = ".app.github.dev";

/**
 * Approximate largest request body the Codespaces tunnel accepts.
 *
 * The tunnel's front-end answers an oversized upload with its own nginx 413
 * before the request reaches this application, so MMA_MAX_UPLOAD_SIZE_MB is not
 * the binding limit when the UI is opened through a forwarded port. GitHub has
 * raised this over time, so it is treated as a pre-flight hint rather than a
 * guarantee: the authoritative check remains the server's response.
 */
const TUNNEL_BODY_LIMIT_BYTES = 15 * 1024 * 1024;

/** True when this page is being served through a Codespaces forwarded port. */
function isForwardedPortHost() {
  return location.hostname.endsWith(FORWARDED_PORT_HOST_SUFFIX);
}

/** Raised locally when a file cannot plausibly fit through the tunnel. */
class UploadTooLargeError extends Error {
  constructor(sizeBytes) {
    super(
      `The recording is ${formatBytes(sizeBytes)}, which is larger than the ` +
        `${formatBytes(TUNNEL_BODY_LIMIT_BYTES)} that the Codespaces forwarded ` +
        "port accepts."
    );
    this.name = "UploadTooLargeError";
    this.status = 413;
    this.sizeBytes = sizeBytes;
  }
}

/** Ordered pipeline stages, matching the labels shown in the interface. */
const STAGES = ["uploading", "extracting", "transcribing", "analysing", "generating"];

/**
 * Backend status -> how many of the five stages are already finished.
 *
 * The backend distinguishes an in-progress status ("extracting_audio") from
 * the completed one ("audio_extracted"), and both mean the same number of
 * finished stages. Mapping to a count rather than an index keeps the display
 * accurate across every transition instead of skipping steps.
 */
const STATUS_COMPLETED = {
  uploading: 0,
  uploaded: 1, // upload finished; audio extraction is next
  extracting_audio: 1,
  audio_extracted: 2,
  transcribing: 2,
  transcribed: 3,
  analysing: 3,
  analysed: 4, // analysis finished; minutes generation is next
  generating: 4,
  completed: 5,
};

const state = {
  file: null,
  meetingId: null,
  minutes: null,
  pollTimer: null,
};

// --------------------------------------------------------------------------
// Element lookup
// --------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const el = {
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  fileLabel: $("file-label"),
  generateBtn: $("generate-btn"),
  clearBtn: $("clear-btn"),
  progressCard: $("progress-card"),
  progressNote: $("progress-note"),
  progressFill: $("progress-fill"),
  steps: Array.from(document.querySelectorAll(".step")),
  errorCard: $("error-card"),
  errorMessage: $("error-message"),
  errorHint: $("error-hint"),
  results: $("results"),
  momTitle: $("mom-title"),
  momMeta: $("mom-meta"),
  momFacts: $("mom-facts"),
  overviewCard: $("overview-card"),
  overviewText: $("overview-text"),
  discussionCard: $("discussion-card"),
  discussionList: $("discussion-list"),
  decisionsCard: $("decisions-card"),
  decisionsList: $("decisions-list"),
  actionsCard: $("actions-card"),
  actionsBody: $("actions-body"),
  actionsNote: $("actions-note"),
  conclusionCard: $("conclusion-card"),
  conclusionText: $("conclusion-text"),
  provenanceList: $("provenance-list"),
  pdfBtn: $("pdf-btn"),
  docxBtn: $("docx-btn"),
  transcriptBtn: $("transcript-btn"),
  transcriptDialog: $("transcript-dialog"),
  transcriptText: $("transcript-text"),
  transcriptMeta: $("transcript-meta"),
  transcriptClose: $("transcript-close"),
  historyList: $("history-list"),
  historyEmpty: $("history-empty"),
  historyRefresh: $("history-refresh"),
  modelBadge: $("model-badge"),
};

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes === 0) return `${rest}s`;
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

function formatDate(value) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

function isUnspecified(value) {
  return !value || /^not specified$/i.test(String(value).trim());
}

/** Replace a value with a labelled placeholder when it is missing. */
function orUnspecified(value) {
  return isUnspecified(value) ? "Not specified" : String(value);
}

function show(node, visible) {
  node.hidden = !visible;
}

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Create a <li> whose text is set safely. */
function listItem(text, className) {
  const item = document.createElement("li");
  item.textContent = text;
  if (className) item.className = className;
  return item;
}

function renderList(container, values, emptyText) {
  clearChildren(container);
  const items = (values || []).filter((value) => String(value || "").trim());
  if (items.length === 0) {
    container.appendChild(listItem(emptyText, "small"));
    return;
  }
  items.forEach((value) => container.appendChild(listItem(value)));
}

// --------------------------------------------------------------------------
// Server communication
// --------------------------------------------------------------------------

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();

  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    // A forwarded-port tunnel (GitHub Codespaces / VS Code) rejects an
    // oversized body itself and returns an HTML nginx page, not our JSON. The
    // raw HTML is unhelpful in the UI, so translate that specific case.
    const looksLikeProxyPage = /<html|<\/?center>|nginx/i.test(text);
    const isProxy413 = response.status === 413 && (payload === null || looksLikeProxyPage);

    if (isProxy413) {
      const error = new Error(
        "The request body was rejected before reaching the application " +
          "(HTTP 413). This limit comes from the network front-end in front " +
          "of the server, not from the application itself."
      );
      error.status = 413;
      error.fromProxy = true;
      throw error;
    }

    // FastAPI reports errors as {detail: ...}; fall back to the raw body.
    const detail =
      (payload && (payload.detail || payload.message)) ||
      text ||
      `Request failed with status ${response.status}`;
    const error = new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
    error.status = response.status;
    throw error;
  }

  return payload;
}

// --------------------------------------------------------------------------
// Progress
// --------------------------------------------------------------------------

/**
 * Paint the five steps from a backend status string.
 *
 * Stages before ``completed`` are marked done, the stage currently running is
 * marked active. When the job has failed we do not know from the status alone
 * which stage broke, so the step that was last active is marked failed.
 */
function renderProgress(status, note) {
  const failed = status === "failed";
  let completed = STATUS_COMPLETED[status];

  if (failed) {
    // Keep the step that was active as the failure point.
    const activeIndex = el.steps.findIndex((step) =>
      step.classList.contains("is-active")
    );
    completed = activeIndex < 0 ? 0 : activeIndex;
  }
  if (completed === undefined) completed = 0;

  el.steps.forEach((step, index) => {
    step.classList.remove("is-active", "is-done", "is-failed");
    if (failed && index === completed) {
      step.classList.add("is-failed");
    } else if (index < completed) {
      step.classList.add("is-done");
    } else if (!failed && index === completed) {
      step.classList.add("is-active");
    }
  });

  el.progressFill.style.width = `${(completed / STAGES.length) * 100}%`;
  if (note) el.progressNote.textContent = note;
}

function resetProgress() {
  el.steps.forEach((step) => step.classList.remove("is-active", "is-done", "is-failed"));
  el.progressFill.style.width = "0%";
  el.progressNote.textContent = "Waiting to start…";
}

// --------------------------------------------------------------------------
// File selection
// --------------------------------------------------------------------------

function setFile(file) {
  hideError();

  if (!file) {
    state.file = null;
    el.fileLabel.textContent =
      "Drop a Google Meet / Zoom recording here, or click to choose";
    el.dropzone.classList.remove("has-file");
    el.generateBtn.disabled = true;
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    state.file = null;
    el.generateBtn.disabled = true;
    showError(
      `${file.name} is ${formatBytes(file.size)}, which is over the 500 MB limit.`,
      "Trim the recording, or raise MMA_MAX_UPLOAD_SIZE_MB on the server."
    );
    return;
  }

  if (file.size === 0) {
    state.file = null;
    el.generateBtn.disabled = true;
    showError(`${file.name} is empty.`, "Choose a recording that contains audio.");
    return;
  }

  state.file = file;
  el.fileLabel.textContent = `${file.name} · ${formatBytes(file.size)}`;
  el.dropzone.classList.add("has-file");
  el.generateBtn.disabled = false;
}

el.dropzone.addEventListener("click", () => el.fileInput.click());
el.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    el.fileInput.click();
  }
});
el.fileInput.addEventListener("change", () => setFile(el.fileInput.files[0]));

["dragenter", "dragover"].forEach((type) => {
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    el.dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((type) => {
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("is-dragover");
  });
});

el.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) setFile(file);
});

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

function showError(message, hint) {
  el.errorMessage.textContent = message || "Unknown error.";
  el.errorHint.textContent = hint || "";
  show(el.errorHint, Boolean(hint));
  show(el.errorCard, true);
}

function hideError() {
  show(el.errorCard, false);
}

// --------------------------------------------------------------------------
// Results rendering
// --------------------------------------------------------------------------

function renderFacts(extraction, meeting) {
  clearChildren(el.momFacts);

  const facts = [
    ["Date", orUnspecified(extraction.date)],
    ["Time", orUnspecified(extraction.time)],
    [
      "Participants",
      extraction.participants && extraction.participants.length
        ? extraction.participants.join(", ")
        : "Not specified",
    ],
    ["Source file", meeting.original_filename],
    [
      "Duration",
      meeting.duration_seconds ? formatDuration(meeting.duration_seconds) : "Unknown",
    ],
  ];

  facts.forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    el.momFacts.appendChild(dt);
    el.momFacts.appendChild(dd);
  });
}

function renderActionItems(actionItems) {
  clearChildren(el.actionsBody);
  const items = actionItems || [];

  if (items.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.className = "unspecified";
    cell.textContent = "No action items were identified in this meeting.";
    row.appendChild(cell);
    el.actionsBody.appendChild(row);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("tr");

    const task = document.createElement("td");
    task.textContent = item.task || "Not specified";

    const person = document.createElement("td");
    const personUnspecified = isUnspecified(item.responsible_person);
    person.textContent = personUnspecified
      ? "Not specified"
      : item.responsible_person;
    if (personUnspecified) person.className = "unspecified";

    const deadline = document.createElement("td");
    const deadlineUnspecified = isUnspecified(item.deadline);
    deadline.textContent = deadlineUnspecified ? "Not specified" : item.deadline;
    if (deadlineUnspecified) deadline.className = "unspecified";

    row.appendChild(task);
    row.appendChild(person);
    row.appendChild(deadline);
    el.actionsBody.appendChild(row);
  });

  const withoutOwner = items.filter((i) => isUnspecified(i.responsible_person)).length;
  const withoutDeadline = items.filter((i) => isUnspecified(i.deadline)).length;
  if (withoutOwner || withoutDeadline) {
    el.actionsNote.textContent =
      `Of ${items.length} action item(s), ${withoutOwner} have no named owner ` +
      `and ${withoutDeadline} have no stated deadline. The system reports these ` +
      `as "Not specified" rather than inventing a value.`;
    show(el.actionsNote, true);
  } else {
    show(el.actionsNote, false);
  }
}

function renderProvenance(meeting) {
  clearChildren(el.provenanceList);

  const lines = [];
  lines.push(`Speech recognition: ${meeting.model_name || "not recorded"}`);
  if (meeting.extraction_model) {
    lines.push(`Information extraction: ${meeting.extraction_model}`);
  }
  if (meeting.rules_used && meeting.rules_used.length) {
    lines.push(`Deterministic rules supplied: ${meeting.rules_used.join(", ")}`);
  }
  if (meeting.dropped_ungrounded) {
    const dropped = Object.entries(meeting.dropped_ungrounded)
      .filter(([, values]) => values && values.length)
      .map(([field, values]) => `${field} (${values.length})`);
    if (dropped.length) {
      lines.push(`Discarded as not traceable to the transcript: ${dropped.join(", ")}`);
    }
  }
  if (meeting.extraction_seconds != null) {
    lines.push(`Extraction time: ${meeting.extraction_seconds.toFixed(1)}s`);
  }
  if (meeting.generated_at) {
    lines.push(`Generated: ${formatDate(meeting.generated_at)}`);
  }

  lines.forEach((line) => el.provenanceList.appendChild(listItem(line)));
}

function renderResults(minutes, meeting, transcriptInfo) {
  const extraction = minutes.extraction || {};

  el.momTitle.textContent = orUnspecified(extraction.title);
  el.momMeta.textContent = `Meeting #${minutes.meeting_id} · ${
    meeting.original_filename
  }${minutes.generated_at ? ` · generated ${formatDate(minutes.generated_at)}` : ""}`;

  renderFacts(extraction, meeting);

  el.overviewText.textContent = orUnspecified(extraction.overview);
  renderList(el.discussionList, extraction.discussion_points, "No discussion points were identified.");
  renderList(el.decisionsList, extraction.decisions, "No decisions were identified.");
  renderActionItems(extraction.action_items);
  el.conclusionText.textContent = orUnspecified(extraction.conclusion);

  renderProvenance({
    ...transcriptInfo,
    extraction_model: minutes.extraction_model,
    rules_used: minutes.rules_used,
    dropped_ungrounded: minutes.dropped_ungrounded,
    extraction_seconds: minutes.extraction_seconds,
    generated_at: minutes.generated_at,
  });

  state.minutes = minutes;
  show(el.results, true);
}

// --------------------------------------------------------------------------
// The run itself
// --------------------------------------------------------------------------

async function uploadRecording() {
  const form = new FormData();
  form.append("file", state.file, state.file.name);

  // Warn before spending time on a transfer the tunnel will reject anyway.
  // Only applies on a forwarded-port host; on localhost the app's own limit is
  // the only ceiling and is enforced by the server.
  if (isForwardedPortHost() && state.file.size > TUNNEL_BODY_LIMIT_BYTES) {
    throw new UploadTooLargeError(state.file.size);
  }

  // Let the browser set the multipart boundary; do not set Content-Type.
  return requestJSON(`${API}/meetings`, { method: "POST", body: form });
}

/**
 * Poll the meeting until it reaches a terminal state.
 *
 * The status is persisted by the backend, so this reflects real progress. The
 * request that runs the pipeline is launched first and awaited separately, so
 * the UI can update while the server is still working.
 */
async function pollUntilFinished(meetingId, processPromise) {
  const terminal = new Set(["completed", "failed"]);
  let lastStatus = null;
  let finished = false;
  let networkFailures = 0;

  const stop = () => {
    finished = true;
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  };

  // The pipeline request resolves when the whole run is over. Its completion
  // is the authoritative signal, so a failure to poll never loses the result.
  processPromise
    .then(() => stop())
    .catch(() => stop());

  state.pollTimer = setInterval(async () => {
    if (finished) return;
    try {
      const meeting = await requestJSON(`${API}/meetings/${meetingId}`);
      networkFailures = 0;
      if (meeting.status !== lastStatus) {
        lastStatus = meeting.status;
        renderProgress(meeting.status, describeStatus(meeting.status));
        if (meeting.status === "failed" && meeting.error_message) {
          showError(meeting.error_message);
        }
      }
      if (terminal.has(meeting.status)) stop();
    } catch (error) {
      networkFailures += 1;
      // A couple of misses are normal while the server is busy; only give up
      // after repeated failures, and say so rather than hanging silently.
      if (networkFailures >= 5) {
        el.progressNote.textContent =
          "Lost contact with the server while polling; waiting for the run to finish.";
      }
    }
  }, 1200);

  await processPromise;
  stop();
  return lastStatus;
}

function describeStatus(status) {
  switch (status) {
    case "uploading":
      return "Uploading the recording…";
    case "uploaded":
      return "Upload complete.";
    case "extracting_audio":
      return "Extracting audio with ffmpeg…";
    case "audio_extracted":
      return "Audio extracted. Starting speech recognition…";
    case "transcribing":
      return "Transcribing speech with Whisper…";
    case "transcribed":
      return "Transcript ready. Analysing the meeting…";
    case "analysing":
      return "Extracting decisions, action items and participants…";
    case "analysed":
      return "Analysis complete. Generating the minutes…";
    case "generating":
      return "Rendering the Minutes of Meeting…";
    case "completed":
      return "Minutes generated.";
    case "failed":
      return "Processing failed.";
    default:
      return status;
  }
}

async function generate() {
  if (!state.file) return;

  hideError();
  show(el.results, false);
  show(el.progressCard, true);
  resetProgress();
  el.generateBtn.disabled = true;
  el.clearBtn.hidden = true;
  renderProgress("uploading", describeStatus("uploading"));

  try {
    const uploaded = await uploadRecording();
    state.meetingId = uploaded.id;
    renderProgress(uploaded.status, describeStatus(uploaded.status));

    // Kick off the full pipeline, then poll separately for progress.
    const processPromise = requestJSON(
      `${API}/meetings/${uploaded.id}/process`,
      { method: "POST" }
    );

    await pollUntilFinished(uploaded.id, processPromise);

    const result = await processPromise;

    if (!result.succeeded || !result.minutes) {
      const reason = result.error_message || "The pipeline did not produce minutes.";
      renderProgress("failed", "Processing failed.");
      showError(reason, hintForFailure(reason));
      el.generateBtn.disabled = false;
      el.clearBtn.hidden = false;
      return;
    }

    // Fetch the full record for transcript details and provenance in one go.
    let full = null;
    try {
      full = await requestJSON(`${API}/history/${uploaded.id}/full`);
    } catch {
      full = null;
    }

    const transcriptInfo = {
      model_name: full && full.transcript ? full.transcript.model_name : "",
    };
    const minutes = full && full.minutes
      ? { ...result.minutes, ...full.minutes }
      : result.minutes;

    renderProgress("completed", describeStatus("completed"));
    renderResults(minutes, result.meeting, transcriptInfo);
    el.clearBtn.hidden = false;
    loadHistory();
  } catch (error) {
    renderProgress("failed", "Processing failed.");
    showError(error.message, hintForFailure(error.message));
    el.clearBtn.hidden = false;
  } finally {
    el.generateBtn.disabled = !state.file;
  }
}

function hintForFailure(message) {
  const text = String(message || "").toLowerCase();
  if (text.includes("no audio")) {
    return "The recording has no audio track. If this was a screen recording, enable the microphone.";
  }
  if (text.includes("unsupported")) {
    return "Convert the recording to MP4, MKV, WebM, WAV, MP3 or M4A and try again.";
  }
  if (
    text.includes("too large") ||
    text.includes("413") ||
    text.includes("forwarded port")
  ) {
    return (
      "If you are using a Codespaces forwarded port, the tunnel caps the " +
      "request body at about 16 MB regardless of the app's own limit. " +
      "Use a shorter recording, or run the app on a local port and open " +
      "http://localhost:8000 directly."
    );
  }
  if (text.includes("empty") || text.includes("no speech")) {
    return "Whisper found no speech in the audio. Check that the meeting was audible.";
  }
  if (text.includes("ffmpeg")) {
    return "ffmpeg is required to read media files. Install it with: sudo apt-get install -y ffmpeg";
  }
  return "Check the server log for details.";
}

// --------------------------------------------------------------------------
// Transcript and downloads
// --------------------------------------------------------------------------

async function showTranscript() {
  if (!state.meetingId) return;

  el.transcriptText.textContent = "Loading…";
  el.transcriptMeta.textContent = "";
  el.transcriptDialog.showModal();

  try {
    const data = await requestJSON(`${API}/meetings/${state.meetingId}/transcript`);
    el.transcriptText.textContent = data.text || "(empty transcript)";
    const parts = [];
    if (data.model_name) parts.push(`Model: ${data.model_name}`);
    if (data.language) {
      parts.push(
        `Language: ${data.language}` +
          (data.language_probability
            ? ` (${(data.language_probability * 100).toFixed(0)}%)`
            : "")
      );
    }
    if (data.duration_seconds) parts.push(`Audio: ${formatDuration(data.duration_seconds)}`);
    if (data.word_count) parts.push(`${data.word_count} words`);
    if (data.processing_seconds) {
      parts.push(`Transcribed in ${data.processing_seconds.toFixed(1)}s`);
    }
    el.transcriptMeta.textContent = parts.join(" · ");
  } catch (error) {
    el.transcriptText.textContent = `Could not load the transcript: ${error.message}`;
  }
}

function download(kind) {
  if (!state.meetingId) return;
  // Let the browser handle the download so the server's filename is used and
  // the response streams rather than being buffered in memory here.
  window.location.href = `${API}/meetings/${state.meetingId}/export/${kind}`;
}

el.transcriptBtn.addEventListener("click", showTranscript);
el.transcriptClose.addEventListener("click", () => el.transcriptDialog.close());
el.pdfBtn.addEventListener("click", () => download("pdf"));
el.docxBtn.addEventListener("click", () => download("docx"));
el.generateBtn.addEventListener("click", generate);

el.clearBtn.addEventListener("click", () => {
  state.file = null;
  state.meetingId = null;
  state.minutes = null;
  el.fileInput.value = "";
  setFile(null);
  hideError();
  show(el.progressCard, false);
  show(el.results, false);
  el.clearBtn.hidden = true;
  el.generateBtn.disabled = true;
});

// --------------------------------------------------------------------------
// History
// --------------------------------------------------------------------------

function statusClass(status) {
  if (status === "completed") return "status-completed";
  if (status === "failed") return "status-failed";
  return "status-pending";
}

async function loadHistory() {
  try {
    const data = await requestJSON(`${API}/history?limit=25`);
    clearChildren(el.historyList);

    if (!data.items || data.items.length === 0) {
      show(el.historyEmpty, true);
      return;
    }
    show(el.historyEmpty, false);

    data.items.forEach((entry) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item";

      const title = document.createElement("span");
      title.className = "history-title";
      title.textContent = orUnspecified(entry.title);
      button.appendChild(title);

      const sub = document.createElement("span");
      sub.className = "history-sub";
      const bits = [formatDate(entry.created_at)];
      if (entry.duration_seconds) bits.push(formatDuration(entry.duration_seconds));
      if (entry.action_item_count) bits.push(`${entry.action_item_count} actions`);
      sub.textContent = bits.join(" · ");
      button.appendChild(sub);

      const pill = document.createElement("span");
      pill.className = `status-pill ${statusClass(entry.status)}`;
      pill.textContent = entry.status;
      button.appendChild(pill);

      button.addEventListener("click", () => openHistoryEntry(entry.id));
      item.appendChild(button);
      el.historyList.appendChild(item);
    });
  } catch (error) {
    clearChildren(el.historyList);
    el.historyEmpty.textContent = `Could not load history: ${error.message}`;
    show(el.historyEmpty, true);
  }
}

async function openHistoryEntry(meetingId) {
  hideError();
  try {
    const full = await requestJSON(`${API}/history/${meetingId}/full`);

    if (!full.minutes || !full.extraction) {
      showError(
        `Meeting ${meetingId} has no generated minutes.`,
        full.meeting && full.meeting.error_message
          ? full.meeting.error_message
          : "It may have failed during processing."
      );
      return;
    }

    state.meetingId = meetingId;
    // Any previously selected file is irrelevant once we are viewing history.
    state.file = null;
    el.fileInput.value = "";
    setFile(null);
    el.clearBtn.hidden = false;

    const minutes = {
      meeting_id: meetingId,
      title: full.extraction.title,
      extraction: full.extraction,
      has_pdf: full.minutes.has_pdf,
      has_docx: full.minutes.has_docx,
      generated_at: full.meeting.updated_at,
      extraction_model: full.minutes.extraction_model,
      rules_used: full.minutes.rules_used,
      dropped_ungrounded: full.minutes.dropped_ungrounded,
      extraction_seconds: full.minutes.extraction_seconds,
    };

    show(el.progressCard, false);
    renderResults(minutes, full.meeting, {
      model_name: full.transcript ? full.transcript.model_name : "",
    });
    el.results.scrollIntoView({ block: "start" });
  } catch (error) {
    showError(`Could not open meeting ${meetingId}: ${error.message}`);
  }
}

el.historyRefresh.addEventListener("click", loadHistory);

// --------------------------------------------------------------------------
// Capability banner
// --------------------------------------------------------------------------

async function loadCapabilities() {
  try {
    const health = await requestJSON(`${API}/health`);
    const config = health.config || {};
    const bits = [];
    if (config.whisper_model_size) bits.push(`Whisper ${config.whisper_model_size}`);
    if (config.extraction_model_name) {
      // Show only the final path segment; the full repo id is long and noisy.
      bits.push(config.extraction_model_name.split("/").pop());
    }
    el.modelBadge.textContent = bits.length ? bits.join(" · ") : "models ready";
    el.modelBadge.title = JSON.stringify(health, null, 2);
  } catch {
    el.modelBadge.textContent = "server unreachable";
  }
}

// --------------------------------------------------------------------------
// Start-up
// --------------------------------------------------------------------------

setFile(null);
resetProgress();
loadCapabilities();
loadHistory();
