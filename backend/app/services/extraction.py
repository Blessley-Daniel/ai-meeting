"""The project's own approach to meeting information extraction.

This module is the part of the system we designed, as opposed to the parts we
borrowed off the shelf:

* **pretrained** - a small instruction-tuned language model (Qwen2.5-0.5B-Instruct
  by default) provides general language understanding.
* **ours** - the prompt, the JSON-repair logic, the grounding check that
  rejects invented details, and the domain rules that decide what counts as a
  decision versus a discussion point.

The design problem
------------------

Asking a small language model "please return JSON" is unreliable: it may wrap
the JSON in prose or a ```json fence, emit trailing commas or Python literals,
truncate mid-object, or - worst of all - invent a plausible-looking fact that
appears nowhere in the transcript. Each failure mode is handled separately
here, because a single "call the model and hope" approach degrades badly.

The grounding check is the important one. A language model asked to fill a
schema will happily produce a deadline for an action item that has none. We
therefore verify that proper nouns appearing in extracted fields actually occur
(somewhere, allowing for speech-recognition error) in the transcript. Anything
that fails the check is reset to ``Not specified`` rather than reported. This
trades a little recall for far fewer fabrications, which is the right trade for
minutes of meeting: a missing deadline is a nuisance, an invented one is a
false record of what a team agreed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.extraction import (
    NOT_SPECIFIED,
    SCHEMA_SPEC,
    MeetingExtraction,
)
from app.services.rules import extract_with_rules
from app.utils.errors import ExtractionError

logger = logging.getLogger(__name__)

# Loading a model is expensive; keep one per configuration.
_MODEL_CACHE: dict[str, tuple[object, object]] = {}
_MODEL_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise meeting-minutes assistant. You read a meeting "
    "transcript and return structured data as JSON.\n"
    "Rules you must follow:\n"
    "1. Return ONLY a JSON object. No explanation, no markdown, no code fences.\n"
    "2. Use only information stated in the transcript. Never infer or invent "
    "names, dates, deadlines or decisions.\n"
    "3. If a value is not stated in the transcript, use the exact string "
    '"Not specified" (for action items, for the responsible person and the '
    "deadline alike).\n"
    "4. A 'decision' is something the group explicitly agreed, typically "
    "introduced by phrases such as 'we decided', 'let us use', 'we agreed'. "
    "A 'discussion point' is merely a topic that was talked about. Do not "
    "record a topic as a decision unless agreement was explicit.\n"
    "5. An 'action item' is a task someone agreed to do. The responsible "
    "person must be stated or clearly implied in the transcript.\n"
    "6. Keep each list entry short: one task, topic or decision per entry.\n"
    "7. Phrases such as 'next Monday' or 'on Wednesday' are acceptable "
    "deadlines exactly as spoken; do not convert them to calendar dates.\n"
)


def build_prompt(transcript: str, settings: Settings | None = None) -> str:
    """Build the user turn: instructions, schema, then the transcript.

    The transcript is truncated to the configured token budget. Truncation is
    applied to the *end*, because meeting minutes are usually summed up near
    the end and the opening contains introductions - but this is a real
    limitation for long meetings, and it is recorded in the result metadata.
    """
    settings = settings or get_settings()
    max_chars = settings.extraction_max_input_tokens * 4  # ~4 chars per token

    text = transcript.strip()
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    note = ""
    if truncated:
        note = (
            "\n(Note: the transcript was truncated because it was too long; "
            "extract what you can from the text provided.)\n"
        )

    return (
        f"{note}"
        "Extract the meeting information from the transcript below.\n\n"
        f"Return JSON with exactly this structure:\n{SCHEMA_SPEC}\n\n"
        "TRANSCRIPT:\n"
        f'"""\n{text}\n"""\n\n'
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(settings: Settings | None = None):
    """Return a cached ``(tokenizer, model)`` pair for the configured model.

    If a LoRA adapter produced by the fine-tuning stage (step 14) exists on
    disk, it is attached to the base model. That is how the optional
    fine-tuning path feeds into the running system without replacing the base
    weights.

    Raises:
        ExtractionError: transformers is missing or the weights cannot load
            (usually no network access on the very first run).
    """
    settings = settings or get_settings()
    adapter = settings.extraction_adapter_path
    key = f"{settings.extraction_model_name}|{adapter or ''}"
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ExtractionError(
                "transformers is not installed. "
                "Install it with: pip install -r requirements-ai.txt"
            ) from exc

        logger.info("Loading extraction model %s", settings.extraction_model_name)
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.extraction_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                settings.extraction_model_name,
                dtype="auto",
                low_cpu_mem_usage=True,
            )
            if adapter:
                model = _attach_adapter(model, adapter)
        except ExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise ExtractionError(
                f"Could not load the extraction model "
                f"{settings.extraction_model_name!r}: {exc}"
            ) from exc

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()

        logger.info("Extraction model ready in %.1fs", time.perf_counter() - started)
        _MODEL_CACHE[key] = (tokenizer, model)
        return tokenizer, model


def _attach_adapter(model, adapter_path: str):
    """Load a LoRA/PEFT adapter onto a base model, if it is present.

    A missing adapter is not an error: the fine-tuning step is optional, and
    the system is expected to work with the pretrained base model alone.
    """
    from pathlib import Path as _Path

    if not _Path(adapter_path).exists():
        logger.warning(
            "Configured adapter %s does not exist; using the base model.", adapter_path
        )
        return model
    try:
        from peft import PeftModel
    except ImportError:  # pragma: no cover - depends on install
        logger.warning("peft is not installed; ignoring the adapter at %s", adapter_path)
        return model

    logger.info("Attaching LoRA adapter from %s", adapter_path)
    return PeftModel.from_pretrained(model, adapter_path)


def _generate(prompt: str, settings: Settings) -> str:
    """Run the model and return its raw text output."""
    import torch

    tokenizer, model = load_model(settings)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=settings.extraction_max_input_tokens + 512)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=settings.extraction_max_new_tokens,
            do_sample=False,          # greedy: deterministic and reproducible
            # The model's stored generation config sets sampling parameters.
            # With do_sample=False they are unused, and transformers warns
            # about them on every call; None marks them as deliberately unset.
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str) -> dict:
    """Recover a JSON object from whatever the model produced.

    Handles, in order: markdown code fences, prose before/after the object,
    single quotes, trailing commas, and truncation. Raises if nothing usable
    can be recovered.
    """
    if not raw or not raw.strip():
        raise ExtractionError("The extraction model returned an empty response.")

    text = raw.strip()

    # 1. Unwrap a ```json fence if present.
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # 2. Take the substring from the first '{' onward. We do not naively take
    #    the *last* '}' because a truncated response can end on an inner brace,
    #    e.g. '{"a": {"b": 2}' - the outer object is still open there.
    start = text.find("{")
    if start == -1:
        raise ExtractionError("The extraction model did not return a JSON object.")
    text = text[start:]

    # 3. Progressively more aggressive repairs.
    candidates = [text]
    balanced = _close_unbalanced(text)
    if balanced != text:
        candidates.append(balanced)
    for candidate in list(candidates):
        candidates.append(re.sub(r",\s*([}\]])", r"\1", candidate))       # trailing commas
        candidates.append(
            re.sub(r",\s*([}\]])", r"\1", candidate.replace("'", '"'))    # single quotes
        )

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # 4. Last resort: pull out balanced top-level objects one by one.
    for candidate in candidates:
        for obj in _iter_objects(candidate):
            try:
                parsed = json.loads(obj)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ExtractionError(
        "Could not parse JSON from the extraction model's output. "
        f"First 200 characters were: {raw[:200]!r}"
    )


def _close_unbalanced(text: str) -> str:
    """Append the closing brackets needed to balance a truncated JSON object.

    Required because generation can hit ``max_new_tokens`` mid-object, leaving
    a response that is almost valid and only needs closing. Tracks both braces
    and brackets, since a truncated response often ends inside ``action_items``.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        text += '"'
    # Remove a dangling separator before closing, e.g. '{"a": 1,'
    text = re.sub(r",\s*$", "", text.rstrip())
    return text + "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _iter_objects(text: str):
    """Yield substrings that look like balanced JSON objects."""
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


# ---------------------------------------------------------------------------
# Grounding check: the anti-hallucination guard
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "by", "at", "is", "are", "was", "were", "be", "been", "will", "would",
    "should", "can", "could", "may", "might", "must", "that", "this", "these",
    "those", "it", "its", "as", "from", "we", "our", "us", "i", "you", "they",
    "them", "he", "she", "his", "her", "not", "no", "yes", "ok", "okay", "so",
    "then", "there", "here", "when", "where", "what", "who", "how", "why",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] via a simple ratio.

    Used instead of exact matching so that a name the ASR mangled slightly
    (``Priya`` -> ``prior``) is not automatically discarded as a fabrication.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _is_grounded(value: str, transcript_lower: str, transcript_tokens: set[str],
                 *, min_similarity: float = 0.80) -> bool:
    """Check whether ``value`` plausibly comes from the transcript.

    A value counts as grounded if it appears verbatim, if most of its
    distinctive words appear, or if it is close to something that does appear
    (tolerating minor ASR corruption).

    Note the deliberate asymmetry for short values: a single word such as
    "Friday" is *not* automatically accepted. If the transcript never mentions
    Friday, a deadline of "Friday" is exactly the fabrication we are trying to
    prevent, so it must fail this check.
    """
    value = value.strip()
    if not value or value == NOT_SPECIFIED:
        return True

    lowered = value.lower()
    if lowered in transcript_lower:
        return True

    words = [w for w in re.findall(r"[a-z0-9']+", lowered) if len(w) > 2]
    words = [w for w in words if w not in _STOPWORDS]
    if not words:
        return True

    # Fraction of distinctive words that appear in the transcript.
    exact = sum(1 for w in words if w in transcript_tokens)
    if exact / len(words) >= 0.6:
        return True

    # Tolerate ASR corruption: is any word close to a transcript word?
    for word in words:
        if any(_similarity(word, candidate) >= min_similarity
               for candidate in transcript_tokens):
            return True

    return False


@dataclass
class ExtractionDiagnostics:
    """What happened during extraction, for debugging and reporting."""

    model_name: str = ""
    prompt_chars: int = 0
    truncated: bool = False
    raw_output_chars: int = 0
    json_repairs_used: bool = False
    processing_seconds: float = 0.0
    # Which fields the deterministic rules supplied (rather than the model).
    rule_sources: list[str] = field(default_factory=list)
    # Field name -> values dropped by the grounding check.
    dropped_ungrounded: dict[str, list[str]] = field(default_factory=dict)

    @property
    def dropped_count(self) -> int:
        return sum(len(v) for v in self.dropped_ungrounded.values())


def _apply_grounding(
    extraction: MeetingExtraction,
    transcript: str,
    diagnostics: ExtractionDiagnostics,
) -> MeetingExtraction:
    """Reset any field that cannot be traced back to the transcript.

    This is the mechanism that fulfils the "do not hallucinate" requirement.
    It is applied to free-text claims - decisions, action-item tasks and the
    conclusion - where a model is most likely to invent something. List fields
    such as participants and discussion points are checked with the same
    predicate, but a failure there is recorded rather than raised, because a
    missing discussion point is a much smaller error than a fabricated
    decision.
    """
    transcript_lower = transcript.lower()
    transcript_tokens = _tokens(transcript)

    def grounded(value: str) -> bool:
        return _is_grounded(value, transcript_lower, transcript_tokens)

    for label, items in (
        ("decisions", extraction.decisions),
        ("discussion_points", extraction.discussion_points),
        ("participants", extraction.participants),
    ):
        kept = []
        for item in items:
            if grounded(item):
                kept.append(item)
            else:
                diagnostics.dropped_ungrounded.setdefault(label, []).append(item)
        setattr(extraction, label, kept)

    kept_actions = []
    for action in extraction.action_items:
        if not grounded(action.task):
            diagnostics.dropped_ungrounded.setdefault("action_items", []).append(
                action.task
            )
            continue
        if action.responsible_person != NOT_SPECIFIED and not grounded(
            action.responsible_person
        ):
            diagnostics.dropped_ungrounded.setdefault(
                "responsible_person", []
            ).append(action.responsible_person)
            action = action.model_copy(
                update={"responsible_person": NOT_SPECIFIED}
            )
        if action.deadline != NOT_SPECIFIED and not grounded(action.deadline):
            diagnostics.dropped_ungrounded.setdefault("deadline", []).append(
                action.deadline
            )
            action = action.model_copy(update={"deadline": NOT_SPECIFIED})
        kept_actions.append(action)
    extraction.action_items = kept_actions

    if extraction.conclusion != NOT_SPECIFIED and not grounded(extraction.conclusion):
        diagnostics.dropped_ungrounded.setdefault("conclusion", []).append(
            extraction.conclusion
        )
        extraction = extraction.model_copy(update={"conclusion": NOT_SPECIFIED})

    return extraction


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _clean_discussion_points(points: list[str], conclusion: str) -> list[str]:
    """Remove entries that are not really discussion topics.

    Small models tend to append their own meta-commentary and to repeat the
    conclusion inside the discussion list (observed: entries prefixed
    "Conclusion:" and "Final Outcome:"). Those belong in ``conclusion``, so
    they are dropped here rather than shown as topics.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for point in points:
        text = point.strip()
        lowered = text.lower()

        # Model meta-commentary about the transcript, not meeting content.
        if re.match(r"^(conclusion|final outcome|summary|overview|action items?)\s*[:\-]", lowered):
            continue
        # Note: the remaining "no deadline was mentioned" style entries are
        # still informational, so they are kept.

        # Avoid duplicating the conclusion as a discussion point.
        if conclusion != NOT_SPECIFIED and lowered in conclusion.lower():
            continue

        if len(text.split()) < 3:
            continue
        key = lowered
        if key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned


def _strip_deadline_from_task(task: str, deadline: str) -> str:
    """Remove a trailing deadline phrase from a task description.

    The pattern extractor captures "update the docs by next Monday" as the
    task; with the deadline also recorded separately, the minutes would repeat
    it. Trimming keeps the table readable.
    """
    if not deadline or deadline == NOT_SPECIFIED:
        return task
    trimmed = re.sub(
        rf"\s+(?:by|before|on|due)\s+{re.escape(deadline)}\s*$",
        "",
        task,
        flags=re.IGNORECASE,
    )
    trimmed = trimmed.strip(" .,;:")
    return trimmed if len(trimmed.split()) >= 3 else task


def extract_meeting_information(
    transcript: str,
    settings: Settings | None = None,
) -> tuple[MeetingExtraction, ExtractionDiagnostics]:
    """Extract structured meeting information from a transcript.

    A hybrid of two sources:

    * the **language model**, which paraphrases well and judges the overall
      gist, filling ``title``, ``overview`` and ``discussion_points``;
    * the **rule-based extractors** in :mod:`app.services.rules`, which are
      more reliable for rigidly defined facts such as decisions and action
      items.

    The two are merged, then every surviving value passes the grounding check.
    When the rules find nothing and the model produced something, the model's
    output is retained, so behaviour degrades gracefully rather than failing.

    Returns the extraction together with diagnostics describing how it was
    produced and what was rejected.

    Raises:
        ExtractionError: the transcript was empty, or the model failed badly
            enough that no usable object could be recovered.
    """
    settings = settings or get_settings()

    if not transcript or not transcript.strip():
        raise ExtractionError("Cannot extract information from an empty transcript.")

    prompt = build_prompt(transcript, settings)
    diagnostics = ExtractionDiagnostics(
        model_name=settings.extraction_model_name,
        prompt_chars=len(prompt),
        truncated=len(transcript.strip()) > settings.extraction_max_input_tokens * 4,
    )

    started = time.perf_counter()
    raw = _generate(prompt, settings)
    diagnostics.processing_seconds = time.perf_counter() - started
    diagnostics.raw_output_chars = len(raw)

    payload = extract_json_object(raw)
    diagnostics.json_repairs_used = (
        raw.strip().startswith("```") or not raw.strip().startswith("{")
    )

    extraction = MeetingExtraction.model_validate(payload)

    # --- merge in the deterministic rules -------------------------------
    # Rules take precedence for the fields they are trusted with. Empty rule
    # results leave the model's answer in place.
    rules = extract_with_rules(transcript)

    if rules.decisions:
        extraction.decisions = rules.decisions
        diagnostics.rule_sources.append("decisions")
    if rules.action_items:
        extraction.action_items = [
            item.model_copy(
                update={"task": _strip_deadline_from_task(item.task, item.deadline)}
            )
            for item in rules.action_items
        ]
        diagnostics.rule_sources.append("action_items")
    if rules.participants:
        extraction.participants = rules.participants
        diagnostics.rule_sources.append("participants")
    if rules.conclusion != NOT_SPECIFIED:
        extraction.conclusion = rules.conclusion
        diagnostics.rule_sources.append("conclusion")

    # --- anti-hallucination check ---------------------------------------
    extraction = _apply_grounding(extraction, transcript, diagnostics)

    # --- final tidying ---------------------------------------------------
    extraction.discussion_points = _clean_discussion_points(
        extraction.discussion_points, extraction.conclusion
    )

    logger.info(
        "Extracted from transcript in %.1fs: %d discussion points, %d decisions, "
        "%d action items, %d ungrounded values dropped (rules: %s)",
        diagnostics.processing_seconds,
        len(extraction.discussion_points),
        len(extraction.decisions),
        len(extraction.action_items),
        diagnostics.dropped_count,
        ",".join(diagnostics.rule_sources) or "none",
    )
    return extraction, diagnostics


def save_extraction(extraction: MeetingExtraction, path: Path) -> None:
    """Persist an extraction result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(extraction.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_extraction(path: Path) -> MeetingExtraction:
    """Load a previously saved extraction result."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MeetingExtraction.model_validate(payload)