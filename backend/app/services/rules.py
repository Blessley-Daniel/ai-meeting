"""Deterministic extraction rules: our own domain logic over the transcript.

Why this module exists
----------------------

Measured behaviour of the pretrained 0.5B model on our sample meeting
(see ``docs/benchmarks.md``) showed three concrete failures:

1. It invented participants ("Samantha", "John") who are never mentioned.
2. It missed explicit decisions ("the team decided to use PostgreSQL").
3. It filed conclusion-like text under discussion points.

A language model is good at paraphrase and at judging what a passage is
*about*; it is unreliable at remembering rigid definitions. Decisions and
action items, however, are announced with a small set of fairly reliable
linguistic patterns. So rather than trusting the model for everything, this
module detects those patterns directly.

This is a hybrid design: the model supplies the overview and the gist of the
discussion, and these rules supply the high-precision structured facts. Every
value either source produces still passes through the grounding check in
``services.extraction``.

The rules are intentionally conservative. A false positive here means minutes
recording a decision that was never made, which is worse than omitting one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.extraction import NOT_SPECIFIED, ActionItem

# ---------------------------------------------------------------------------
# Decision detection
# ---------------------------------------------------------------------------

# Explicit agreement markers. Each pattern captures the agreed content in
# group "content" where the phrase is followed by a clause.
_DECISION_PATTERNS: list[re.Pattern[str]] = [
    # "the team decided to use PostgreSQL" / "we decided that X"
    re.compile(
        r"\b(?:we|they|the team|everyone|it)\s+(?:have\s+|has\s+)?decided\s+"
        r"(?:to|that|on)?\s*(?P<content>[^.!?;]{3,200})",
        re.IGNORECASE,
    ),
    # "we agreed to use X" / "agreed on X" / "agreed that X"
    re.compile(
        r"\bagreed\s+(?:to|on|that)\s+(?P<content>[^.!?;]{3,200})",
        re.IGNORECASE,
    ),
    # "let's use Python" / "let us proceed with X".
    # Restricted to verbs of *choice or adoption*. A bare "let's start the
    # meeting" is an agenda action, not a decision, and including it produced a
    # false positive in testing.
    re.compile(
        r"\blet(?:'s| us)\s+(?:use|adopt|go\s+with|proceed\s+with|choose|"
        r"stick\s+with|implement|switch\s+to|standardise\s+on|standardize\s+on)\s+"
        r"(?P<content>[^.!?;]{3,200})",
        re.IGNORECASE,
    ),
    # "we will / we'll go with X"
    re.compile(
        r"\bwe(?:'ll| will)\s+go\s+with\s+(?P<content>[^.!?;]{3,200})",
        re.IGNORECASE,
    ),
    # "the decision is to X" / "our decision is X"
    re.compile(
        r"\b(?:the|our)\s+decision\s+(?:is|was)\s+(?:to\s+)?(?P<content>[^.!?;]{3,200})",
        re.IGNORECASE,
    ),
]

# Lead-ins that mark the *end* of an agreement clause; anything after them
# belongs to the next sentence and must be trimmed off.
_DECISION_TAIL = re.compile(
    r"\s+(?:also|but|however|and then|then we|so we|agreed\b|okay\b|ok\b).*$",
    re.IGNORECASE,
)


def _tidy(clause: str) -> str:
    """Clean a captured clause into a readable list entry."""
    text = clause.strip().strip(",").strip()
    text = re.sub(r"\s+", " ", text)
    text = _DECISION_TAIL.sub("", text)
    # Drop leading filler left over from the capture, so a task reads as the
    # task ("prepare the checklist") rather than a fragment of speech.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^(?:that|to|also|then|just|now|please|first|next)\s+",
                      "", text, flags=re.IGNORECASE)
    return text.strip(" .,;:")


def find_decisions(text: str) -> list[str]:
    """Return clauses the speakers explicitly agreed on."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _DECISION_PATTERNS:
        for match in pattern.finditer(text):
            content = _tidy(match.group("content"))
            if len(content.split()) < 2:
                continue
            key = content.lower()
            if key not in seen:
                seen.add(key)
                found.append(content)
    return found


# ---------------------------------------------------------------------------
# Action-item detection
# ---------------------------------------------------------------------------

# Names are approximated as a capitalised token that is not a common word at
# the start of a sentence. This is crude, and it is a designed limitation:
# Whisper's capitalisation is not perfectly reliable either.
_SENTENCE_STARTERS = {
    "the", "we", "i", "he", "she", "they", "it", "this", "that", "there",
    "then", "so", "also", "and", "but", "good", "great", "okay", "ok", "yes",
    "no", "agreed", "first", "next", "finally", "let", "let's", "to", "in",
    "on", "for", "with", "as", "if", "when", "after", "before", "well",
}

_NAME = r"[A-Z][a-z]{1,20}"

_ACTION_PATTERNS: list[re.Pattern[str]] = [
    # "Rahul will update the schema documentation by next Monday"
    re.compile(
        rf"\b(?P<person>{_NAME})\s+will\s+(?P<task>[a-z][^.!?;]{{3,200}})",
    ),
    # "Priya is going to run the tests"
    re.compile(
        rf"\b(?P<person>{_NAME})\s+is\s+going\s+to\s+(?P<task>[a-z][^.!?;]{{3,200}})",
    ),
    # "Rahul agreed to update the docs" / "Priya agreed to test it"
    re.compile(
        rf"\b(?P<person>{_NAME})\s+agreed\s+to\s+(?P<task>[a-z][^.!?;]{{3,200}})",
    ),
    # First person: "I will prepare the deployment checklist"
    re.compile(
        r"\bI(?:'ll| will)\s+(?P<task>[a-z][^.!?;]{3,200})",
    ),
    # "Priya, can you send the report"  (requests carry an implied owner)
    re.compile(
        rf"\b(?P<person>{_NAME}),\s*(?:can|could|would)\s+you\s+"
        r"(?P<task>[a-z][^.!?;]{3,200})",
    ),
]

# Phrases that signal no deadline was given. Checked before deadline parsing so
# that "no deadlines are needed for that one yet" is not read as a deadline.
_NO_DEADLINE = re.compile(
    r"\b(?:no\s+deadlines?|not?\s+(?:yet|needed)|whenever|no\s+rush|"
    r"no\s+date\s+(?:set|yet))\b",
    re.IGNORECASE,
)

# Deadline expressions. Relative phrases are kept verbatim, as spoken: the
# prompt forbids inventing calendar dates, and resolving "next Monday" would
# require knowing the meeting date, which is usually absent.
_DEADLINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bby\s+(?:the\s+)?(?P<deadline>[^.!?;]{2,60})", re.IGNORECASE),
    re.compile(r"\bbefore\s+(?:the\s+)?(?P<deadline>[^.!?;]{2,60})", re.IGNORECASE),
    re.compile(r"\b(?:due|deadline(?:\s+is)?)\s+(?:on\s+|by\s+)?(?P<deadline>[^.!?;]{2,60})", re.IGNORECASE),
    re.compile(r"\b(?P<deadline>next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))", re.IGNORECASE),
    re.compile(r"\bon\s+(?P<deadline>(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))", re.IGNORECASE),
    re.compile(r"\b(?P<deadline>(?:this|tomorrow|today)\b[^.!?;]{0,40})", re.IGNORECASE),
    re.compile(r"\b(?P<deadline>end\s+of\s+(?:the\s+)?(?:week|month|day))", re.IGNORECASE),
]

_DEADLINE_TAIL = re.compile(
    r"\s+(?:that|which|and|but|so|we|i|he|she|they|to)\b.*$", re.IGNORECASE
)


def find_deadline(text: str) -> str:
    """Find a deadline expression near ``text``, or ``NOT_SPECIFIED``."""
    if _NO_DEADLINE.search(text):
        return NOT_SPECIFIED
    for pattern in _DEADLINE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group("deadline").strip()
        value = _DEADLINE_TAIL.sub("", value).strip(" .,;:()")
        # Guard against nonsense captures such as "by the team".
        if len(value.split()) > 6:
            value = " ".join(value.split()[:6])
        if len(value) >= 3 and not value.lower().startswith(("the ", "a ", "an ")):
            return value
    return NOT_SPECIFIED


def _looks_like_name(token: str) -> bool:
    return bool(re.fullmatch(_NAME, token)) and token.lower() not in _SENTENCE_STARTERS


def find_action_items(text: str, known_people: list[str] | None = None) -> list[ActionItem]:
    """Detect commitments and turn them into structured action items.

    Args:
        text: the transcript.
        known_people: names already identified as participants. Used to
            resolve first-person commitments ("I will ...") to a speaker when
            exactly one candidate is plausible.
    """
    items: list[ActionItem] = []
    seen: set[str] = set()

    for pattern in _ACTION_PATTERNS:
        for match in pattern.finditer(text):
            spans = match.groupdict()
            task = spans.get("task") or ""
            person = spans.get("person") or NOT_SPECIFIED
            task = _tidy(task)
            if len(task.split()) < 3:
                continue

            # Look for a deadline in the same sentence as the commitment.
            start = text.rfind(".", 0, match.start())
            end = text.find(".", match.end())
            sentence = text[start + 1 : end if end != -1 else len(text)]
            deadline = find_deadline(sentence)

            # "No deadlines are needed for that one yet" -> not a commitment.
            if _NO_DEADLINE.search(sentence) and "deadline" in task.lower():
                continue

            key = task.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(
                ActionItem(
                    task=task,
                    responsible_person=person,
                    deadline=deadline,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Participants and conclusion
# ---------------------------------------------------------------------------

# A person's name typically appears as a capitalised token that is not a
# sentence starter and is near a speech verb or a vocative.
_SPEECH_CUE = re.compile(
    r"(?:great work|thanks|thank you|well done|over to|as|per)\s+(?P<name>[A-Z][a-z]{1,20})"
)
_VOCATIVE = re.compile(r"(?:^|[,.]\s+)(?P<name>[A-Z][a-z]{1,20})\s*[,]")


def find_participants(text: str, action_people: list[str] | None = None) -> list[str]:
    """Collect likely participant names from the transcript.

    Deliberately conservative: a name is only counted when it is addressed
    directly, praised by name, or credited with an action. Merely being a
    capitalised word is not enough, because Whisper capitalises inconsistently
    and that would produce false participants - the exact failure seen in the
    language-model output.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip(" .,;:")
        if not _looks_like_name(candidate):
            return
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            names.append(candidate)

    for match in _SPEECH_CUE.finditer(text):
        add(match.group("name"))
    for match in _VOCATIVE.finditer(text):
        add(match.group("name"))
    for person in action_people or []:
        if person and person != NOT_SPECIFIED:
            add(person)

    return names


_CONCLUSION_CUES = re.compile(
    r"\b(?:to conclude|in conclusion|so,? to (?:conclude|summarise|summarize)|"
    r"to summarise|to summarize|in summary|to wrap up|to sum up)\b[,:]?\s*",
    re.IGNORECASE,
)


def find_conclusion(text: str) -> str:
    """Find the final summary, usually introduced by an explicit cue."""
    matches = list(_CONCLUSION_CUES.finditer(text))
    if matches:
        tail = text[matches[-1].end() :].strip()
        # Take up to the end of the sentence, allowing a couple of clauses.
        tail = re.split(r"(?<=[.!?])\s", tail)
        candidate = " ".join(tail[:2]).strip()
        if len(candidate.split()) >= 4:
            return _tidy(candidate)
    # Fall back to the final sentence, which usually carries the wrap-up.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if sentences:
        last = sentences[-1]
        if len(last.split()) >= 5:
            return last
    return NOT_SPECIFIED


@dataclass
class RuleExtraction:
    """Everything the deterministic rules found."""

    decisions: list[str]
    action_items: list[ActionItem]
    participants: list[str]
    conclusion: str

    @property
    def count(self) -> int:
        return (
            len(self.decisions)
            + len(self.action_items)
            + len(self.participants)
            + (1 if self.conclusion != NOT_SPECIFIED else 0)
        )


def extract_with_rules(text: str) -> RuleExtraction:
    """Run every rule-based extractor over a transcript."""
    actions = find_action_items(text)
    people = find_participants(
        text, [a.responsible_person for a in actions if a.responsible_person != NOT_SPECIFIED]
    )
    return RuleExtraction(
        decisions=find_decisions(text),
        action_items=actions,
        participants=people,
        conclusion=find_conclusion(text),
    )