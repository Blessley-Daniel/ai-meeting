"""Structured schema for meeting information extraction.

This module is the single source of truth for *what* we try to extract from a
transcript. Everything else - the prompt shown to the language model, the
validators that check its response, and the renderers that produce the final
document - is derived from these classes.

The sentinel defaults matter. They are what stop the system fabricating a
deadline or a name that was never mentioned: any field the model cannot ground
in the transcript must come back as :data:`NOT_SPECIFIED` rather than a
plausible guess.

The schema is deliberately Pydantic rather than free-form JSON. Validation is
then a property of the type, not a separate code path that can drift out of
sync with the prompt.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The exact string used whenever a field is genuinely absent from the meeting.
# Kept as one constant so prompt, validator and renderer cannot disagree.
NOT_SPECIFIED = "Not specified"

# Sentinel values a language model may emit to mean "nothing here". Treated as
# equivalent to NOT_SPECIFIED during validation so the model's phrasing does
# not leak into the output.
_NULL_LIKE = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "unknown",
    "not mentioned",
    "not specified",
    "not available",
    "no deadline",
    "no deadline specified",
    "no date",
    "unspecified",
    "tbd",
    "to be determined",
    "not applicable",
}


def _clean_text(value: object) -> str:
    """Normalise a scalar the model produced, mapping null-likes to the sentinel."""
    if value is None:
        return NOT_SPECIFIED
    text = str(value).strip()
    if text.lower().rstrip(".") in _NULL_LIKE:
        return NOT_SPECIFIED
    return text


def _clean_list(value: object) -> list[str]:
    """Normalise a list of strings: drop entries that are empty or null-like."""
    if value is None:
        return []
    if isinstance(value, str):
        # A model asked for a list sometimes returns one delimited string.
        value = re.split(r"\n|;", value)
    if not isinstance(value, (list, tuple, set)):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            # e.g. {"point": "..."} - take the first string value.
            item = next((v for v in item.values() if isinstance(v, str)), "")
        text = _clean_text(item)
        if text == NOT_SPECIFIED:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


class ActionItem(BaseModel):
    """A single agreed task, with its owner and deadline when stated."""

    model_config = ConfigDict(extra="ignore")

    task: str = ""
    responsible_person: str = NOT_SPECIFIED
    deadline: str = NOT_SPECIFIED

    @field_validator("task", "responsible_person", "deadline", mode="before")
    @classmethod
    def _normalise(cls, value: object) -> str:
        return _clean_text(value)

    @property
    def is_actionable(self) -> bool:
        """True if there is a real task to report (not an empty placeholder)."""
        return bool(self.task) and self.task != NOT_SPECIFIED


class MeetingExtraction(BaseModel):
    """Everything the system extracts from one meeting transcript.

    Every field has a safe default, so a partially malformed model response
    still produces a usable object rather than raising. Missing information is
    represented explicitly by ``NOT_SPECIFIED`` (or an empty list), never by an
    invented value.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = NOT_SPECIFIED
    date: str = NOT_SPECIFIED
    time: str = NOT_SPECIFIED
    participants: list[str] = Field(default_factory=list)
    overview: str = NOT_SPECIFIED

    discussion_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    conclusion: str = NOT_SPECIFIED

    @field_validator("title", "date", "time", "overview", "conclusion", mode="before")
    @classmethod
    def _normalise_scalar(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("participants", "discussion_points", "decisions", mode="before")
    @classmethod
    def _normalise_lists(cls, value: object) -> list[str]:
        return _clean_list(value)

    @field_validator("action_items", mode="before")
    @classmethod
    def _normalise_action_items(cls, value: object) -> list[ActionItem]:
        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []

        items: list[ActionItem] = []
        seen: set[str] = set()
        for raw in value:
            if isinstance(raw, str):
                raw = {"task": raw}
            if not isinstance(raw, dict):
                continue
            # Tolerate common key aliases from small models.
            normalised = {
                "task": raw.get("task") or raw.get("action") or raw.get("description") or "",
                "responsible_person": raw.get("responsible_person")
                or raw.get("owner")
                or raw.get("assignee")
                or raw.get("person")
                or NOT_SPECIFIED,
                "deadline": raw.get("deadline") or raw.get("due") or raw.get("due_date") or NOT_SPECIFIED,
            }
            try:
                item = ActionItem(**normalised)
            except Exception:  # noqa: BLE001 - skip malformed entries
                continue
            if not item.is_actionable:
                continue
            key = item.task.lower()
            if key not in seen:
                seen.add(key)
                items.append(item)
        return items

    # ---------------------------------------------------------------- helpers
    def to_json_dict(self) -> dict:
        """Plain-dict form suitable for JSON serialisation."""
        return self.model_dump()

    def is_empty(self) -> bool:
        """True if nothing at all was extracted."""
        return not (
            self.discussion_points
            or self.decisions
            or self.action_items
            or self.conclusion != NOT_SPECIFIED
            or self.overview != NOT_SPECIFIED
        )


# Machine-readable description of the schema, embedded in the prompt. Keeping
# it next to the model reduces the chance of the two drifting apart.
SCHEMA_SPEC = """{
  "title": "short name for the meeting, or \\"Not specified\\"",
  "date": "date the meeting took place, or \\"Not specified\\"",
  "time": "time the meeting took place, or \\"Not specified\\"",
  "participants": ["names of people who spoke or were addressed"],
  "overview": "one or two sentence summary of the meeting",
  "discussion_points": ["topics that were discussed"],
  "decisions": ["things the group explicitly agreed or decided"],
  "action_items": [
    {"task": "what must be done",
     "responsible_person": "who agreed to do it, or \\"Not specified\\"",
     "deadline": "when by, or \\"Not specified\\""}
  ],
  "conclusion": "how the meeting ended / final outcome"
}"""