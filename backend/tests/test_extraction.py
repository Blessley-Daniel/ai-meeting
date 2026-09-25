"""Tests for the AI extraction stage: schema, JSON repair, grounding and rules.

These tests deliberately avoid loading the language model. They target the
logic that is *ours* - the schema, the JSON repair, the grounding check and the
deterministic rules - because that is where bugs will be introduced by future
changes, and because a test suite that downloads and runs a 0.5B model is too
slow to run on every commit.

The model itself is exercised by the evaluation harness (training/evaluate.py)
and by the manual pipeline test, not here.
"""

from __future__ import annotations

import json

import pytest

from app.schemas.extraction import (
    NOT_SPECIFIED,
    ActionItem,
    MeetingExtraction,
)
from app.services import rules
from app.services.extraction import (
    ExtractionDiagnostics,
    _apply_grounding,
    _close_unbalanced,
    extract_json_object,
)

# ---------------------------------------------------------------------------
# Schema behaviour
# ---------------------------------------------------------------------------


def test_missing_fields_default_to_not_specified():
    """An empty object must still produce a complete, valid schema."""
    result = MeetingExtraction.model_validate({})

    assert result.title == NOT_SPECIFIED
    assert result.date == NOT_SPECIFIED
    assert result.overview == NOT_SPECIFIED
    assert result.conclusion == NOT_SPECIFIED
    assert result.discussion_points == []
    assert result.decisions == []
    assert result.action_items == []


def test_action_item_without_owner_or_deadline_is_marked_not_specified():
    """The anti-hallucination contract: absent facts are labelled, not invented."""
    result = MeetingExtraction.model_validate(
        {"action_items": [{"task": "Prepare the report"}]}
    )

    item = result.action_items[0]
    assert item.task == "Prepare the report"
    assert item.responsible_person == NOT_SPECIFIED
    assert item.deadline == NOT_SPECIFIED


def test_action_item_key_aliases_are_accepted():
    """Small models emit 'owner'/'due' instead of our field names."""
    result = MeetingExtraction.model_validate(
        {
            "action_items": [
                {"action": "Send the invite", "owner": "Priya", "due": "Friday"}
            ]
        }
    )

    item = result.action_items[0]
    assert item.task == "Send the invite"
    assert item.responsible_person == "Priya"
    assert item.deadline == "Friday"


def test_action_item_instances_are_preserved():
    """Passing ActionItem objects must not silently discard them.

    This was a real bug: the validator only handled dicts and strings, so the
    rules layer's ActionItem objects were dropped and extraction scored zero.
    """
    items = [ActionItem(task="Update the docs", responsible_person="Rahul")]
    result = MeetingExtraction.model_validate({"action_items": items})

    assert len(result.action_items) == 1
    assert result.action_items[0].task == "Update the docs"


def test_empty_or_placeholder_action_items_are_dropped():
    """Entries with no real task text are noise, not action items."""
    result = MeetingExtraction.model_validate(
        {
            "action_items": [
                {"task": ""},
                {"task": NOT_SPECIFIED},
                {"task": "Real task"},
                12345,
                None,
            ]
        }
    )

    assert [a.task for a in result.action_items] == ["Real task"]


def test_a_bare_string_action_item_is_accepted_as_a_task():
    """Some models return a list of task strings with no owner or deadline."""
    result = MeetingExtraction.model_validate({"action_items": ["Send the report"]})

    assert result.action_items[0].task == "Send the report"
    assert result.action_items[0].responsible_person == NOT_SPECIFIED


def test_duplicate_action_items_are_removed():
    result = MeetingExtraction.model_validate(
        {
            "action_items": [
                {"task": "Test the prototype"},
                {"task": "test the prototype"},
            ]
        }
    )

    assert len(result.action_items) == 1


def test_lists_tolerate_a_bare_string():
    """A model that returns a single string instead of a list must not crash."""
    result = MeetingExtraction.model_validate(
        {"decisions": "Use PostgreSQL", "discussion_points": None}
    )

    assert result.decisions == ["Use PostgreSQL"]
    assert result.discussion_points == []


def test_to_json_dict_round_trips():
    original = MeetingExtraction.model_validate(
        {
            "title": "Sprint Planning",
            "decisions": ["Use PostgreSQL"],
            "action_items": [{"task": "Test it", "responsible_person": "Meera"}],
        }
    )

    restored = MeetingExtraction.model_validate(original.to_json_dict())
    assert restored == original


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------


def test_extracts_bare_json_object():
    parsed = extract_json_object('{"decisions": ["Use PostgreSQL"]}')
    assert parsed["decisions"] == ["Use PostgreSQL"]


def test_extracts_json_from_markdown_fence():
    raw = 'Here you go:\n```json\n{"decisions": ["Use Redis"]}\n```\nHope that helps.'
    parsed = extract_json_object(raw)
    assert parsed["decisions"] == ["Use Redis"]


def test_extracts_json_with_surrounding_prose():
    raw = 'Sure! {"title": "Standup"} and that is all.'
    parsed = extract_json_object(raw)
    assert parsed["title"] == "Standup"


def test_extracts_json_embedded_in_thinking_text():
    """Some models narrate before emitting the object.

    A naive first-{ to last-} slice takes the wrong span here and produces
    unparseable text, so the extractor must find the real object.
    """
    raw = (
        "Let me think about the {meeting} first. "
        'The answer is {"decisions": ["Ship on Friday"]}. Done.'
    )
    parsed = extract_json_object(raw)
    assert parsed["decisions"] == ["Ship on Friday"]


def test_trailing_commas_are_removed():
    raw = '{"decisions": ["Use PostgreSQL",], "title": "Sync",}'
    parsed = extract_json_object(raw)
    assert parsed["decisions"] == ["Use PostgreSQL"]
    assert parsed["title"] == "Sync"


def test_single_quotes_are_converted():
    raw = "{'title': 'Daily Standup', 'decisions': ['Use Redis']}"
    parsed = extract_json_object(raw)
    assert parsed["title"] == "Daily Standup"


def test_truncated_object_is_closed():
    """A response cut off at max_new_tokens is repaired, not discarded.

    Small models regularly hit the token limit mid-object; throwing the whole
    extraction away would lose the fields that did arrive.
    """
    raw = '{"title": "Sprint Planning", "decisions": ["Use PostgreSQL"'
    parsed = extract_json_object(raw)
    assert parsed["title"] == "Sprint Planning"
    assert parsed["decisions"] == ["Use PostgreSQL"]


def test_truncated_nested_action_items_are_closed():
    """Bracket balancing must track both braces and square brackets.

    A response cut off inside an action item leaves an unclosed list *and* an
    unclosed object; handling only braces corrupts the result.
    """
    raw = '{"action_items": [{"task": "Write tests", "responsible_person": "Meera"'
    parsed = extract_json_object(raw)
    assert parsed["action_items"][0]["task"] == "Write tests"


def test_close_unbalanced_handles_stray_openers():
    assert _close_unbalanced('{"a": [1, 2') == '{"a": [1, 2]}'
    assert _close_unbalanced('{"a": 1') == '{"a": 1}'
    assert _close_unbalanced('plain text') == 'plain text'


def test_returned_payload_is_plain_json_only():
    """The repair step must return parseable JSON, whatever the input."""
    for raw in ('{"a": 1}', '```json\n{"a": 1}\n```', 'blah {"a": 1', '{}'):
        assert isinstance(extract_json_object(raw), dict)


# ---------------------------------------------------------------------------
# Grounding (anti-hallucination)
# ---------------------------------------------------------------------------


def test_fabricated_participants_are_removed():
    """The original motivating failure: invented attendees must not survive."""
    transcript = (
        "Rahul said the authentication module is on track. "
        "The team decided to use PostgreSQL."
    )
    extracted = MeetingExtraction.model_validate(
        {
            "participants": ["Rahul", "Samantha", "John"],
            "decisions": ["use PostgreSQL"],
        }
    )

    grounded = _apply_grounding(extracted, transcript, ExtractionDiagnostics())

    assert grounded.participants == ["Rahul"]
    assert "Samantha" not in grounded.participants


def test_grounded_values_are_kept():
    transcript = "The team decided to use PostgreSQL for the main database."
    extracted = MeetingExtraction.model_validate(
        {"decisions": ["use PostgreSQL for the main database"]}
    )

    grounded = _apply_grounding(extracted, transcript, ExtractionDiagnostics())

    assert grounded.decisions == ["use PostgreSQL for the main database"]


def test_ungrounded_deadline_is_not_invented():
    """A deadline that appears nowhere in the transcript must not be reported.

    Accepting a bare 'Friday' simply because it looks like a date is exactly
    the fabrication the grounding check exists to prevent.
    """
    transcript = "Rahul will update the schema documentation."
    extracted = MeetingExtraction.model_validate(
        {
            "action_items": [
                {"task": "update the schema documentation", "deadline": "Friday"}
            ]
        }
    )

    grounded = _apply_grounding(extracted, transcript, ExtractionDiagnostics())
    item = grounded.action_items[0]

    assert item.task == "update the schema documentation"
    assert item.deadline == NOT_SPECIFIED


def test_grounded_deadline_is_kept():
    transcript = "Rahul will update the schema documentation by next Monday."
    extracted = MeetingExtraction.model_validate(
        {
            "action_items": [
                {"task": "update the schema documentation", "deadline": "next Monday"}
            ]
        }
    )

    grounded = _apply_grounding(extracted, transcript, ExtractionDiagnostics())
    assert grounded.action_items[0].deadline == "next Monday"


def test_slightly_corrupted_name_is_accepted():
    """Whisper mishears names; rejecting on any difference loses real people.

    'Priya' heard as 'Prya' is still recognisably the same participant, so
    approximate matching is used rather than exact string equality.
    """
    transcript = "Prya will send the invoice."
    extracted = MeetingExtraction.model_validate({"participants": ["Priya"]})

    grounded = _apply_grounding(extracted, transcript, ExtractionDiagnostics())
    assert grounded.participants == ["Priya"]


def test_grounding_survives_missing_keys():
    """A partial dict from a small model must not raise."""
    extracted = MeetingExtraction.model_validate({"title": "Sync"})
    grounded = _apply_grounding(
        extracted, "Some transcript about a sync.", ExtractionDiagnostics()
    )
    assert grounded.participants == []


# ---------------------------------------------------------------------------
# Deterministic rules
# ---------------------------------------------------------------------------


def test_rules_find_an_explicit_decision():
    text = "John said the prototype should be ready by Friday. The team decided to use Python for the backend."
    decisions = rules.find_decisions(text)

    assert any("python" in d.lower() for d in decisions)


def test_rules_find_agreed_decision():
    decisions = rules.find_decisions("Everyone agreed that we should move to AWS.")
    assert any("aws" in d.lower() for d in decisions)


def test_rules_do_not_treat_discussion_as_a_decision():
    """Talking about an option is not deciding it - the core distinction."""
    text = "Meera suggested that we could maybe use Postgres, but nothing was decided."
    decisions = rules.find_decisions(text)

    assert not any("postgres" in d.lower() for d in decisions)


def test_rules_extract_action_item_with_owner_and_deadline():
    text = "Sarah agreed to test the prototype on Thursday."
    items = rules.find_action_items(text)

    assert len(items) == 1
    assert items[0].responsible_person == "Sarah"
    assert "test the prototype" in items[0].task
    assert items[0].deadline == "Thursday"


def test_rules_leave_deadline_out_of_the_task_text():
    """The deadline is its own field; repeating it inside the task is noise."""
    text = "Rahul will rerun the baseline experiments by the end of the month."
    items = rules.find_action_items(text)

    assert items[0].task == "rerun the baseline experiments"
    assert items[0].deadline == "end of the month"


def test_rules_mark_missing_deadline_as_not_specified():
    items = rules.find_action_items("Rahul will prepare the deployment checklist.")
    assert items[0].deadline == NOT_SPECIFIED


def test_rules_ignore_narration_as_an_action_item():
    """'will walk us through' describes the meeting, not a task."""
    text = "Lakshmi will walk us through the details."
    items = rules.find_action_items(text)

    assert items == []


def test_rules_respect_an_explicit_no_deadline():
    text = "Rahul will prepare the checklist. No deadline has been set for that."
    items = rules.find_action_items(text)

    if items:
        assert items[0].deadline == NOT_SPECIFIED


def test_rules_keep_titles_with_names():
    """Dr. Menon must not be recorded as 'Menon' in one field only."""
    text = "Dr. Menon will review the draft paper."
    items = rules.find_action_items(text)

    assert items[0].responsible_person == "Dr Menon"


def test_rules_find_participants_from_actions():
    text = "Priya will run the tests. Rahul will write the docs."
    items = rules.find_action_items(text)
    participants = rules.find_participants(text, [i.responsible_person for i in items])

    assert "Priya" in participants
    assert "Rahul" in participants


def test_rules_return_nothing_for_empty_input():
    assert rules.find_decisions("") == []
    assert rules.find_action_items("") == []
    assert rules.find_participants("") == []


def test_rules_extract_conclusion_from_cue():
    text = "We covered the budget. To conclude, the training budget is increased and tooling is consolidated."
    conclusion = rules.find_conclusion(text)

    assert "training budget" in conclusion.lower()


def test_extract_with_rules_returns_a_full_schema():
    """The rules layer alone must produce a complete, valid extraction."""
    text = (
        "Priya said the schema is ready. The team decided to use PostgreSQL. "
        "Rahul will update the schema documentation by next Monday. "
        "To conclude, the schema work is on track."
    )
    result = rules.extract_with_rules(text)

    assert result.decisions
    assert result.action_items
    assert result.conclusion != NOT_SPECIFIED


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("by next Monday", "next Monday"),
        ("before Friday", "Friday"),
        ("on Wednesday", "Wednesday"),
        ("due tomorrow", "tomorrow"),
        ("by the end of the month", "end of the month"),
    ],
)
def test_deadline_phrases_are_recognised(phrase, expected):
    assert rules.find_deadline(f"Rahul will do it {phrase}.") == expected


def test_no_deadline_phrase_returns_the_sentinel():
    assert rules.find_deadline("Rahul will do it.") == NOT_SPECIFIED
    assert rules.find_deadline("There is no deadline for this.") == NOT_SPECIFIED
