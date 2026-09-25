"""Tests for Minutes of Meeting rendering and the PDF/DOCX exporters.

The exporters are lazy-imported in the application so the API can boot without
the report-writing libraries. These tests skip cleanly when those libraries are
absent, rather than failing on an environment that only runs the core.
"""

from __future__ import annotations

import zipfile

import pytest

from app.schemas.extraction import NOT_SPECIFIED, MeetingExtraction
from app.services import exporters, mom

# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def _extraction(**overrides) -> MeetingExtraction:
    base = {
        "title": "Sprint Planning Meeting",
        "participants": ["Priya", "Rahul"],
        "overview": "The team planned the sprint.",
        "discussion_points": ["Backlog review", "Database choice"],
        "decisions": ["Use PostgreSQL for the main database"],
        "action_items": [
            {
                "task": "update the schema documentation",
                "responsible_person": "Rahul",
                "deadline": "next Monday",
            }
        ],
        "conclusion": "The sprint is planned.",
    }
    base.update(overrides)
    return MeetingExtraction.model_validate(base)


def test_build_mom_carries_every_section():
    document = mom.build_mom(_extraction(), source_filename="sprint.mp4")

    assert document.title == "Sprint Planning Meeting"
    assert document.participants == ["Priya", "Rahul"]
    assert document.discussion_points
    assert document.decisions
    assert document.action_items
    assert document.conclusion


def test_missing_values_render_as_not_specified():
    """A reader must be able to tell 'not agreed' from 'not recorded'."""
    document = mom.build_mom(
        MeetingExtraction.model_validate({}), source_filename="meeting.mp4"
    )
    text = mom.render_text(document)

    assert NOT_SPECIFIED in text
    assert "Date:" in text


def test_action_item_without_owner_or_deadline_still_renders():
    document = mom.build_mom(
        _extraction(
            action_items=[{"task": "Prepare the checklist"}],
        ),
        source_filename="m.mp4",
    )
    row = document.action_items[0]

    assert row["responsible_person"] == NOT_SPECIFIED
    assert row["deadline"] == NOT_SPECIFIED


def test_title_falls_back_to_the_filename_when_absent():
    """A meeting with no stated topic still needs a heading."""
    document = mom.build_mom(
        MeetingExtraction.model_validate({}),
        source_filename="weekly_team_sync_2024.mp4",
        fallback_title=mom.title_from_filename("weekly_team_sync_2024.mp4"),
    )

    assert document.title != NOT_SPECIFIED
    assert "Weekly" in document.title or "weekly" in document.title


def test_title_from_filename_is_readable():
    assert mom.title_from_filename("weekly-team-sync.mp4") == "Weekly Team Sync"
    assert mom.title_from_filename("my_meeting.m4a") == "My Meeting"


def test_render_text_contains_the_required_sections():
    text = mom.render_text(mom.build_mom(_extraction(), source_filename="m.mp4"))

    for heading in (
        "MINUTES OF MEETING",
        "Meeting Title:",
        "Participants:",
        "Key Discussion Points",
        "Decisions Made",
        "Action Items",
        "Conclusion",
    ):
        assert heading in text


def test_render_text_includes_the_action_item_table():
    text = mom.render_text(mom.build_mom(_extraction(), source_filename="m.mp4"))

    assert "| Task | Responsible Person | Deadline |" in text
    assert "update the schema documentation" in text
    assert "next Monday" in text


def test_important_outcomes_are_derived_from_decisions():
    """Outcomes must not be a second, potentially conflicting model output.

    Deriving them from the decisions and deadlines keeps the two sections
    consistent by construction.
    """
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    outcomes = " ".join(document.outcomes)

    assert "PostgreSQL" in outcomes
    assert "next Monday" in outcomes


def test_provenance_footer_names_the_models():
    """The report should let a reader see how it was produced."""
    document = mom.build_mom(
        _extraction(),
        source_filename="m.mp4",
        asr_model="faster-whisper:base",
        extraction_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    text = mom.render_text(document)

    assert "faster-whisper:base" in text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in text


def test_render_is_deterministic():
    """The same document must render identically every time."""
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    assert mom.render_text(document) == mom.render_text(document)


def test_markdown_tables_escape_pipes():
    """A pipe inside a task must not break the table layout."""
    document = mom.build_mom(
        _extraction(action_items=[{"task": "check a|b throughput"}]),
        source_filename="m.mp4",
    )
    text = mom.render_text(document)
    row = [line for line in text.splitlines() if "throughput" in line][0]

    assert row.count("|") == 4  # opening, two separators, closing


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

reportlab = pytest.importorskip("reportlab", reason="reportlab not installed")
docx = pytest.importorskip("docx", reason="python-docx not installed")


def test_export_pdf_writes_a_valid_pdf(tmp_path):
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    path = tmp_path / "minutes.pdf"

    exporters.export_pdf(document, path)

    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_export_docx_writes_a_valid_docx(tmp_path):
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    path = tmp_path / "minutes.docx"

    exporters.export_docx(document, path)

    assert path.exists()
    # A DOCX is a ZIP container; this is the real signature check rather than
    # just confirming some bytes were written.
    assert zipfile.is_zipfile(path)


def test_exported_docx_contains_the_content(tmp_path):
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    path = tmp_path / "minutes.docx"
    exporters.export_docx(document, path)

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")

    assert "update the schema documentation" in xml
    assert "PostgreSQL" in xml


def test_export_pdf_survives_markup_characters(tmp_path):
    """Transcript text can contain '<' or '&'.

    reportlab parses a markup subset, so unescaped text raises or is silently
    swallowed. This is a real failure mode, not a hypothetical one.
    """
    document = mom.build_mom(
        _extraction(
            overview="Latency < 200ms and R&D are priorities.",
            discussion_points=["Compare a < b and x & y"],
        ),
        source_filename="m.mp4",
    )
    path = tmp_path / "markup.pdf"

    exporters.export_pdf(document, path)
    assert path.read_bytes().startswith(b"%PDF")


def test_safe_stem_produces_a_portable_filename():
    stem = exporters.safe_stem("Sprint Planning: Q3/Q4 & Beyond!")
    assert "/" not in stem
    assert ":" not in stem
    assert " " not in stem


def test_safe_stem_uses_the_fallback_when_the_title_is_empty():
    assert exporters.safe_stem("", fallback="meeting_1") == "meeting_1"
    assert exporters.safe_stem(NOT_SPECIFIED, fallback="meeting_1") == "meeting_1"


def test_export_creates_parent_directories(tmp_path):
    """Exporting must work even if the reports directory does not exist yet."""
    document = mom.build_mom(_extraction(), source_filename="m.mp4")
    path = tmp_path / "nested" / "deeper" / "minutes.pdf"

    exporters.export_pdf(document, path)
    assert path.exists()