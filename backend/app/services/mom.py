"""Turn a structured extraction into formatted Minutes of Meeting.

Kept separate from both the extraction and the exporters:

* extraction concerns itself with *what* was said;
* this module concerns itself with *how the minutes read*;
* the exporters (PDF, DOCX) concern themselves with file formats.

The document is built once as a :class:`MomDocument` and rendered from there,
so the screen view, the PDF and the DOCX cannot disagree about the content.

Every missing value is rendered as ``Not specified`` rather than being silently
omitted. That is deliberate: a reader of the minutes must be able to tell the
difference between "no deadline was agreed" and "we forgot to write the
deadline down".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.database.session import utcnow
from app.schemas.extraction import NOT_SPECIFIED, MeetingExtraction

# Section headings, in the order required by the project specification.
SECTION_OVERVIEW = "Meeting Overview"
SECTION_DISCUSSION = "Key Discussion Points"
SECTION_DECISIONS = "Decisions Made"
SECTION_ACTIONS = "Action Items"
SECTION_OUTCOMES = "Important Outcomes"
SECTION_CONCLUSION = "Conclusion"


@dataclass
class MomDocument:
    """A fully resolved set of minutes, ready to render in any format."""

    title: str
    date: str
    time: str
    participants: list[str]
    overview: str
    discussion_points: list[str]
    decisions: list[str]
    action_items: list[dict]
    conclusion: str
    outcomes: list[str] = field(default_factory=list)

    # Provenance, printed in the footer so a reader can judge the source.
    source_filename: str = ""
    generated_at: object = None
    asr_model: str = ""
    extraction_model: str = ""

    @property
    def participants_display(self) -> str:
        return ", ".join(self.participants) if self.participants else NOT_SPECIFIED


def _as_sentence(text: str) -> str:
    """Capitalise the first letter of a phrase for display."""
    text = text.strip()
    if not text:
        return NOT_SPECIFIED
    return text[0].upper() + text[1:]


def _derive_outcomes(extraction: MeetingExtraction) -> list[str]:
    """Summarise the concrete outcomes of the meeting.

    "Important Outcomes" is not a field the model is asked for, because it
    would largely duplicate decisions and action items. It is derived instead,
    which keeps the document internally consistent by construction.
    """
    outcomes: list[str] = []
    for decision in extraction.decisions:
        outcomes.append(f"Agreed: {_as_sentence(decision)}")
    for action in extraction.action_items:
        if action.deadline != NOT_SPECIFIED:
            outcomes.append(
                f"{action.task.capitalize()} to be completed by {action.deadline}."
            )
    return outcomes


def build_mom(
    extraction: MeetingExtraction,
    *,
    source_filename: str = "",
    asr_model: str = "",
    extraction_model: str = "",
    fallback_title: str = "",
) -> MomDocument:
    """Assemble minutes from an extraction.

    Args:
        extraction: the structured information.
        source_filename: original upload name, shown in the footer.
        asr_model: speech-recognition model, shown in the footer so a reader
            can judge how reliable the transcript was.
        extraction_model: the language model used, likewise for provenance.
        fallback_title: used when the meeting has no extractable title.
    """
    title = extraction.title
    if title == NOT_SPECIFIED and fallback_title:
        # Prefer a readable title derived from the filename over "Not specified".
        title = fallback_title

    return MomDocument(
        title=title,
        date=extraction.date,
        time=extraction.time,
        participants=list(extraction.participants),
        overview=extraction.overview,
        discussion_points=list(extraction.discussion_points),
        decisions=[_as_sentence(d) for d in extraction.decisions],
        action_items=[a.model_dump() for a in extraction.action_items],
        conclusion=extraction.conclusion,
        outcomes=_derive_outcomes(extraction),
        source_filename=source_filename,
        generated_at=utcnow(),
        asr_model=asr_model,
        extraction_model=extraction_model,
    )


def title_from_filename(filename: str) -> str:
    """Derive a human-readable title from an uploaded filename.

    ``weekly_sync-2026-05-12.mp4`` becomes ``Weekly Sync 2026 05 12``. Used only
    as a fallback when the transcript provides no title.
    """
    stem = filename.rsplit(".", 1)[0]
    stem = stem.replace("_", " ").replace("-", " ")
    words = [w for w in stem.split() if w]
    if not words:
        return NOT_SPECIFIED
    return " ".join(word.capitalize() if not word.isdigit() else word for word in words)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_text(mom: MomDocument) -> str:
    """Render minutes as plain text, matching the required layout exactly."""
    lines: list[str] = []
    add = lines.append

    add("MINUTES OF MEETING")
    add("=" * 60)
    add("")
    add(f"Meeting Title: {mom.title}")
    add(f"Date: {mom.date}")
    add(f"Time: {mom.time}")
    add(f"Participants: {mom.participants_display}")
    add("")

    add(f"1. {SECTION_OVERVIEW}")
    add("")
    add(mom.overview)
    add("")

    add(f"2. {SECTION_DISCUSSION}")
    add("")
    if mom.discussion_points:
        for point in mom.discussion_points:
            add(f"   - {_as_sentence(point)}")
    else:
        add(f"   {NOT_SPECIFIED}")
    add("")

    add(f"3. {SECTION_DECISIONS}")
    add("")
    if mom.decisions:
        for decision in mom.decisions:
            add(f"   - {decision}")
    else:
        add(f"   {NOT_SPECIFIED}")
    add("")

    add(f"4. {SECTION_ACTIONS}")
    add("")
    if mom.action_items:
        add("   | Task | Responsible Person | Deadline |")
        add("   |------|--------------------|----------|")
        for item in mom.action_items:
            task = str(item.get("task", NOT_SPECIFIED)).replace("|", "/")
            person = str(item.get("responsible_person", NOT_SPECIFIED)).replace("|", "/")
            deadline = str(item.get("deadline", NOT_SPECIFIED)).replace("|", "/")
            add(f"   | {task} | {person} | {deadline} |")
    else:
        add(f"   {NOT_SPECIFIED}")
    add("")

    add(f"5. {SECTION_OUTCOMES}")
    add("")
    if mom.outcomes:
        for outcome in mom.outcomes:
            add(f"   - {outcome}")
    else:
        add(f"   {NOT_SPECIFIED}")
    add("")

    add(f"6. {SECTION_CONCLUSION}")
    add("")
    add(_as_sentence(mom.conclusion) if mom.conclusion != NOT_SPECIFIED else NOT_SPECIFIED)
    add("")
    add("-" * 60)
    add("Generated automatically by the AI-Based Meeting Minutes Generator.")
    if mom.source_filename:
        add(f"Source recording: {mom.source_filename}")
    if mom.asr_model:
        add(f"Speech recognition: {mom.asr_model}")
    if mom.generated_at:
        add(f"Generated at: {mom.generated_at.isoformat(timespec='seconds')}Z")
    add(
        "Values shown as 'Not specified' were not stated in the recording and "
        "were not inferred."
    )

    return "\n".join(lines) + "\n"


def render_markdown(mom: MomDocument) -> str:
    """Render minutes as Markdown, used by the web interface."""
    lines: list[str] = []
    add = lines.append

    add("# MINUTES OF MEETING")
    add("")
    add(f"**Meeting Title:** {mom.title}  ")
    add(f"**Date:** {mom.date}  ")
    add(f"**Time:** {mom.time}  ")
    add(f"**Participants:** {mom.participants_display}")
    add("")

    add(f"## 1. {SECTION_OVERVIEW}")
    add("")
    add(mom.overview)
    add("")

    add(f"## 2. {SECTION_DISCUSSION}")
    add("")
    if mom.discussion_points:
        for point in mom.discussion_points:
            add(f"- {_as_sentence(point)}")
    else:
        add(f"_{NOT_SPECIFIED}_")
    add("")

    add(f"## 3. {SECTION_DECISIONS}")
    add("")
    if mom.decisions:
        for decision in mom.decisions:
            add(f"- {decision}")
    else:
        add(f"_{NOT_SPECIFIED}_")
    add("")

    add(f"## 4. {SECTION_ACTIONS}")
    add("")
    if mom.action_items:
        add("| Task | Responsible Person | Deadline |")
        add("|------|--------------------|----------|")
        for item in mom.action_items:
            task = str(item.get("task", NOT_SPECIFIED)).replace("|", "/")
            person = str(item.get("responsible_person", NOT_SPECIFIED)).replace("|", "/")
            deadline = str(item.get("deadline", NOT_SPECIFIED)).replace("|", "/")
            add(f"| {task} | {person} | {deadline} |")
    else:
        add(f"_{NOT_SPECIFIED}_")
    add("")

    add(f"## 5. {SECTION_OUTCOMES}")
    add("")
    if mom.outcomes:
        for outcome in mom.outcomes:
            add(f"- {outcome}")
    else:
        add(f"_{NOT_SPECIFIED}_")
    add("")

    add(f"## 6. {SECTION_CONCLUSION}")
    add("")
    add(_as_sentence(mom.conclusion) if mom.conclusion != NOT_SPECIFIED else NOT_SPECIFIED)
    add("")
    add("---")
    add("_Generated automatically by the AI-Based Meeting Minutes Generator._")
    if mom.source_filename:
        add(f"_Source recording: {mom.source_filename}_")
    if mom.asr_model:
        add(f"_Speech recognition: {mom.asr_model}_")
    if mom.generated_at:
        add(f"_Generated at: {mom.generated_at.isoformat(timespec='seconds')}Z_")

    return "\n".join(lines) + "\n"