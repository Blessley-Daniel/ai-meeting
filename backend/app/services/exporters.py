"""Export Minutes of Meeting to DOCX and PDF.

Both exporters take the same :class:`~app.services.mom.MomDocument` that the
web view renders, so the three presentations cannot drift apart.

Libraries: ``python-docx`` for DOCX (editable, for people who want to annotate
the minutes) and ``reportlab`` for PDF (fixed layout, for circulating).

Neither library is imported at module load. They are only needed when a user
actually exports a document, and importing them eagerly would slow down
start-up and make the API fail to boot on a machine where the AI requirements
are not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.services.mom import MomDocument
from app.utils.errors import DocumentExportError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def export_docx(mom: MomDocument, path: Path) -> Path:
    """Write minutes as a Word document.

    Raises:
        DocumentExportError: python-docx is missing or the file cannot be written.
    """
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DocumentExportError(
            "python-docx is not installed. Install it with: "
            "pip install -r requirements-ai.txt"
        ) from exc

    try:
        document = Document()

        heading = document.add_heading("MINUTES OF MEETING", level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Metadata block.
        meta = [
            ("Meeting Title", mom.title),
            ("Date", mom.date),
            ("Time", mom.time),
            ("Participants", mom.participants_display),
        ]
        for label, value in meta:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(f"{label}: ")
            run.bold = True
            paragraph.add_run(value)

        document.add_paragraph()

        def section(number: int, title: str) -> None:
            document.add_heading(f"{number}. {title}", level=1)

        def bullets(items: list[str]) -> None:
            if items:
                for item in items:
                    document.add_paragraph(item, style="List Bullet")
            else:
                paragraph = document.add_paragraph()
                run = paragraph.add_run("Not specified")
                run.italic = True

        section(1, "Meeting Overview")
        document.add_paragraph(mom.overview)

        section(2, "Key Discussion Points")
        bullets(mom.discussion_points)

        section(3, "Decisions Made")
        bullets(mom.decisions)

        section(4, "Action Items")
        if mom.action_items:
            table = document.add_table(rows=1, cols=3)
            table.style = "Light Grid Accent 1"
            for cell, label in zip(table.rows[0].cells, ("Task", "Responsible Person", "Deadline")):
                cell.text = ""
                run = cell.paragraphs[0].add_run(label)
                run.bold = True
            for item in mom.action_items:
                row = table.add_row().cells
                for index, key in enumerate(("task", "responsible_person", "deadline")):
                    row[index].text = str(item.get(key, "Not specified"))
        else:
            paragraph = document.add_paragraph()
            run = paragraph.add_run("Not specified")
            run.italic = True

        section(5, "Important Outcomes")
        bullets(mom.outcomes)

        section(6, "Conclusion")
        document.add_paragraph(mom.conclusion)

        # Provenance footer.
        document.add_paragraph()
        note = document.add_paragraph()
        run = note.add_run(
            "Generated automatically by the AI-Based Meeting Minutes Generator."
        )
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        if mom.source_filename:
            source = document.add_paragraph()
            run = source.add_run(
                f"Source recording: {mom.source_filename} | "
                f"Speech recognition: {mom.asr_model or 'n/a'}"
            )
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        disclaimer = document.add_paragraph()
        run = disclaimer.add_run(
            "Values shown as 'Not specified' were not stated in the recording and "
            "were not inferred."
        )
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(str(path))
    except DocumentExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise DocumentExportError(f"Could not write the DOCX report: {exc}") from exc

    logger.info("Wrote DOCX report to %s", path)
    return path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def export_pdf(mom: MomDocument, path: Path) -> Path:
    """Write minutes as a PDF.

    Raises:
        DocumentExportError: reportlab is missing or the file cannot be written.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DocumentExportError(
            "reportlab is not installed. Install it with: "
            "pip install -r requirements-ai.txt"
        ) from exc

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"Minutes of Meeting - {mom.title}",
            author="AI-Based Meeting Minutes Generator",
        )

        base = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "MomTitle", parent=base["Title"], fontSize=17, spaceAfter=12,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            "MomHeading", parent=base["Heading2"], fontSize=12,
            spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a3c6e"),
        )
        body = ParagraphStyle("MomBody", parent=base["BodyText"], fontSize=10,
                              leading=14, spaceAfter=4)
        bullet = ParagraphStyle("MomBullet", parent=body, leftIndent=10,
                                bulletIndent=2)
        small = ParagraphStyle("MomSmall", parent=body, fontSize=8,
                               textColor=colors.HexColor("#666666"))

        # reportlab interprets text as a mini-markup language, so any literal
        # angle brackets in the content must be escaped or the document fails
        # to build. A transcript can easily contain "<".
        def esc(text: object) -> str:
            return (
                str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        story: list = [Paragraph("MINUTES OF MEETING", title_style)]

        metadata = [
            ["Meeting Title:", esc(mom.title)],
            ["Date:", esc(mom.date)],
            ["Time:", esc(mom.time)],
            ["Participants:", esc(mom.participants_display)],
        ]
        meta_table = Table(metadata, colWidths=[32 * mm, None])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story += [meta_table, Spacer(1, 8)]

        def add_section(number: int, title: str, items: list[str], empty: str = "Not specified") -> None:
            story.append(Paragraph(f"{number}. {esc(title)}", heading_style))
            if items:
                for item in items:
                    story.append(
                        Paragraph(f"&bull;&nbsp; {esc(item)}", bullet)
                    )
            else:
                story.append(Paragraph(f"<i>{esc(empty)}</i>", body))

        story.append(Paragraph("1. Meeting Overview", heading_style))
        story.append(Paragraph(esc(mom.overview), body))

        add_section(2, "Key Discussion Points", mom.discussion_points)
        add_section(3, "Decisions Made", mom.decisions)

        story.append(Paragraph("4. Action Items", heading_style))
        if mom.action_items:
            data = [["Task", "Responsible Person", "Deadline"]]
            for item in mom.action_items:
                data.append(
                    [
                        Paragraph(esc(item.get("task", "Not specified")), body),
                        Paragraph(esc(item.get("responsible_person", "Not specified")), body),
                        Paragraph(esc(item.get("deadline", "Not specified")), body),
                    ]
                )
            table = Table(data, colWidths=[85 * mm, 40 * mm, 35 * mm], repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#f2f5fa")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
        else:
            story.append(Paragraph("<i>Not specified</i>", body))

        add_section(5, "Important Outcomes", mom.outcomes)

        story.append(Paragraph("6. Conclusion", heading_style))
        story.append(Paragraph(esc(mom.conclusion), body))

        story.append(Spacer(1, 14))
        story.append(
            Paragraph(
                "Generated automatically by the AI-Based Meeting Minutes Generator.",
                small,
            )
        )
        if mom.source_filename:
            story.append(
                Paragraph(
                    f"Source recording: {esc(mom.source_filename)} | "
                    f"Speech recognition: {esc(mom.asr_model or 'n/a')}",
                    small,
                )
            )
        if mom.generated_at is not None:
            story.append(
                Paragraph(f"Generated at: {esc(mom.generated_at)}Z", small)
            )
        story.append(
            Paragraph(
                "Values shown as 'Not specified' were not stated in the recording "
                "and were not inferred.",
                small,
            )
        )

        document.build(story)
    except DocumentExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise DocumentExportError(f"Could not write the PDF report: {exc}") from exc

    logger.info("Wrote PDF report to %s", path)
    return path


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def safe_stem(text: str, fallback: str = "meeting_minutes", max_length: int = 60) -> str:
    """Make a filesystem-safe filename stem out of arbitrary text."""
    import re

    stem = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")
    stem = re.sub(r"_+", "_", stem).lower()
    if not stem:
        stem = fallback
    return stem[:max_length].strip("_") or fallback