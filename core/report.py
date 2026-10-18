"""Renders a stored analysis as a PDF report."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

# The built-in PDF fonts only cover Latin-1, so common typographic characters are mapped
# to plain equivalents and anything else becomes "?".
_REPLACEMENTS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": "-", "…": "...",
    " ": " ", "→": "->",
})
# Gaps first, so the reader sees what to work on
_SKILL_ORDER = {"missing": 0, "related": 1, "matched": 2}

_STATUS_LABELS = {
    "matched": "Matched", "related": "Related", "missing": "Missing",
    "strong": "Strong", "partial": "Partial", "weak": "Weak",
    "pass": "Pass", "warn": "Warning", "fail": "Fail",
}
_GOOD, _MIDDLE, _BAD = (22, 128, 61), (161, 98, 7), (185, 28, 28)
_STATUS_COLOURS = {
    "matched": _GOOD, "strong": _GOOD, "pass": _GOOD,
    "related": _MIDDLE, "partial": _MIDDLE, "warn": _MIDDLE,
    "missing": _BAD, "weak": _BAD, "fail": _BAD,
}
_MUTED = (100, 100, 100)
_HEADER_STYLE = FontFace(emphasis="BOLD", fill_color=(235, 238, 242))


def _clean(text: object) -> str:
    return str(text).translate(_REPLACEMENTS).encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*_MUTED)
        self.cell(0, 6, f"Smart Resume Analyzer - page {self.page_no()}", align="C")

    def heading(self, text: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, _clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def paragraph(self, text: str, size: int = 10, colour=(0, 0, 0)):
        self.set_font("Helvetica", size=size)
        self.set_text_color(*colour)
        self.multi_cell(0, 5, _clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def status_table(self, headers, widths, rows):
        """A table whose first column is a status word, coloured by its meaning."""
        self.set_font("Helvetica", size=9)
        self.set_text_color(0, 0, 0)
        with self.table(
            col_widths=widths, headings_style=_HEADER_STYLE, line_height=5, text_align="LEFT"
        ) as table:
            table.row(headers)
            for status, *cells in rows:
                row = table.row()
                row.cell(_STATUS_LABELS[status], style=FontFace(color=_STATUS_COLOURS[status]))
                for cell in cells:
                    row.cell(_clean(cell))


def build_report(analysis: dict) -> bytes:
    """Builds the PDF for one analysis, as stored in the history."""
    pdf = _Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Resume Match Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.paragraph(
        f"Resume: {analysis['resume_filename']}\n"
        f"Job: {analysis['job_title']}\n"
        f"Analysed: {analysis['created_at'][:16].replace('T', ' ')} UTC",
        colour=_MUTED,
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(0, 0, 0)
    headline = f"{analysis['match_score']:.0f}% overall match"
    pdf.cell(0, 12, headline, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.paragraph(f"ATS format score: {analysis['ats']['score']:.0f}%", size=11)

    pdf.heading("Score breakdown")
    pdf.set_font("Helvetica", size=9)
    with pdf.table(
        col_widths=(40, 16, 16, 118),
        headings_style=_HEADER_STYLE,
        line_height=5,
        text_align="LEFT",
    ) as table:
        table.row(["Component", "Score", "Weight", "Detail"])
        for component in analysis["components"]:
            score = "n/a" if component["score"] is None else f"{component['score']:.0f}%"
            weight = f"{component['weight']:.0%}"
            table.row([component["label"], score, weight, _clean(component["detail"])])

    pdf.heading("Skills the job asks for")
    if analysis["skills"]:
        pdf.status_table(
            ["Status", "Skill", "Category", "Note"],
            (22, 48, 60, 60),
            [
                (
                    skill["status"],
                    skill["name"],
                    skill["category"],
                    f"Resume has {skill['related_to']}" if skill["related_to"] else "",
                )
                for skill in sorted(analysis["skills"], key=lambda s: _SKILL_ORDER[s["status"]])
            ],
        )
    else:
        pdf.paragraph("No known skills were found in the job description.")

    pdf.heading("Requirement coverage")
    pdf.paragraph(
        "Each job requirement, with the resume sentence that supports it best.", colour=_MUTED
    )
    pdf.status_table(
        ["Match", "Job requirement", "Best evidence in resume"],
        (20, 85, 85),
        [(r["status"], r["requirement"], r["evidence"]) for r in analysis["requirements"]],
    )

    pdf.heading("ATS format checks")
    pdf.status_table(
        ["Result", "Check", "Finding"],
        (20, 45, 125),
        [(c["status"], c["label"], c["message"]) for c in analysis["ats"]["checks"]],
    )

    pdf.ln(4)
    pdf.paragraph(
        "Scores are heuristic. They compare the wording of the resume and the job description "
        "and have not been validated against hiring outcomes.",
        size=8,
        colour=_MUTED,
    )
    return bytes(pdf.output())
