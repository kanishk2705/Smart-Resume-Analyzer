import io

import pypdf

from core.report import build_report

ANALYSIS = {
    "id": 7,
    "created_at": "2026-10-11T14:30:00+00:00",
    "resume_filename": "jane_doe.pdf",
    "job_title": "Senior Data Analyst",
    "match_score": 64.11,
    "components": [
        {"key": "skills", "label": "Skills match", "score": 63.64, "weight": 0.4,
         "detail": "6 of 11 required skills found."},
        {"key": "experience", "label": "Experience relevance", "score": None, "weight": 0.2,
         "detail": "No Experience or Projects section found in the resume."},
    ],
    "skills": [
        {"name": "Python", "category": "Programming Languages", "status": "matched",
         "related_to": None},
        {"name": "PostgreSQL", "category": "Databases", "status": "related",
         "related_to": "MySQL"},
        {"name": "Airflow", "category": "Data Engineering & BI", "status": "missing",
         "related_to": None},
    ],
    "additional_skills": [],
    "requirements": [
        {"requirement": "Build and maintain dashboards.",
         "evidence": "Built interactive dashboards in Tableau.", "similarity": 0.57,
         "status": "strong"},
    ],
    "top_sentences": [],
    "sections_found": ["experience"],
    "ats": {
        "score": 85.0,
        "checks": [
            {"key": "email", "label": "Email address", "status": "pass",
             "message": "Email address found.", "examples": []},
            {"key": "pronouns", "label": "First-person pronouns", "status": "warn",
             "message": "Resumes conventionally drop them.", "examples": ["I build tools."]},
        ],
    },
}


def report_text(analysis) -> str:
    pdf = build_report(analysis)
    assert pdf.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return " ".join(" ".join(page.extract_text().split()) for page in reader.pages)


def test_report_contains_every_section_of_the_analysis():
    text = report_text(ANALYSIS)

    for expected in [
        "Resume Match Report",
        "jane_doe.pdf",
        "Senior Data Analyst",
        "2026-10-11 14:30 UTC",
        "64% overall match",
        "ATS format score: 85%",
        "Skills match",
        "n/a",
        "PostgreSQL",
        "Resume has MySQL",
        "Build and maintain dashboards.",
        "Built interactive dashboards in Tableau.",
        "First-person pronouns",
    ]:
        assert expected in text, expected


def test_missing_skills_are_listed_before_matched_ones():
    text = report_text(ANALYSIS)

    assert text.index("Airflow") < text.index("PostgreSQL") < text.index("Python")


def test_text_outside_latin_1_does_not_break_the_report():
    analysis = {
        **ANALYSIS,
        "job_title": "Data Analyst – “Growth” team • 数据分析师 🚀",
        "skills": [],
    }

    text = report_text(analysis)

    assert 'Data Analyst - "Growth" team' in text
    assert "No known skills were found" in text


def test_long_tables_flow_onto_further_pages():
    requirement = {
        "requirement": "Design and analyse experiments to guide product decisions. " * 3,
        "evidence": "Ran A/B tests on pricing pages and presented the findings. " * 3,
        "similarity": 0.4,
        "status": "partial",
    }

    pdf = build_report({**ANALYSIS, "requirements": [requirement] * 40})

    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) > 1
