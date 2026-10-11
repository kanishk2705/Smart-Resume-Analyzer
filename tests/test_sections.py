from core.sections import JD_HEADINGS, Section, split_sections

RESUME = """Priya Sharma
priya@example.com

Work Experience
Data Analyst, Northwind
Built dashboards

EDUCATION:
B.Sc. in Statistics

Technical Skills
Python, SQL
"""


def test_splits_resume_at_known_headings():
    assert split_sections(RESUME) == [
        Section("header", "Priya Sharma\npriya@example.com"),
        Section("experience", "Data Analyst, Northwind\nBuilt dashboards"),
        Section("education", "B.Sc. in Statistics"),
        Section("skills", "Python, SQL"),
    ]


def test_text_without_headings_is_one_header_section():
    assert split_sections("Built dashboards.\nWrote SQL.") == [
        Section("header", "Built dashboards.\nWrote SQL.")
    ]


def test_headings_of_the_same_kind_are_merged():
    text = "Experience\nAnalyst at Northwind\nEducation\nB.Sc.\nInternships\nIntern at Bluebird"

    sections = {s.name: s.text for s in split_sections(text)}

    assert sections["experience"] == "Analyst at Northwind\nIntern at Bluebird"


def test_recognises_letter_spaced_and_ampersand_headings():
    text = "E D U C A T I O N\nB.Sc.\nSkills & Tools\nPython"

    assert [s.name for s in split_sections(text)] == ["education", "skills"]


def test_a_sentence_containing_a_heading_word_is_not_a_heading():
    text = "Experience\nI have experience in education technology and skills training."

    assert [s.name for s in split_sections(text)] == ["experience"]


def test_empty_sections_are_dropped():
    assert [s.name for s in split_sections("Experience\nEducation\nB.Sc.")] == ["education"]


def test_splits_job_description_with_its_own_headings():
    jd = (
        "About Us\nWe make payments simple.\n"
        "What you'll do\nBuild dashboards.\nPerks & Benefits\nSnacks."
    )

    sections = {s.name: s.text for s in split_sections(jd, JD_HEADINGS)}

    assert sections == {
        "company": "We make payments simple.",
        "responsibilities": "Build dashboards.",
        "benefits": "Snacks.",
    }
