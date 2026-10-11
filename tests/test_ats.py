from core.ats import check_resume
from core.parsing import ParsedDocument

GOOD_RESUME = """Jane Doe
jane.doe@example.com | +1 416 555 0199 | linkedin.com/in/janedoe

Experience
Data Analyst, Northwind (2021 - 2024)
• Built 12 dashboards in Tableau used by 40 store managers every week.
• Cut monthly reporting time from 2 days to 1 hour by automating Excel exports.
• Trained 3 junior analysts on data quality checks and peer review.

Education
B.Sc. in Statistics, University of Toronto, 2021

Skills
Python, SQL, Tableau, Excel
""" + ("Additional detail about analytics work. " * 40)


def check(text: str, key: str, **document_fields):
    report = check_resume(ParsedDocument(text, **document_fields))
    return next(c for c in report.checks if c.key == key)


def test_clean_resume_passes_every_check():
    report = check_resume(ParsedDocument(GOOD_RESUME, page_count=1))

    assert {c.key: c.status for c in report.checks} == {
        "email": "pass",
        "phone": "pass",
        "profiles": "pass",
        "sections": "pass",
        "length": "pass",
        "characters": "pass",
        "quantified": "pass",
        "verbs": "pass",
        "bullet_length": "pass",
        "pronouns": "pass",
    }
    assert report.score == 100.0


def test_score_gives_half_credit_for_warnings():
    report = check_resume(ParsedDocument("Short resume with nothing useful in it."))

    statuses = [c.status for c in report.checks]
    expected = (statuses.count("pass") + 0.5 * statuses.count("warn")) / len(statuses) * 100
    assert report.score == round(expected, 2)
    assert report.score < 60


def test_contact_details_hidden_behind_links_are_a_warning():
    text = "Jane Doe\nMobile | Email | LinkedIn | GitHub"
    links = ["mailto:jane@example.com", "tel:+14165550199", "https://linkedin.com/in/jane"]

    assert check(text, "email", links=links).status == "warn"
    assert check(text, "phone", links=links).status == "warn"
    assert check(text, "profiles", links=links).status == "warn"
    assert "hyperlink" in check(text, "email", links=links).message


def test_missing_contact_details_fail():
    assert check("Jane Doe", "email").status == "fail"
    assert check("Jane Doe", "phone").status == "fail"


def test_date_range_is_not_a_phone_number():
    assert check("Analyst, Northwind 2019 - 2021", "phone").status == "fail"
    assert check("Call (416) 555-0199", "phone").status == "pass"


def test_missing_sections_are_named():
    result = check("Experience\nBuilt things\nSkills\nPython", "sections")

    assert result.status == "fail"
    assert "education" in result.message
    assert "skills" not in result.message.split(".")[0]


def test_projects_can_stand_in_for_experience():
    text = "Projects\nBuilt things\nEducation\nB.Sc.\nSkills\nPython"

    assert check(text, "sections").status == "pass"


def test_length_warnings():
    assert check("too short", "length").status == "warn"
    assert check("word " * 1200, "length").status == "warn"
    assert check("word " * 500, "length", page_count=3).status == "warn"
    assert check("word " * 500, "length", page_count=2).message == "500 words on 2 pages."


def test_icon_glyphs_are_flagged():
    result = check(" Mobile |  Email", "characters")

    assert result.status == "warn"
    assert result.message.startswith("2 symbols")


def test_unquantified_statements_are_listed():
    text = (
        "Experience\n"
        "• Built dashboards for store managers.\n"
        "• Wrote pipelines for the data team.\n"
        "• Trained junior analysts on quality checks.\n"
        "• Cut reporting time by 90%.\n"
    )

    result = check(text, "quantified")

    assert result.status == "warn"
    assert result.message.startswith("1 of 4 statements include a number.")
    assert result.examples == [
        "Built dashboards for store managers.",
        "Wrote pipelines for the data team.",
        "Trained junior analysts on quality checks.",
    ]


def test_only_experience_and_project_bullets_are_judged():
    text = (
        "Experience\n• Cut reporting time by 90%.\n"
        "Education\n• Relevant coursework: databases and statistics.\n"
    )

    assert check(text, "quantified").message.startswith("1 of 1 statements")


def test_passive_openers_are_flagged():
    text = "Experience\n• Responsible for 5 dashboards.\n• Built 3 pipelines.\n"

    result = check(text, "verbs")

    assert result.status == "warn"
    assert result.examples == ["Responsible for 5 dashboards."]


def test_overlong_bullets_are_flagged():
    text = "Experience\n• " + "word " * 45 + "\n• Built 3 pipelines.\n"

    assert check(text, "bullet_length").status == "warn"


def test_first_person_pronouns_are_flagged():
    result = check("Summary\nI build data tools that my team relies on.", "pronouns")

    assert result.status == "warn"
    assert result.examples == ["I build data tools that my team relies on."]
    assert check("Summary\nBuilds data tools for the team.", "pronouns").status == "pass"
