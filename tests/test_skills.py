import json

import pytest

from core.skills import SKILLS_FILE, get_taxonomy


def names(text: str) -> set[str]:
    return {skill.name for skill in get_taxonomy().extract(text)}


def test_finds_skills_with_category_and_group():
    skills = {s.name: s for s in get_taxonomy().extract("I use PostgreSQL and Tableau daily.")}

    assert skills["PostgreSQL"].category == "Databases"
    assert skills["PostgreSQL"].group == "relational database"
    assert skills["Tableau"].group == "BI tool"


def test_matching_is_case_insensitive_and_follows_aliases():
    assert names("PYTHON, k8s, sklearn and amazon web services") == {
        "Python",
        "Kubernetes",
        "scikit-learn",
        "AWS",
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Languages: C++, C#, Java", {"C++", "C#", "Java"}),
        ("Deployed on AWS S3 with CI/CD", {"AWS", "Amazon S3", "CI/CD"}),
        ("Built Node.js and ASP.NET services", {"Node.js", "ASP.NET"}),
    ],
)
def test_skills_with_symbols_and_digits_survive(text, expected):
    assert names(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I wrote JavaScript.",  # not Java
        "We are an excellent team.",  # not Excel
        "Built a scalable service.",  # not Scala
        "I excel at delivery.",  # not Excel: lower-case verb
        "Enrolled in Spring 2024.",  # not Spring Boot
    ],
)
def test_does_not_match_inside_other_words(text):
    assert names(text) <= {"JavaScript"}


def test_single_letter_languages_only_match_in_lists():
    assert names("Languages: Python, R, C, Go") == {"Python", "R", "C", "Go"}
    assert names("A C Kanishk led R&D. Go to market strategy.") == set()


def test_multi_word_skills_match_across_line_breaks():
    assert "Machine Learning" in names("experience in machine\nlearning systems")


def test_dashboards_count_as_data_visualization():
    assert "Data Visualization" in names("Built dashboards for executives.")


def test_no_skills_in_unrelated_text():
    assert names("Designed seasonal menus and managed a kitchen.") == set()


def test_taxonomy_file_has_unique_skill_names():
    data = json.loads(SKILLS_FILE.read_text(encoding="utf-8"))
    all_names = [entry["name"] for entries in data.values() for entry in entries]

    assert len(all_names) == len(set(all_names)) == len(get_taxonomy())


def test_every_group_has_at_least_two_skills():
    data = json.loads(SKILLS_FILE.read_text(encoding="utf-8"))
    groups = [e["group"] for entries in data.values() for e in entries if "group" in e]

    assert all(groups.count(group) >= 2 for group in groups)
