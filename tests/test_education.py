import pytest

from core.education import education_score, highest_degree, required_degree

DIPLOMA, BACHELOR, MASTER, DOCTORATE = 1, 2, 3, 4


@pytest.mark.parametrize(
    "text, level",
    [
        ("B.Sc. in Statistics, University of Delhi", BACHELOR),
        ("Bachelor of Technology in Computer Science", BACHELOR),
        ("B.Tech, Electronics", BACHELOR),
        ("Master of Science in Data Science", MASTER),
        ("MS in Computer Science", MASTER),
        ("MBA, Finance", MASTER),
        ("Ph.D. in Machine Learning", DOCTORATE),
        ("Diploma in Culinary Arts", DIPLOMA),
        ("B.Tech 2019, M.Tech 2021", MASTER),
    ],
)
def test_highest_degree(text, level):
    assert highest_degree(text) == level


@pytest.mark.parametrize(
    "text",
    [
        "Certified Scrum Master",
        "Merged the feature into the master branch",
        "Proficient in MS Excel and MS Office",
        "Response time under 200 ms",
        "Mastered several frameworks",
    ],
)
def test_lookalikes_are_not_degrees(text):
    assert highest_degree(text) is None


def test_required_degree_is_the_lowest_one_mentioned():
    assert required_degree("Bachelor's or Master's degree in Computer Science") == BACHELOR
    assert required_degree("Master's degree required; PhD preferred") == MASTER


def test_unspecified_degree_means_bachelor():
    assert required_degree("A degree in a quantitative field") == BACHELOR


def test_no_requirement():
    assert required_degree("3+ years of experience with Python") is None


@pytest.mark.parametrize(
    "held, required, expected",
    [
        (MASTER, BACHELOR, 100.0),
        (BACHELOR, BACHELOR, 100.0),
        (BACHELOR, MASTER, 50.0),
        (DIPLOMA, MASTER, 0.0),
        (None, BACHELOR, 0.0),
        (MASTER, None, None),
    ],
)
def test_education_score(held, required, expected):
    assert education_score(held, required) == expected
