import pytest

from core.config import Settings
from core.scoring import ResumeAnalyzer, ScoreComponent, _weighted_mean, get_analyzer
from tests.conftest import FakeEncoder

RESUME = """Jane Doe
jane@example.com

Experience
• I write Python and SQL services for the data team every day.
• I deploy Python services with Docker to production clusters.
• I ran marketing campaigns for retail brands last year.

Education
B.Sc. in Computer Science, 2020

Skills
Python, SQL, Docker, MySQL
"""

JD = """About us
We run marketing and sales software for small shops.

Requirements
- Strong Python and SQL knowledge is required.
- You will deploy services with Docker and Kubernetes.
- Knowledge of PostgreSQL is needed for this role.
- Bachelor's degree in Computer Science.
"""


def component(result, key):
    return next(c for c in result.components if c.key == key)


def test_returns_four_weighted_components(analyzer):
    result = analyzer.analyze(RESUME, JD)

    assert [(c.key, c.weight) for c in result.components] == [
        ("skills", 0.40),
        ("coverage", 0.30),
        ("experience", 0.20),
        ("education", 0.10),
    ]
    assert all(0 <= c.score <= 100 for c in result.components)
    assert result.match_score == _weighted_mean(result.components)
    assert result.sections_found == ["experience", "education", "skills"]


def test_skills_are_matched_related_or_missing(analyzer):
    result = analyzer.analyze(RESUME, JD)

    assert {(s.name, s.status, s.related_to) for s in result.skills} == {
        ("Python", "matched", None),
        ("SQL", "matched", None),
        ("Docker", "matched", None),
        ("PostgreSQL", "related", "MySQL"),
        ("Kubernetes", "missing", None),
    }
    assert result.additional_skills == ["MySQL"]
    # 3 matched + half credit for 1 related, out of 5
    assert component(result, "skills").score == 70.0


def test_education_requirement_met(analyzer):
    education = component(analyzer.analyze(RESUME, JD), "education")

    assert education.score == 100.0
    assert "Bachelor's" in education.detail


def test_education_is_read_from_the_education_section_only(analyzer):
    resume = RESUME.replace("B.Sc. in Computer Science, 2020", "Self-taught").replace(
        "last year.", "last year for an MBA programme."
    )

    assert component(analyzer.analyze(resume, JD), "education").score == 0.0


def test_components_that_cannot_be_assessed_are_left_out(analyzer):
    resume = "I write Python and SQL services for the data team every day."
    jd = "We want someone friendly who enjoys working with people."

    result = analyzer.analyze(resume, jd)

    assert component(result, "skills").score is None
    assert component(result, "experience").score is None
    assert component(result, "education").score is None
    assert result.match_score == component(result, "coverage").score
    assert result.sections_found == []


def test_weighted_mean_renormalises_over_assessed_components():
    components = [
        ScoreComponent("skills", "", 80.0, 0.4, ""),
        ScoreComponent("coverage", "", 50.0, 0.1, ""),
        ScoreComponent("experience", "", None, 0.5, ""),
    ]

    assert _weighted_mean(components) == 74.0
    assert _weighted_mean([ScoreComponent("skills", "", None, 1.0, "")]) == 0.0


def test_similarity_is_rescaled_between_floor_and_ceiling():
    analyzer = ResumeAnalyzer(
        FakeEncoder(), settings=Settings(similarity_floor=0.2, similarity_ceiling=0.7)
    )

    assert analyzer._rescale(0.1) == 0.0
    assert analyzer._rescale(0.45) == 50.0
    assert analyzer._rescale(0.9) == 100.0


def test_relevant_resume_outscores_irrelevant_one(analyzer):
    irrelevant = RESUME.replace("Python", "marketing").replace("SQL", "sales").replace(
        "Docker", "sales"
    )

    assert analyzer.analyze(RESUME, JD).match_score > analyzer.analyze(irrelevant, JD).match_score


def test_each_requirement_is_paired_with_its_best_evidence(analyzer):
    result = analyzer.analyze(RESUME, JD)

    by_requirement = {r.requirement: r for r in result.requirements}
    docker = by_requirement["You will deploy services with Docker and Kubernetes."]
    assert docker.evidence == "I deploy Python services with Docker to production clusters."
    assert by_requirement["Strong Python and SQL knowledge is required."].status == "strong"
    assert by_requirement["Bachelor's degree in Computer Science."].status == "weak"


def test_company_and_benefits_text_is_not_treated_as_a_requirement(analyzer):
    result = analyzer.analyze(RESUME, JD)

    assert len(result.requirements) == 4
    assert not any("marketing and sales software" in r.requirement for r in result.requirements)


def test_content_late_in_a_long_resume_still_counts(analyzer):
    """The old scorer embedded the resume as one string, which the model cut at 256 tokens."""
    filler = "\n".join(f"• Organised community event number {i} for residents." for i in range(80))
    resume = f"Experience\n{filler}\n• I write Python and SQL services for the data team."

    result = analyzer.analyze(resume, JD)

    python_requirement = result.requirements[0]
    assert python_requirement.evidence == "I write Python and SQL services for the data team."
    assert python_requirement.status == "strong"


def test_top_sentences_are_ranked_and_limited():
    analyzer = ResumeAnalyzer(FakeEncoder(), settings=Settings(top_k_sentences=2))

    result = analyzer.analyze(RESUME, JD)

    assert len(result.top_sentences) == 2
    assert result.top_sentences[0].score >= result.top_sentences[1].score
    assert "marketing" not in result.top_sentences[0].text


def test_experience_uses_only_the_most_relevant_statements():
    analyzer = ResumeAnalyzer(FakeEncoder(), settings=Settings(top_experience_units=1))
    padded = RESUME.replace(
        "Education", "• I organised the annual office party for everyone.\n\nEducation"
    )

    assert component(analyzer.analyze(padded, JD), "experience").score == (
        component(analyzer.analyze(RESUME, JD), "experience").score
    )


@pytest.mark.parametrize("resume", ["Python. SQL. Docker.", "x"])
def test_fragmentary_resumes_do_not_crash(analyzer, resume):
    result = analyzer.analyze(resume, JD)

    assert 0 <= result.match_score <= 100
    assert len(result.requirements) == 4


def test_very_short_job_description_does_not_crash(analyzer):
    result = analyzer.analyze(RESUME, "Python")

    assert [r.requirement for r in result.requirements] == ["Python"]


@pytest.mark.slow
def test_real_model_ranks_semantic_match_above_unrelated_text():
    """Loads the real SBERT model, so it is excluded from the fast CI run."""
    get_analyzer.cache_clear()
    analyzer = get_analyzer()
    jd = "Looking for a data analyst skilled in data visualization and reporting."

    relevant = analyzer.analyze("I built interactive dashboards and charts for executives.", jd)
    unrelated = analyzer.analyze("I repaired diesel engines and maintained truck fleets.", jd)

    assert relevant.match_score > unrelated.match_score
    assert 0 <= relevant.match_score <= 100
