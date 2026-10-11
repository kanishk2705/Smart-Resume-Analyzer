import pytest

from core.scoring import ResumeAnalyzer, get_analyzer
from tests.conftest import FakeEncoder

JD = "We need Python, SQL and Docker experience. Kubernetes and Kubernetes operators are a plus."


def test_identical_documents_score_100(analyzer):
    result = analyzer.analyze(JD, JD)

    assert result.match_score == 100.0
    assert result.missing_keywords == []


def test_relevant_resume_outscores_irrelevant_one(analyzer):
    relevant = analyzer.analyze("I write Python and SQL and deploy with Docker every day.", JD)
    irrelevant = analyzer.analyze("I ran marketing and sales campaigns for retail brands.", JD)

    assert relevant.match_score > irrelevant.match_score
    assert irrelevant.match_score == 0.0


def test_missing_keywords_are_jd_terms_absent_from_resume_most_frequent_first(analyzer):
    result = analyzer.analyze("I write Python and SQL and deploy with Docker every day.", JD)

    assert result.missing_keywords[0] == "kubernetes"
    assert "python" not in result.missing_keywords
    assert "sql" not in result.missing_keywords


def test_missing_keywords_respects_limit():
    analyzer = ResumeAnalyzer(FakeEncoder(), max_missing_keywords=2)

    result = analyzer.analyze("I write Python.", JD)

    assert len(result.missing_keywords) == 2


def test_top_sentences_are_ranked_and_limited():
    analyzer = ResumeAnalyzer(FakeEncoder(), top_k_sentences=2)
    resume = (
        "I ran marketing campaigns for retail brands. "
        "I write Python and SQL and deploy with Docker. "
        "I maintain Python services for the data team."
    )

    result = analyzer.analyze(resume, JD)

    assert [s.text for s in result.top_sentences] == [
        "I write Python and SQL and deploy with Docker.",
        "I maintain Python services for the data team.",
    ]
    assert result.top_sentences[0].score > result.top_sentences[1].score


def test_short_fragments_are_not_hero_sentences(analyzer):
    result = analyzer.analyze("Python. SQL. Docker. Skills.", JD)

    assert result.top_sentences == []


def test_documents_with_only_stopwords_do_not_crash(analyzer):
    result = analyzer.analyze("the and of", "to be or not")

    assert result.missing_keywords == []


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
