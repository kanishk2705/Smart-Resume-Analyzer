import pytest

from core import coaching
from core.coaching import learning_path, rewrite_bullets


@pytest.fixture
def llm(monkeypatch):
    """Replaces the LLM call; records prompts and returns whatever `llm.answer` is set to."""

    class FakeLLM:
        answer: dict = {}
        prompts: list[tuple[str, str]] = []

        def __call__(self, system, user):
            self.prompts.append((system, user))
            return self.answer

    fake = FakeLLM()
    fake.prompts = []
    monkeypatch.setattr(coaching, "complete_json", fake)
    return fake


def test_learning_path_adds_search_links(llm):
    llm.answer = {
        "recommendations": [
            {"skill": "SQL", "resource_name": "Coursera: SQL Basics", "project": "Build a report"}
        ]
    }

    assert learning_path(["SQL"], "Data Analyst") == [
        {
            "skill": "SQL",
            "resource_name": "Coursera: SQL Basics",
            "project": "Build a report",
            "link": "https://www.google.com/search?q=Coursera%3A%20SQL%20Basics",
        }
    ]


def test_learning_path_sends_job_title_and_only_the_top_skills(llm):
    learning_path(["a1", "b2", "c3", "d4", "e5", "f6", "g7"], "Data Analyst")

    _, user = llm.prompts[0]
    assert "Target job: Data Analyst" in user
    assert "a1, b2, c3, d4, e5" in user
    assert "f6" not in user


def test_nothing_to_coach_makes_no_llm_call(llm):
    assert learning_path([], "Data Analyst") == []
    assert rewrite_bullets([], "Data Analyst") == []
    assert llm.prompts == []


@pytest.mark.parametrize(
    "answer",
    [{}, {"recommendations": "none"}, {"recommendations": ["text", 3, None]}],
)
def test_malformed_answers_yield_no_recommendations(llm, answer):
    llm.answer = answer

    assert learning_path(["SQL"], "Data Analyst") == []


def test_missing_and_null_fields_become_empty_strings(llm):
    llm.answer = {"recommendations": [{"skill": "SQL", "resource_name": None}]}

    assert learning_path(["SQL"], "Data Analyst") == [
        {
            "skill": "SQL",
            "resource_name": "",
            "project": "",
            "link": "https://www.google.com/search?q=SQL",
        }
    ]


def test_rewrites_keep_the_real_original_text(llm):
    llm.answer = {
        "rewrites": [
            {"original": "paraphrased by the model", "rewritten": "Built [N] dashboards.",
             "note": "Added a placeholder"}
        ]
    }

    assert rewrite_bullets(["Worked on dashboards."], "Data Analyst") == [
        {
            "original": "Worked on dashboards.",
            "rewritten": "Built [N] dashboards.",
            "note": "Added a placeholder",
        }
    ]


def test_unchanged_and_empty_rewrites_are_dropped(llm):
    llm.answer = {
        "rewrites": [
            {"rewritten": "Built 3 pipelines.", "note": "No change"},
            {"rewritten": "", "note": "Nothing"},
            {"rewritten": "Led a team of [N] analysts.", "note": "Added size"},
        ]
    }

    rewrites = rewrite_bullets(
        ["Built 3 pipelines.", "Helped the team.", "Was team lead."], "Data Analyst"
    )

    assert [r["rewritten"] for r in rewrites] == ["Led a team of [N] analysts."]
    assert rewrites[0]["original"] == "Was team lead."


def test_rewrite_prompt_forbids_inventing_facts_and_numbers_statements(llm):
    rewrite_bullets(["Worked on dashboards.", "Helped the team."], "Data Analyst")

    system, user = llm.prompts[0]
    assert "NEVER invent facts" in system
    assert "1. Worked on dashboards.\n2. Helped the team." in user
