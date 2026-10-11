from streamlit.testing.v1 import AppTest

import api_client
from api_client import BackendError

APP_PATH = "frontend/app.py"

ANALYSIS = {
    "match_score": 62.5,
    "components": [
        {"key": "skills", "label": "Skills match", "score": 50.0, "weight": 0.4,
         "detail": "1 of 3 required skills found, 1 more covered by a related skill."},
        {"key": "coverage", "label": "Requirement coverage", "score": 71.0, "weight": 0.3,
         "detail": "1 of 1 job requirements have strong evidence in the resume."},
        {"key": "experience", "label": "Experience relevance", "score": None, "weight": 0.2,
         "detail": "No Experience or Projects section found in the resume."},
        {"key": "education", "label": "Education", "score": 100.0, "weight": 0.1,
         "detail": "Job asks for a Bachelor's degree; resume shows a Master's degree."},
    ],
    "skills": [
        {"name": "Docker", "category": "Cloud & DevOps", "status": "matched", "related_to": None},
        {"name": "Kubernetes", "category": "Cloud & DevOps", "status": "missing",
         "related_to": None},
        {"name": "PostgreSQL", "category": "Databases", "status": "related",
         "related_to": "MySQL"},
    ],
    "additional_skills": ["Python"],
    "requirements": [
        {"requirement": "Deploy services with Docker.",
         "evidence": "I deploy Python services with Docker.", "similarity": 0.71,
         "status": "strong"},
    ],
    "top_sentences": [{"text": "I deploy Python services with Docker.", "score": 0.71}],
    "sections_found": ["experience", "skills"],
}


def healthy(gemini_configured):
    return lambda: {
        "status": "ok",
        "embedding_model": "test-model",
        "gemini_configured": gemini_configured,
    }


def run_with_analysis(analysis):
    app = AppTest.from_file(APP_PATH)
    app.session_state.analysis_results = analysis
    return app.run()


def markdown_text(app):
    return "\n".join(block.value for block in app.markdown)


def test_app_renders_when_backend_is_down(monkeypatch):
    def unreachable():
        raise BackendError("Could not reach the backend at http://localhost:8000.")

    monkeypatch.setattr(api_client, "get_health", unreachable)

    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert "Could not reach the backend" in app.sidebar.error[0].value


def test_app_displays_score_breakdown_skills_and_evidence(monkeypatch):
    monkeypatch.setattr(api_client, "get_health", healthy(gemini_configured=True))

    app = run_with_analysis(ANALYSIS)

    assert not app.exception
    assert app.metric[0].value == "62%"
    text = markdown_text(app)
    assert "**Skills match** — 50%" in text
    assert "**Experience relevance** — not assessed" in text
    assert "✅ `Docker`" in text
    assert "❌ `Kubernetes`" in text
    assert "🟡 `PostgreSQL` (you have MySQL)" in text
    assert any("I deploy Python services with Docker." in info.value for info in app.info)
    assert len(app.dataframe) == 1
    # The key is configured on the backend, so the user is not asked for one
    assert len(app.text_input) == 0


def test_app_asks_for_api_key_when_backend_has_none(monkeypatch):
    monkeypatch.setattr(api_client, "get_health", healthy(gemini_configured=False))

    app = run_with_analysis(ANALYSIS)

    assert len(app.text_input) == 1


def test_learning_path_requests_only_missing_skills(monkeypatch):
    requested = []

    def fake_recommendations(skills, api_key=None):
        requested.append(skills)
        return [{"skill": "Kubernetes", "resource_name": "Course", "project": "Deploy an app",
                 "link": "https://example.com"}]

    monkeypatch.setattr(api_client, "get_health", healthy(gemini_configured=True))
    monkeypatch.setattr(api_client, "get_recommendations", fake_recommendations)

    app = run_with_analysis(ANALYSIS)
    next(b for b in app.button if b.label == "Generate Learning Path").click().run()

    assert not app.exception
    assert requested == [["Kubernetes"]]
    assert "Learn Kubernetes" in markdown_text(app)


def test_no_learning_path_when_nothing_is_missing(monkeypatch):
    monkeypatch.setattr(api_client, "get_health", healthy(gemini_configured=True))
    analysis = {**ANALYSIS, "skills": [ANALYSIS["skills"][0]]}

    app = run_with_analysis(analysis)

    assert not app.exception
    assert not any(b.label == "Generate Learning Path" for b in app.button)
