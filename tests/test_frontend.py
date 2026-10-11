from streamlit.testing.v1 import AppTest

import api_client
from api_client import BackendError

APP_PATH = "frontend/app.py"

ANALYSIS = {
    "id": 3,
    "created_at": "2026-10-11T14:30:00+00:00",
    "resume_filename": "jane.pdf",
    "job_title": "Platform Engineer",
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
    "ats": {
        "score": 75.0,
        "checks": [
            {"key": "email", "label": "Email address", "status": "pass",
             "message": "Email address found.", "examples": []},
            {"key": "pronouns", "label": "First-person pronouns", "status": "warn",
             "message": "Resumes conventionally drop them.",
             "examples": ["I deploy Python services with Docker."]},
        ],
    },
}

SUMMARIES = [
    {"id": 3, "created_at": "2026-10-11T14:30:00+00:00", "resume_filename": "jane_v2.pdf",
     "job_title": "Platform Engineer", "match_score": 62.5, "ats_score": 75.0},
    {"id": 2, "created_at": "2026-10-10T09:00:00+00:00", "resume_filename": "jane_v1.pdf",
     "job_title": "Platform Engineer", "match_score": 50.0, "ats_score": 70.0},
]

RANKING = {
    "job_title": "Platform Engineer",
    "candidates": [
        {"rank": 1, "filename": "engineer.pdf", "match_score": 80.0,
         "components": ANALYSIS["components"], "matched_skills": ["Docker"],
         "related_skills": ["PostgreSQL"], "missing_skills": ["Kubernetes"]},
        {"rank": 2, "filename": "marketer.pdf", "match_score": 12.0,
         "components": ANALYSIS["components"], "matched_skills": [], "related_skills": [],
         "missing_skills": ["Docker", "Kubernetes"]},
    ],
    "failed": [{"filename": "scan.pdf", "error": "No text found in 'scan.pdf'."}],
}


def connect(monkeypatch, gemini_configured=True, history=()):
    """Stands in for a healthy backend."""
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: {
            "status": "ok",
            "embedding_model": "test-model",
            "gemini_configured": gemini_configured,
        },
    )
    monkeypatch.setattr(api_client, "get_history", lambda: list(history))
    monkeypatch.setattr(api_client, "get_report", lambda analysis_id: b"%PDF-fake")


def run_with_analysis(analysis=ANALYSIS):
    app = AppTest.from_file(APP_PATH)
    app.session_state.analysis_results = analysis
    return app.run()


def markdown_text(app):
    return "\n".join(block.value for block in app.markdown)


def test_app_renders_when_backend_is_down(monkeypatch):
    def unreachable():
        raise BackendError("Could not reach the backend at http://localhost:8000.")

    monkeypatch.setattr(api_client, "get_health", unreachable)
    monkeypatch.setattr(api_client, "get_history", unreachable)

    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert "Could not reach the backend" in app.sidebar.error[0].value


def test_app_displays_score_breakdown_skills_and_evidence(monkeypatch):
    connect(monkeypatch)

    app = run_with_analysis()

    assert not app.exception
    assert [m.value for m in app.metric] == ["62%", "75%"]
    text = markdown_text(app)
    assert "**Skills match** — 50%" in text
    assert "**Experience relevance** — not assessed" in text
    assert "✅ `Docker`" in text
    assert "❌ `Kubernetes`" in text
    assert "🟡 `PostgreSQL` (you have MySQL)" in text
    assert any("I deploy Python services with Docker." in info.value for info in app.info)
    # The key is configured on the backend, so the user is not asked for one
    assert len(app.text_input) == 0


def test_app_displays_ats_checks_with_examples(monkeypatch):
    connect(monkeypatch)

    text = markdown_text(run_with_analysis())

    assert "✅ **Email address** — Email address found." in text
    assert "⚠️ **First-person pronouns** — Resumes conventionally drop them." in text
    assert "- I deploy Python services with Docker." in text


def test_app_asks_for_api_key_when_backend_has_none(monkeypatch):
    connect(monkeypatch, gemini_configured=False)

    app = run_with_analysis()

    assert len(app.text_input) == 1


def test_learning_path_requests_only_missing_skills(monkeypatch):
    requested = []

    def fake_recommendations(skills, api_key=None):
        requested.append(skills)
        return [{"skill": "Kubernetes", "resource_name": "Course", "project": "Deploy an app",
                 "link": "https://example.com"}]

    connect(monkeypatch)
    monkeypatch.setattr(api_client, "get_recommendations", fake_recommendations)

    app = run_with_analysis()
    next(b for b in app.button if b.label == "Generate Learning Path").click().run()

    assert not app.exception
    assert requested == [["Kubernetes"]]
    assert "Learn Kubernetes" in markdown_text(app)


def test_no_learning_path_when_nothing_is_missing(monkeypatch):
    connect(monkeypatch)

    app = run_with_analysis({**ANALYSIS, "skills": [ANALYSIS["skills"][0]]})

    assert not app.exception
    assert not any(b.label == "Generate Learning Path" for b in app.button)


def test_ranking_shows_candidates_and_skipped_files(monkeypatch):
    connect(monkeypatch)
    app = AppTest.from_file(APP_PATH)
    app.session_state.ranking_results = RANKING

    app.run()

    assert not app.exception
    assert any("scan.pdf" in warning.value for warning in app.warning)
    assert [e.label for e in app.expander] == [
        "#1 engineer.pdf — 80%",
        "#2 marketer.pdf — 12%",
    ]
    assert "❌ **Missing:** `Docker`, `Kubernetes`" in markdown_text(app)


def test_history_lists_analyses_and_opens_one(monkeypatch):
    connect(monkeypatch, history=SUMMARIES)
    monkeypatch.setattr(api_client, "get_analysis", lambda analysis_id: ANALYSIS)

    app = AppTest.from_file(APP_PATH).run()
    assert app.selectbox[0].options == [
        "#3 · jane_v2.pdf · Platform Engineer",
        "#2 · jane_v1.pdf · Platform Engineer",
    ]
    next(b for b in app.button if b.label == "Open").click().run()

    assert not app.exception
    assert "62%" in [m.value for m in app.metric]


def test_same_analysis_can_be_open_in_two_tabs(monkeypatch):
    connect(monkeypatch, gemini_configured=False, history=SUMMARIES)
    app = AppTest.from_file(APP_PATH)
    app.session_state.analysis_results = ANALYSIS
    app.session_state.history_view = ANALYSIS

    app.run()

    assert not app.exception
    assert len(app.text_input) == 2


def test_history_delete_removes_the_selected_analysis(monkeypatch):
    deleted = []
    connect(monkeypatch, history=SUMMARIES)
    monkeypatch.setattr(api_client, "delete_analysis", deleted.append)

    app = AppTest.from_file(APP_PATH).run()
    next(b for b in app.button if b.label == "Delete").click().run()

    assert not app.exception
    assert deleted == [3]
