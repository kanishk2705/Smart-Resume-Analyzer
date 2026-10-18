from streamlit.testing.v1 import AppTest

import api_client
from api_client import BackendError
from tests.conftest import make_docx

APP_PATH = "frontend/app.py"
RESUME_TEXT = "I write Python and SQL and deploy with Docker every day."
JOB_TEXT = "We need Python, SQL and Docker experience."

ANALYSIS = {
    "id": 3,
    "share_token": "tok-abc-123",
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


def connect(monkeypatch, llm_configured=True, history=(), history_scope="shared"):
    """Stands in for a healthy backend."""
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: {
            "status": "ok",
            "embedding_model": "test-model",
            "llm_configured": llm_configured,
            "history_scope": history_scope,
        },
    )
    monkeypatch.setattr(api_client, "get_history", lambda: list(history))
    monkeypatch.setattr(api_client, "get_report", lambda share_token: b"%PDF-fake")

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


def test_app_displays_ats_checks_with_examples(monkeypatch):
    connect(monkeypatch)

    text = markdown_text(run_with_analysis())

    assert "✅ **Email address** — Email address found." in text
    assert "⚠️ **First-person pronouns** — Resumes conventionally drop them." in text
    assert "- I deploy Python services with Docker." in text


def click(app, label):
    return next(b for b in app.button if b.label == label).click().run()


def test_ai_coach_is_hidden_when_the_backend_has_no_llm(monkeypatch):
    connect(monkeypatch, llm_configured=False)

    app = run_with_analysis()

    assert not app.exception
    assert any("GROQ_API_KEY" in info.value for info in app.info)
    assert not any(b.label == "Generate Learning Path" for b in app.button)


def test_learning_path_requests_only_missing_skills_for_this_job(monkeypatch):
    requested = []

    def fake_learning_path(skills, job_title):
        requested.append((skills, job_title))
        return [{"skill": "Kubernetes", "resource_name": "Course", "project": "Deploy an app",
                 "link": "https://example.com"}]

    connect(monkeypatch)
    monkeypatch.setattr(api_client, "get_learning_path", fake_learning_path)

    app = click(run_with_analysis(), "Generate Learning Path")

    assert not app.exception
    assert requested == [(["Kubernetes"], "Platform Engineer")]
    assert "Learn Kubernetes" in markdown_text(app)


def test_coach_answers_stay_on_screen_after_another_interaction(monkeypatch):
    connect(monkeypatch)
    monkeypatch.setattr(
        api_client,
        "get_learning_path",
        lambda skills, job_title: [{"skill": "Kubernetes", "resource_name": "Course",
                                    "project": "Deploy an app", "link": "https://example.com"}],
    )
    monkeypatch.setattr(api_client, "get_rewrites", lambda statements, job_title: [])

    app = click(click(run_with_analysis(), "Generate Learning Path"), "Suggest Rewrites")

    assert "Learn Kubernetes" in markdown_text(app)


def test_rewrites_are_requested_for_statements_the_ats_check_flagged(monkeypatch):
    requested = []

    def fake_rewrites(statements, job_title):
        requested.append(statements)
        return [{"original": statements[0], "rewritten": "Deployed [N] Python services.",
                 "note": "Dropped the pronoun"}]

    connect(monkeypatch)
    monkeypatch.setattr(api_client, "get_rewrites", fake_rewrites)

    app = click(run_with_analysis(), "Suggest Rewrites")

    assert not app.exception
    assert requested == [["I deploy Python services with Docker."]]
    assert "**After:** Deployed [N] Python services." in markdown_text(app)


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
    app = click(app, "Open")

    assert not app.exception
    assert "62%" in [m.value for m in app.metric]


def test_same_analysis_can_be_open_in_two_tabs(monkeypatch):
    connect(monkeypatch, history=SUMMARIES)
    app = AppTest.from_file(APP_PATH)
    app.session_state.analysis_results = ANALYSIS
    app.session_state.history_view = ANALYSIS

    app.run()

    assert not app.exception
    assert sum(b.label == "Generate Learning Path" for b in app.button) == 2


def test_history_delete_removes_the_selected_analysis(monkeypatch):
    deleted = []
    connect(monkeypatch, history=SUMMARIES)
    monkeypatch.setattr(api_client, "delete_analysis", deleted.append)

    app = AppTest.from_file(APP_PATH).run()
    app = click(app, "Delete")

    assert not app.exception
    assert deleted == [3]


def test_share_link_opens_that_analysis(monkeypatch):
    opened = []
    connect(monkeypatch)
    monkeypatch.setattr(api_client, "get_shared", lambda t: opened.append(t) or ANALYSIS)
    app = AppTest.from_file(APP_PATH)
    app.query_params["share"] = "tok-abc-123"

    app.run()

    assert not app.exception
    assert opened == ["tok-abc-123"]
    assert "62%" in [m.value for m in app.metric]


def test_stale_share_link_says_so_and_opens_the_app_normally(monkeypatch):
    def missing(share_token):
        raise BackendError("No shared analysis at this link.")

    connect(monkeypatch)
    monkeypatch.setattr(api_client, "get_shared", missing)
    app = AppTest.from_file(APP_PATH)
    app.query_params["share"] = "gone"

    app.run()

    assert not app.exception
    assert len(app.metric) == 0
    assert any("no longer valid" in warning.value for warning in app.warning)


def test_analysis_shows_its_share_link(monkeypatch):
    connect(monkeypatch)

    app = run_with_analysis()

    assert any(code.value == "http://localhost:8501/?share=tok-abc-123" for code in app.code)


def test_per_session_history_explains_it_is_private_and_temporary(monkeypatch):
    connect(monkeypatch, history_scope="session")

    app = AppTest.from_file(APP_PATH).run()

    assert any("private to this browser tab" in caption.value for caption in app.caption)


def test_session_id_is_created_once_and_sent_with_requests(monkeypatch):
    connect(monkeypatch)
    used = []
    monkeypatch.setattr(api_client, "use_session", used.append)

    app = AppTest.from_file(APP_PATH).run()
    app.run()

    assert len(used) == 2
    assert used[0] == used[1] == app.session_state.session_id
    assert len(used[0]) >= 24

def test_session_header_is_sent_only_from_the_thread_that_set_it(monkeypatch):
    import threading

    sent = []

    def fake_request(method, url, timeout, **kwargs):
        sent.append(kwargs.get("headers", {}).get("X-Session-Id"))
        response = type("R", (), {"status_code": 200, "json": lambda self: {"analyses": []}})()
        return response

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    api_client.use_session("session-of-main-thread-0001")
    api_client.get_history()
    other = threading.Thread(target=api_client.get_history)
    other.start()
    other.join()

    assert sent == ["session-of-main-thread-0001", None]


def test_in_process_mode_calls_the_api_without_a_server(client, monkeypatch):
    # The `client` fixture points the app at a fake encoder and a temporary history
    monkeypatch.setattr(api_client, "BACKEND_URL", api_client.IN_PROCESS)
    api_client.use_session("in-process-session-0001")

    health = api_client.get_health()
    saved = api_client.analyze("resume.docx", make_docx(RESUME_TEXT), JOB_TEXT)
    shared = api_client.get_shared(saved["share_token"])

    assert health["status"] == "ok"
    assert shared["id"] == saved["id"]
    assert api_client.get_report(saved["share_token"]).startswith(b"%PDF")


def test_in_process_mode_reports_api_errors_like_http(client, monkeypatch):
    monkeypatch.setattr(api_client, "BACKEND_URL", api_client.IN_PROCESS)

    try:
        api_client.analyze("notes.txt", b"plain text", JOB_TEXT)
    except BackendError as e:
        assert "Unsupported file type" in str(e)
    else:
        raise AssertionError("expected a BackendError")