from streamlit.testing.v1 import AppTest

import api_client
from api_client import BackendError

APP_PATH = "frontend/app.py"

ANALYSIS = {
    "match_score": 62.5,
    "missing_keywords": ["kubernetes", "terraform"],
    "top_sentences": [{"text": "I deploy Python services with Docker.", "score": 0.71}],
}


def test_app_renders_when_backend_is_down(monkeypatch):
    def unreachable():
        raise BackendError("Could not reach the backend at http://localhost:8000.")

    monkeypatch.setattr(api_client, "get_health", unreachable)

    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert "Could not reach the backend" in app.sidebar.error[0].value


def test_app_displays_stored_analysis(monkeypatch):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: {"status": "ok", "embedding_model": "test-model", "gemini_configured": True},
    )
    app = AppTest.from_file(APP_PATH)
    app.session_state.analysis_results = ANALYSIS

    app.run()

    assert not app.exception
    assert app.metric[0].value == "62.5%"
    assert any("I deploy Python services with Docker." in info.value for info in app.info)
    assert any("`kubernetes`" in block.value for block in app.markdown)
    # The key is configured on the backend, so the user is not asked for one
    assert len(app.text_input) == 0
