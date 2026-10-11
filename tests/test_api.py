import pytest

from backend import main
from core.config import get_settings
from core.recommendations import RecommendationError
from tests.conftest import make_docx, make_pdf

JD = "We need Python, SQL and Docker experience. Kubernetes is a plus."
RESUME = "I write Python and SQL and deploy with Docker every day."


def post_analyze(client, filename, content, job_description=JD):
    return client.post(
        "/api/v1/analyze",
        files={"resume": (filename, content)},
        data={"job_description": job_description},
    )


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_model": "all-MiniLM-L6-v2",
        "gemini_configured": False,
    }


@pytest.mark.parametrize(
    "filename, content",
    [("resume.docx", make_docx(RESUME)), ("resume.pdf", make_pdf(RESUME))],
    ids=["docx", "pdf"],
)
def test_analyze_returns_score_keywords_and_sentences(client, filename, content):
    response = post_analyze(client, filename, content)

    assert response.status_code == 200
    body = response.json()
    assert 0 < body["match_score"] <= 100
    assert "kubernetes" in body["missing_keywords"]
    assert body["top_sentences"][0]["text"] == RESUME
    assert body["top_sentences"][0]["score"] == pytest.approx(1.0)


def test_analyze_rejects_unsupported_file_type(client):
    response = post_analyze(client, "resume.txt", b"plain text")

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_rejects_corrupted_file(client):
    response = post_analyze(client, "resume.pdf", b"not a pdf")

    assert response.status_code == 422
    assert "Could not read" in response.json()["detail"]


def test_analyze_rejects_blank_job_description(client):
    response = post_analyze(client, "resume.docx", make_docx(RESUME), job_description="   ")

    assert response.status_code == 422


def test_analyze_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()

    response = post_analyze(client, "resume.pdf", b"x" * (1024 * 1024 + 1))

    assert response.status_code == 413


def test_recommendations_without_any_api_key_returns_503(client):
    response = client.post("/api/v1/recommendations", json={"missing_skills": ["sql"]})

    assert response.status_code == 503


def test_recommendations_requires_at_least_one_skill(client):
    response = client.post(
        "/api/v1/recommendations",
        json={"missing_skills": []},
        headers={"X-Gemini-Api-Key": "key"},
    )

    assert response.status_code == 422


def test_recommendations_uses_client_key_and_fills_defaults(client, monkeypatch):
    calls = []

    def fake_recommendations(skills, api_key):
        calls.append((skills, api_key))
        return [{"skill": "sql", "link": "https://example.com"}]

    monkeypatch.setattr(main, "get_ai_recommendations", fake_recommendations)

    response = client.post(
        "/api/v1/recommendations",
        json={"missing_skills": ["sql"]},
        headers={"X-Gemini-Api-Key": "client-key"},
    )

    assert response.status_code == 200
    assert calls == [(["sql"], "client-key")]
    assert response.json() == {
        "recommendations": [
            {
                "skill": "sql",
                "resource_name": "Recommended Course",
                "project": "No project generated",
                "link": "https://example.com",
            }
        ]
    }


def test_recommendations_upstream_failure_returns_502(client, monkeypatch):
    def failing(skills, api_key):
        raise RecommendationError("gemini-x: HTTP 429")

    monkeypatch.setattr(main, "get_ai_recommendations", failing)

    response = client.post(
        "/api/v1/recommendations",
        json={"missing_skills": ["sql"]},
        headers={"X-Gemini-Api-Key": "key"},
    )

    assert response.status_code == 502
    assert "HTTP 429" in response.json()["detail"]
