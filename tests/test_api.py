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
def test_analyze_returns_score_breakdown_and_explanations(client, filename, content):
    response = post_analyze(client, filename, content)

    assert response.status_code == 200
    body = response.json()
    assert 0 < body["match_score"] <= 100
    assert [c["key"] for c in body["components"]] == [
        "skills",
        "coverage",
        "experience",
        "education",
    ]
    assert {s["name"]: s["status"] for s in body["skills"]} == {
        "Python": "matched",
        "SQL": "matched",
        "Docker": "matched",
        "Kubernetes": "missing",
    }
    assert body["requirements"][0]["evidence"] == RESUME
    assert body["top_sentences"][0]["text"] == RESUME
    assert body["sections_found"] == []


def test_analyze_reports_sections_and_table_content_from_docx(client):
    content = make_docx(
        "Experience", RESUME, "Education", "B.Sc. in Statistics", table=[["Skills"], ["Kubernetes"]]
    )

    body = post_analyze(client, "resume.docx", content).json()

    assert body["sections_found"] == ["experience", "education", "skills"]
    assert {s["name"]: s["status"] for s in body["skills"]}["Kubernetes"] == "matched"


def test_analyze_includes_ats_checks_and_metadata(client):
    body = post_analyze(client, "resume.docx", make_docx(RESUME), f"Data Engineer\n{JD}").json()

    assert body["id"] == 1
    assert body["resume_filename"] == "resume.docx"
    assert body["job_title"] == "Data Engineer"
    assert 0 <= body["ats"]["score"] <= 100
    assert {c["key"]: c["status"] for c in body["ats"]["checks"]}["email"] == "fail"


def test_long_first_line_of_job_description_is_trimmed_for_the_title(client):
    body = post_analyze(client, "resume.docx", make_docx(RESUME), "Python and SQL " * 20).json()

    assert len(body["job_title"]) == 80
    assert body["job_title"].endswith("…")


def test_analyses_are_saved_to_history(client):
    post_analyze(client, "v1.docx", make_docx("I write Python every day at work."))
    latest = post_analyze(client, "v2.docx", make_docx(RESUME)).json()

    analyses = client.get("/api/v1/history").json()["analyses"]

    assert [a["resume_filename"] for a in analyses] == ["v2.docx", "v1.docx"]
    assert analyses[0]["match_score"] == latest["match_score"]
    assert client.get(f"/api/v1/history/{latest['id']}").json() == latest


def test_history_item_can_be_deleted(client):
    analysis_id = post_analyze(client, "resume.docx", make_docx(RESUME)).json()["id"]

    assert client.delete(f"/api/v1/history/{analysis_id}").status_code == 204
    assert client.get(f"/api/v1/history/{analysis_id}").status_code == 404
    assert client.delete(f"/api/v1/history/{analysis_id}").status_code == 404
    assert client.get("/api/v1/history").json() == {"analyses": []}


def test_report_is_a_pdf_download(client):
    analysis_id = post_analyze(client, "resume.docx", make_docx(RESUME)).json()["id"]

    response = client.get(f"/api/v1/history/{analysis_id}/report")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "resume-report-1.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")


def test_report_for_unknown_analysis_is_404(client):
    assert client.get("/api/v1/history/999/report").status_code == 404


def post_rank(client, resumes, job_description=JD):
    return client.post(
        "/api/v1/rank",
        files=[("resumes", resume) for resume in resumes],
        data={"job_description": job_description},
    )


def test_rank_orders_candidates_best_first(client):
    response = post_rank(
        client,
        [
            ("marketer.docx", make_docx("I ran marketing and sales campaigns for retail brands.")),
            ("engineer.docx", make_docx(RESUME)),
            ("junior.docx", make_docx("I write Python scripts for small automation tasks.")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert [(c["rank"], c["filename"]) for c in body["candidates"]] == [
        (1, "engineer.docx"),
        (2, "junior.docx"),
        (3, "marketer.docx"),
    ]
    best = body["candidates"][0]
    assert best["matched_skills"] == ["Python", "SQL", "Docker"]
    assert best["missing_skills"] == ["Kubernetes"]
    assert body["failed"] == []


def test_rank_reports_unreadable_files_without_failing_the_batch(client):
    body = post_rank(
        client,
        [("good.docx", make_docx(RESUME)), ("bad.pdf", b"not a pdf"), ("notes.txt", b"text")],
    ).json()

    assert [c["filename"] for c in body["candidates"]] == ["good.docx"]
    assert [f["filename"] for f in body["failed"]] == ["bad.pdf", "notes.txt"]
    assert "Could not read" in body["failed"][0]["error"]


def test_rank_does_not_write_to_history(client):
    post_rank(client, [("good.docx", make_docx(RESUME))])

    assert client.get("/api/v1/history").json() == {"analyses": []}


def test_rank_limits_batch_size(client, monkeypatch):
    monkeypatch.setenv("MAX_BATCH_FILES", "2")
    get_settings.cache_clear()

    response = post_rank(client, [(f"{i}.docx", make_docx(RESUME)) for i in range(3)])

    assert response.status_code == 413


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
