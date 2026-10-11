import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class BackendError(Exception):
    """The backend was unreachable or rejected the request. The message is safe to show users."""


def _request(method: str, path: str, timeout: float, **kwargs) -> requests.Response:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {BACKEND_URL}.") from e

    if response.ok:
        return response

    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if not isinstance(detail, str):
        detail = f"Backend returned HTTP {response.status_code}."
    raise BackendError(detail)


def get_health() -> dict:
    return _request("GET", "/health", timeout=3).json()


def analyze(filename: str, content: bytes, job_description: str) -> dict:
    return _request(
        "POST",
        "/api/v1/analyze",
        timeout=120,
        files={"resume": (filename, content)},
        data={"job_description": job_description},
    ).json()


def rank(resumes: list[tuple[str, bytes]], job_description: str) -> dict:
    """Ranks (filename, content) pairs against one job description."""
    return _request(
        "POST",
        "/api/v1/rank",
        timeout=600,
        files=[("resumes", resume) for resume in resumes],
        data={"job_description": job_description},
    ).json()


def get_history() -> list[dict]:
    return _request("GET", "/api/v1/history", timeout=10).json()["analyses"]


def get_analysis(analysis_id: int) -> dict:
    return _request("GET", f"/api/v1/history/{analysis_id}", timeout=10).json()


def delete_analysis(analysis_id: int) -> None:
    _request("DELETE", f"/api/v1/history/{analysis_id}", timeout=10)


def get_report(analysis_id: int) -> bytes:
    return _request("GET", f"/api/v1/history/{analysis_id}/report", timeout=30).content


def get_learning_path(missing_skills: list[str], job_title: str) -> list[dict]:
    return _request(
        "POST",
        "/api/v1/coach/learning-path",
        timeout=120,
        json={"missing_skills": missing_skills, "job_title": job_title},
    ).json()["recommendations"]


def get_rewrites(statements: list[str], job_title: str) -> list[dict]:
    return _request(
        "POST",
        "/api/v1/coach/rewrite",
        timeout=120,
        json={"statements": statements, "job_title": job_title},
    ).json()["rewrites"]
