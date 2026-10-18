import os
import threading

import requests

# "inprocess" runs the API inside this process instead of calling it over HTTP, for hosts
# that run a single Streamlit process (see deploy/streamlit)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
IN_PROCESS = "inprocess"

# Streamlit runs each browser session's script in its own thread, so the session id is
# kept per thread; a plain module variable would be shared by every visitor.
_local = threading.local()


_in_process_lock = threading.Lock()


def _in_process_client():
    """A client that calls the FastAPI app directly, created on first use."""
    with _in_process_lock:
        if not hasattr(_in_process_client, "client"):
            from fastapi.testclient import TestClient

            from backend.main import app

            # Server errors come back as 500 responses, as they would over HTTP
            _in_process_client.client = TestClient(app, raise_server_exceptions=False)
        return _in_process_client.client


def use_session(session_id: str) -> None:
    """Sends this id with every request made from the current script run."""
    _local.session_id = session_id


class BackendError(Exception):
    """The backend was unreachable or rejected the request. The message is safe to show users."""


def _request(method: str, path: str, timeout: float, **kwargs):
    session_id = getattr(_local, "session_id", None)
    if session_id:
        kwargs["headers"] = {**kwargs.get("headers", {}), "X-Session-Id": session_id}
    try:
        if BACKEND_URL == IN_PROCESS:
            response = _in_process_client().request(method, path, timeout=timeout, **kwargs)
        else:
            response = requests.request(
                method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs
            )
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {BACKEND_URL}.") from e

    if response.status_code < 400:
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


def get_report(share_token: str) -> bytes:
    return _request("GET", f"/api/v1/shared/{share_token}/report", timeout=30).content


def get_shared(share_token: str) -> dict:
    return _request("GET", f"/api/v1/shared/{share_token}", timeout=10).json()


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
