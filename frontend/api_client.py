import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class BackendError(Exception):
    """The backend was unreachable or rejected the request. The message is safe to show users."""


def _request(method: str, path: str, timeout: float, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {BACKEND_URL}.") from e

    if response.ok:
        return response.json()

    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if not isinstance(detail, str):
        detail = f"Backend returned HTTP {response.status_code}."
    raise BackendError(detail)


def get_health() -> dict:
    return _request("GET", "/health", timeout=3)


def analyze(filename: str, content: bytes, job_description: str) -> dict:
    return _request(
        "POST",
        "/api/v1/analyze",
        timeout=120,
        files={"resume": (filename, content)},
        data={"job_description": job_description},
    )


def get_recommendations(missing_skills: list[str], api_key: str | None = None) -> list[dict]:
    headers = {"X-Gemini-Api-Key": api_key} if api_key else {}
    body = _request(
        "POST",
        "/api/v1/recommendations",
        timeout=120,
        json={"missing_skills": missing_skills},
        headers=headers,
    )
    return body["recommendations"]
