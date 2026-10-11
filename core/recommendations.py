import json
import logging
import urllib.parse

import requests

from core.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class RecommendationError(RuntimeError):
    """No Gemini model returned usable recommendations."""


def get_ai_recommendations(missing_skills: list[str], api_key: str) -> list[dict]:
    """Asks Gemini for a learning resource and project idea per missing skill.

    Tries each configured model in turn, so an outdated or rate-limited model
    falls through to the next one.
    """
    if not missing_skills:
        return []

    settings = get_settings()
    target_skills = missing_skills[: settings.max_recommendation_skills]
    skills_str = ", ".join(target_skills)

    # The key goes in a header, not the URL, so it stays out of logs and error messages
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                You are a career coach. A candidate is missing: {skills_str}.
                For each skill, provide:
                1. A specific resource name (e.g. "Coursera: Python for Everybody").
                2. A project idea.

                Return strictly JSON.
                Format: [{{"skill": "...", "resource_name": "...", "project": "..."}}]
                """
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
        },
    }

    last_error = "no models configured"
    for model_name in settings.gemini_model_list:
        try:
            response = requests.post(
                GEMINI_URL.format(model=model_name),
                headers=headers,
                json=payload,
                timeout=settings.gemini_timeout_seconds,
            )
            if response.status_code != 200:
                last_error = f"{model_name}: HTTP {response.status_code}"
                logger.warning("Gemini call failed (%s)", last_error)
                continue

            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            # Keep only well-formed entries, and drop null fields so API defaults apply
            recommendations = [
                {key: str(value) for key, value in rec.items() if value is not None}
                for rec in json.loads(text)
                if isinstance(rec, dict)
            ]
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
            last_error = f"{model_name}: {type(e).__name__}"
            logger.warning("Gemini call failed (%s)", last_error)
            continue

        # LLM-written URLs are often invented, so we link to a search for the resource instead
        for rec in recommendations:
            query = urllib.parse.quote(str(rec.get("resource_name") or rec.get("skill", "")))
            rec["link"] = f"https://www.google.com/search?q={query}"
        return recommendations

    raise RecommendationError(last_error)
