import json
import logging

import requests

from core.config import get_settings

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    """No configured model returned a usable answer."""


def complete_json(system: str, user: str) -> dict:
    """Asks the LLM a question and returns its answer as a JSON object.

    Tries each configured model in turn, so a retired or rate-limited model falls
    through to the next one.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set")

    last_error = "no models configured"
    for model in settings.groq_model_list:
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.4,
                },
                timeout=settings.llm_timeout_seconds,
            )
            if response.status_code != 200:
                last_error = f"{model}: HTTP {response.status_code}"
                logger.warning("LLM call failed (%s)", last_error)
                continue

            answer = json.loads(response.json()["choices"][0]["message"]["content"])
            if isinstance(answer, dict):
                return answer
            last_error = f"{model}: answer was not a JSON object"
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
            last_error = f"{model}: {type(e).__name__}"
        logger.warning("LLM call failed (%s)", last_error)

    raise LLMError(last_error)
