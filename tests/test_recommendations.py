import json
from unittest.mock import Mock

import pytest
import requests

from core import recommendations
from core.recommendations import RecommendationError, get_ai_recommendations


def gemini_response(payload, status_code=200):
    response = Mock(status_code=status_code)
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    return response


@pytest.fixture
def post(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(recommendations.requests, "post", mock)
    return mock


def test_no_skills_makes_no_request(post):
    assert get_ai_recommendations([], "key") == []
    post.assert_not_called()


def test_returns_recommendations_with_search_links(post):
    post.return_value = gemini_response(
        [{"skill": "SQL", "resource_name": "Coursera: SQL Basics", "project": "Build a report"}]
    )

    recs = get_ai_recommendations(["SQL"], "key")

    assert recs == [
        {
            "skill": "SQL",
            "resource_name": "Coursera: SQL Basics",
            "project": "Build a report",
            "link": "https://www.google.com/search?q=Coursera%3A%20SQL%20Basics",
        }
    ]


def test_api_key_is_sent_in_header_not_url(post):
    post.return_value = gemini_response([])

    get_ai_recommendations(["SQL"], "secret-key")

    url = post.call_args.args[0]
    assert "secret-key" not in url
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "secret-key"


def test_only_top_skills_are_sent(post):
    post.return_value = gemini_response([])

    get_ai_recommendations(["a1", "b2", "c3", "d4", "e5"], "key")

    prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
    assert "a1, b2, c3." in prompt
    assert "d4" not in prompt


def test_falls_back_to_next_model_on_failure(post):
    post.side_effect = [
        gemini_response([], status_code=404),
        requests.ConnectionError("boom"),
        gemini_response([{"skill": "SQL", "resource_name": "Course", "project": "Project"}]),
    ]

    recs = get_ai_recommendations(["SQL"], "key")

    assert post.call_count == 3
    assert recs[0]["skill"] == "SQL"


def test_raises_when_every_model_fails(post):
    post.return_value = gemini_response([], status_code=429)

    with pytest.raises(RecommendationError, match="HTTP 429"):
        get_ai_recommendations(["SQL"], "key")


def test_malformed_model_output_counts_as_failure(post):
    response = Mock(status_code=200)
    response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
    post.return_value = response

    with pytest.raises(RecommendationError):
        get_ai_recommendations(["SQL"], "key")


def test_null_fields_are_dropped(post):
    post.return_value = gemini_response([{"skill": "SQL", "resource_name": None, "project": None}])

    recs = get_ai_recommendations(["SQL"], "key")

    assert recs == [{"skill": "SQL", "link": "https://www.google.com/search?q=SQL"}]
