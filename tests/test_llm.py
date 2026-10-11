import json
from unittest.mock import Mock

import pytest
import requests

from core import llm
from core.config import get_settings
from core.llm import LLMError, complete_json


def chat_response(content, status_code=200):
    response = Mock(status_code=status_code)
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


@pytest.fixture
def post(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    monkeypatch.setenv("GROQ_MODELS", "model-a,model-b")
    get_settings.cache_clear()
    mock = Mock()
    monkeypatch.setattr(llm.requests, "post", mock)
    yield mock
    get_settings.cache_clear()


def test_returns_the_parsed_json_object(post):
    post.return_value = chat_response(json.dumps({"answer": 42}))

    assert complete_json("system", "user") == {"answer": 42}


def test_sends_key_as_bearer_token_and_asks_for_json(post):
    post.return_value = chat_response("{}")

    complete_json("be helpful", "hello")

    request = post.call_args.kwargs
    assert request["headers"] == {"Authorization": "Bearer secret-key"}
    assert request["json"]["model"] == "model-a"
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["json"]["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]


def test_falls_back_to_the_next_model(post):
    post.side_effect = [chat_response("{}", status_code=429), chat_response('{"ok": true}')]

    assert complete_json("system", "user") == {"ok": True}
    assert [call.kwargs["json"]["model"] for call in post.call_args_list] == ["model-a", "model-b"]


@pytest.mark.parametrize(
    "failure",
    [
        chat_response("{}", status_code=500),
        chat_response("not json"),
        chat_response("[1, 2]"),
        requests.ConnectionError("boom"),
    ],
    ids=["http-error", "invalid-json", "not-an-object", "network-error"],
)
def test_raises_when_every_model_fails(post, failure):
    post.side_effect = [failure, failure]

    with pytest.raises(LLMError, match="model-b"):
        complete_json("system", "user")
    assert post.call_count == 2


def test_error_never_contains_the_api_key(post):
    post.side_effect = requests.ConnectionError("failed for Bearer secret-key")

    with pytest.raises(LLMError) as error:
        complete_json("system", "user")
    assert "secret-key" not in str(error.value)


def test_missing_key_fails_without_a_request(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(llm.requests, "post", mock)
    get_settings.cache_clear()

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        complete_json("system", "user")
    mock.assert_not_called()
