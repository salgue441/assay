import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest

from assay.judge import REPLY_SCHEMA
from assay.llm import ChatRequest, LLMError, PromptTooLongError, TokenPrices
from assay.llm.openai import OpenAIClient, response_format

MODEL = "gpt-4o-mini-2024-07-18"


def completion(
    content: str | None = "## `f`",
    *,
    finish_reason: str = "stop",
    model: str = MODEL,
    tokens: tuple[int, int] = (120, 45),
    refusal: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_792_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    }


def api_error(status: int, code: str | None = None, message: str = "nope") -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": "error", "code": code}}
    )


def serve(
    reply: Callable[[dict[str, Any]], httpx.Response],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        sent.append(body)
        return reply(body)

    return handler, sent


def client_for(handler: Callable[[httpx.Request], httpx.Response], **options: Any) -> OpenAIClient:
    sdk = openai.OpenAI(
        api_key="test-key",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return OpenAIClient(MODEL, client=sdk, **options)


def test_sends_messages_and_options_and_reads_usage() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=completion()))
    request = ChatRequest(
        user="document f", system="be precise", temperature=0.2, seed=7, max_output_tokens=512
    )

    response = client_for(handler).chat(request)

    (body,) = sent
    assert body == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "document f"},
        ],
        "temperature": 0.2,
        "seed": 7,
        "max_completion_tokens": 512,
    }
    assert (response.text, response.prompt_tokens, response.output_tokens) == ("## `f`", 120, 45)
    assert (response.model_id, response.model_revision, response.cost_usd) == (MODEL, MODEL, 0.0)
    assert not response.truncated


def test_omits_unset_options() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=completion()))

    client_for(handler).chat(ChatRequest(user="x"))

    assert sent[0] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.0,
    }


def test_sends_the_judge_schema_strictly_without_size_keywords() -> None:
    handler, sent = serve(
        lambda _: httpx.Response(
            200, json=completion('{"issues": [], "rationale": "", "score": 4}')
        )
    )

    response = client_for(handler).chat(ChatRequest(user="grade", json_schema=REPLY_SCHEMA))

    sent_format = sent[0]["response_format"]
    assert sent_format["type"] == "json_schema"
    assert sent_format["json_schema"]["name"] == "JudgeReply"
    assert sent_format["json_schema"]["strict"] is True
    schema = sent_format["json_schema"]["schema"]
    assert schema["properties"]["rationale"] == {"title": "Rationale", "type": "string"}
    assert schema["properties"]["issues"]["items"] == {"type": "string"}
    assert schema["properties"]["issues"]["maxItems"] == 8
    assert schema["properties"]["score"]["maximum"] == 5
    assert json.loads(response.text)["score"] == 4


def test_keeps_size_keywords_of_the_callers_schema_intact() -> None:
    response_format(REPLY_SCHEMA)

    assert REPLY_SCHEMA["properties"]["rationale"]["maxLength"] == 800


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        {"type": "object", "properties": {}, "additionalProperties": False, "allOf": [{}]},
        {
            "type": "array",
            "items": {"type": "object", "properties": {"a": {"type": "integer"}}},
        },
    ],
    ids=["open-object", "optional-property", "allOf", "nested-open-object"],
)
def test_schemas_strict_mode_cannot_express_are_sent_unchanged_and_not_strict(
    schema: dict[str, Any],
) -> None:
    sent_format = response_format(schema)["json_schema"]

    assert sent_format == {"name": "reply", "schema": schema, "strict": False}


def test_strips_size_keywords_inside_definitions_and_unions() -> None:
    schema = {
        "title": "My reply!",
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "required": ["a"],
        "additionalProperties": False,
        "$defs": {"A": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}},
    }

    sent_format = response_format(schema)["json_schema"]

    assert sent_format["name"] == "My_reply_"
    assert sent_format["strict"] is True
    assert sent_format["schema"]["$defs"]["A"]["anyOf"][0] == {"type": "string"}


def test_flags_replies_cut_off_at_the_token_limit() -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=completion(finish_reason="length")))

    assert client_for(handler).chat(ChatRequest(user="x")).truncated


def test_returns_a_refusal_as_the_reply() -> None:
    handler, _ = serve(
        lambda _: httpx.Response(200, json=completion(None, refusal="I can't help with that."))
    )

    assert client_for(handler).chat(ChatRequest(user="x")).text == "I can't help with that."


def test_prices_the_reported_usage() -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=completion(tokens=(1_000, 200))))
    prices = TokenPrices(input_per_million=0.15, output_per_million=0.6)

    response = client_for(handler, prices=prices).chat(ChatRequest(user="x"))

    assert response.cost_usd == pytest.approx(0.00015 + 0.00012)


def test_warns_once_when_an_alias_is_served_by_another_snapshot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=completion(model="gpt-4o-2099-01-01")))
    client = client_for(handler)

    with caplog.at_level(logging.WARNING, logger="assay.llm.openai"):
        response = client.chat(ChatRequest(user="x"))
        client.chat(ChatRequest(user="x"))

    assert response.model_revision == MODEL
    assert [r.getMessage() for r in caplog.records] == [
        f"openai {MODEL} was served by gpt-4o-2099-01-01; "
        "pin a dated snapshot id so results stay comparable"
    ]


@pytest.mark.parametrize(
    ("status", "code", "transient"),
    [
        (429, "rate_limit_exceeded", True),
        (500, None, True),
        (503, None, True),
        (408, None, True),
        (429, "insufficient_quota", False),
        (400, None, False),
        (401, "invalid_api_key", False),
        (403, None, False),
        (404, "model_not_found", False),
    ],
)
def test_http_errors_are_classified(status: int, code: str | None, transient: bool) -> None:
    handler, _ = serve(lambda _: api_error(status, code))

    with pytest.raises(LLMError, match=f"nope \\(HTTP {status}\\)") as excinfo:
        client_for(handler).chat(ChatRequest(user="x"))

    assert type(excinfo.value) is LLMError
    assert excinfo.value.transient is transient


def test_a_context_overflow_reported_by_openai_is_a_prompt_too_long_error() -> None:
    handler, _ = serve(
        lambda _: api_error(400, "context_length_exceeded", "maximum context length is 128000")
    )

    with pytest.raises(PromptTooLongError, match="maximum context length"):
        client_for(handler).chat(ChatRequest(user="x"))


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (httpx.ReadTimeout, "timed out"),
        (httpx.ConnectError, "Connection error"),
    ],
)
def test_timeouts_and_unreachable_apis_are_transient(
    error: type[httpx.TransportError], match: str
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise error("down", request=request)

    with pytest.raises(LLMError, match=f"API unreachable \\(.*{match}") as excinfo:
        client_for(fail).chat(ChatRequest(user="x"))

    assert excinfo.value.transient


def test_a_response_without_choices_is_a_permanent_error() -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=completion() | {"choices": []}))

    with pytest.raises(LLMError, match="no choices") as excinfo:
        client_for(handler).chat(ChatRequest(user="x"))

    assert not excinfo.value.transient


def test_refuses_a_prompt_that_would_overflow_the_context_without_sending_it() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=completion()))
    client = client_for(handler, context_tokens=1000)

    with pytest.raises(PromptTooLongError, match="the context is 1000") as excinfo:
        client.chat(ChatRequest(user="x" * 3000, max_output_tokens=10))

    assert not excinfo.value.transient
    assert sent == []
    assert client.context_tokens == 1000


def test_a_missing_api_key_is_a_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY") as excinfo:
        OpenAIClient(MODEL)

    assert not excinfo.value.transient


def test_builds_an_sdk_client_from_the_environment_without_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    sdk = OpenAIClient(MODEL, timeout_seconds=42)._client

    assert (sdk.api_key, sdk.max_retries, sdk.timeout) == ("test-key", 0, 42)
