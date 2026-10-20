import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import ollama
import pytest

from assay.llm import ChatRequest, LLMError, OllamaClient, PromptTooLongError

DIGEST = "f72c60cabf6237b0" + "0" * 48


def tags(*names: str) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": n,
                "model": n,
                "digest": DIGEST,
                "size": 1,
                "modified_at": "2026-10-19T00:00:00Z",
            }
            for n in names
        ]
    }


def chat_reply(content: str = "## `f`", *, done_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": "qwen2.5-coder:3b",
        "created_at": "2026-10-19T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 120,
        "eval_count": 45,
    }


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], model: str = "qwen2.5-coder:3b"
) -> OllamaClient:
    return OllamaClient(
        model,
        client=ollama.Client(host="http://ollama.test", transport=httpx.MockTransport(handler)),
    )


def serve(
    chat: Callable[[dict[str, Any]], httpx.Response], *installed: str
) -> tuple[Callable[[httpx.Request], httpx.Response], list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags(*(installed or ("qwen2.5-coder:3b",))))
        body = json.loads(request.content)
        sent.append(body)
        return chat(body)

    return handler, sent


def test_sends_messages_and_options_and_reads_usage() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=chat_reply()))
    request = ChatRequest(
        user="document f", system="be precise", temperature=0.2, seed=7, max_output_tokens=512
    )

    response = client_for(handler).chat(request)

    (body,) = sent
    assert body["model"] == "qwen2.5-coder:3b"
    assert body["messages"] == [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "document f"},
    ]
    assert body["options"] == {"temperature": 0.2, "num_ctx": 8192, "seed": 7, "num_predict": 512}
    assert body["think"] is False
    assert (response.text, response.prompt_tokens, response.output_tokens) == ("## `f`", 120, 45)
    assert (response.model_id, response.model_revision, response.cost_usd) == (
        "qwen2.5-coder:3b",
        DIGEST,
        0.0,
    )
    assert not response.truncated


def test_passes_a_json_schema_as_the_format() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=chat_reply('{"score": 4}')))
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}

    client_for(handler).chat(ChatRequest(user="grade", json_schema=schema))

    assert sent[0]["format"] == schema


def test_flags_replies_cut_off_at_the_token_limit() -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=chat_reply(done_reason="length")))

    assert client_for(handler).chat(ChatRequest(user="x")).truncated


def test_matches_untagged_model_names_to_latest() -> None:
    handler, _ = serve(lambda _: httpx.Response(200, json=chat_reply()), "phi4-mini:latest")

    assert client_for(handler, "phi4-mini").model_revision == DIGEST


def test_a_model_that_is_not_pulled_is_a_permanent_error() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=chat_reply()), "other:1b")

    with pytest.raises(LLMError, match=re.escape("ollama pull qwen2.5-coder:3b")) as excinfo:
        client_for(handler).chat(ChatRequest(user="x"))

    assert not excinfo.value.transient
    assert sent == []


@pytest.mark.parametrize(("status", "transient"), [(503, True), (429, True), (400, False)])
def test_http_errors_are_classified(status: int, transient: bool) -> None:
    handler, _ = serve(lambda _: httpx.Response(status, json={"error": "nope"}))

    with pytest.raises(LLMError, match=f"HTTP {status}") as excinfo:
        client_for(handler).chat(ChatRequest(user="x"))

    assert excinfo.value.transient is transient


def test_an_unreachable_server_is_transient() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMError, match="server unreachable") as excinfo:
        client_for(refuse).chat(ChatRequest(user="x"))

    assert excinfo.value.transient


def test_refuses_a_prompt_that_would_overflow_the_context_without_sending_it() -> None:
    handler, sent = serve(lambda _: httpx.Response(200, json=chat_reply()))
    client = OllamaClient(
        "qwen2.5-coder:3b",
        context_tokens=1000,
        client=ollama.Client(host="http://ollama.test", transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(PromptTooLongError, match="the context is 1000") as excinfo:
        client.chat(ChatRequest(user="x" * 3000, max_output_tokens=10))

    assert not excinfo.value.transient
    assert sent == []
    assert client.context_tokens == 1000


@pytest.mark.llm
def test_real_server_round_trip() -> None:
    # Checks our side of the contract, not the model's wording.
    response = OllamaClient("phi4-mini").chat(
        ChatRequest(user="Reply with one word.", seed=1, max_output_tokens=5)
    )

    assert response.text.strip()
    assert response.prompt_tokens > 0
    assert 0 < response.output_tokens <= 5
    assert len(response.model_revision) == 64
