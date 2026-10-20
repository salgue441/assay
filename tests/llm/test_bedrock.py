import json
from collections.abc import Iterator
from typing import Any

import boto3
import botocore.exceptions
import pytest
from botocore.stub import Stubber

from assay.llm import ChatRequest, LLMError, PromptTooLongError, TokenPrices
from assay.llm.bedrock import REPLY_TOOL_NAME, BedrockClient

MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"


@pytest.fixture
def runtime() -> Any:
    # Fake credentials and a fixed region: the stubber answers before any request is signed
    # or sent, so nothing reaches AWS.
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.fixture
def stubber(runtime: Any) -> Iterator[Stubber]:
    with Stubber(runtime) as stub:
        yield stub
        stub.assert_no_pending_responses()


def converse_reply(
    *content: dict[str, Any], stop_reason: str = "end_turn", tokens: tuple[int, int] = (120, 45)
) -> dict[str, Any]:
    return {
        "output": {
            "message": {"role": "assistant", "content": list(content) or [{"text": "## `f`"}]}
        },
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": tokens[0],
            "outputTokens": tokens[1],
            "totalTokens": sum(tokens),
        },
        "metrics": {"latencyMs": 250},
    }


def test_sends_messages_and_inference_config_and_reads_usage(
    runtime: Any, stubber: Stubber
) -> None:
    stubber.add_response(
        "converse",
        converse_reply(),
        {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "document f"}]}],
            "system": [{"text": "be precise"}],
            "inferenceConfig": {"temperature": 0.2, "maxTokens": 512},
        },
    )
    request = ChatRequest(
        user="document f", system="be precise", temperature=0.2, seed=7, max_output_tokens=512
    )

    response = BedrockClient(MODEL, client=runtime).chat(request)

    assert (response.text, response.prompt_tokens, response.output_tokens) == ("## `f`", 120, 45)
    assert (response.model_id, response.model_revision, response.cost_usd) == (MODEL, MODEL, 0.0)
    assert not response.truncated
    assert response.latency_seconds >= 0


def test_omits_the_system_prompt_and_output_limit_when_unset(
    runtime: Any, stubber: Stubber
) -> None:
    stubber.add_response(
        "converse",
        converse_reply(),
        {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "x"}]}],
            "inferenceConfig": {"temperature": 0.0},
        },
    )

    BedrockClient(MODEL, client=runtime).chat(ChatRequest(user="x"))


def test_forces_a_tool_for_a_json_schema_and_returns_its_input_as_json(
    runtime: Any, stubber: Stubber
) -> None:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["score"],
        "additionalProperties": False,
    }
    tool_use = {"toolUse": {"toolUseId": "t1", "name": REPLY_TOOL_NAME, "input": {"score": 4}}}
    stubber.add_response(
        "converse",
        converse_reply({"text": "Let me grade."}, tool_use, stop_reason="tool_use"),
        {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "grade"}]}],
            "inferenceConfig": {"temperature": 0.0},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": REPLY_TOOL_NAME,
                            "description": "Give your reply. Its input is the reply.",
                            "inputSchema": {"json": schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": REPLY_TOOL_NAME}},
            },
        },
    )

    response = BedrockClient(MODEL, client=runtime).chat(
        ChatRequest(user="grade", json_schema=schema)
    )

    assert json.loads(response.text) == {"score": 4}
    assert not response.truncated


def test_flags_replies_cut_off_at_the_token_limit(runtime: Any, stubber: Stubber) -> None:
    stubber.add_response("converse", converse_reply({"text": "## "}, stop_reason="max_tokens"))

    response = BedrockClient(MODEL, client=runtime).chat(ChatRequest(user="x"))

    assert response.truncated
    assert response.text == "## "


def test_prices_the_reported_usage(runtime: Any, stubber: Stubber) -> None:
    stubber.add_response("converse", converse_reply(tokens=(1_000, 200)))
    prices = TokenPrices(input_per_million=0.8, output_per_million=4.0)

    response = BedrockClient(MODEL, prices=prices, client=runtime).chat(ChatRequest(user="x"))

    assert response.cost_usd == pytest.approx(0.0008 + 0.0008)


@pytest.mark.parametrize(
    ("code", "status", "transient"),
    [
        ("ThrottlingException", 429, True),
        ("ServiceUnavailableException", 503, True),
        ("ModelTimeoutException", 408, True),
        ("InternalServerException", 500, True),
        ("ModelNotReadyException", 429, True),
        ("ValidationException", 400, False),
        ("AccessDeniedException", 403, False),
        ("ResourceNotFoundException", 404, False),
        ("ModelErrorException", 424, False),
    ],
)
def test_service_errors_are_classified(
    runtime: Any, stubber: Stubber, code: str, status: int, transient: bool
) -> None:
    stubber.add_client_error(
        "converse", service_error_code=code, service_message="nope", http_status_code=status
    )

    with pytest.raises(LLMError, match=f"{code}: nope") as excinfo:
        BedrockClient(MODEL, client=runtime).chat(ChatRequest(user="x"))

    assert type(excinfo.value) is LLMError
    assert excinfo.value.transient is transient


def test_a_context_overflow_reported_by_bedrock_is_a_prompt_too_long_error(
    runtime: Any, stubber: Stubber
) -> None:
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="Input is too long for requested model.",
        http_status_code=400,
    )

    with pytest.raises(PromptTooLongError, match="too long"):
        BedrockClient(MODEL, client=runtime).chat(ChatRequest(user="x"))


class _FailingRuntime:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        botocore.exceptions.ReadTimeoutError(endpoint_url="https://bedrock.test"),
        botocore.exceptions.ConnectTimeoutError(endpoint_url="https://bedrock.test"),
        botocore.exceptions.EndpointConnectionError(endpoint_url="https://bedrock.test"),
    ],
)
def test_timeouts_and_unreachable_endpoints_are_transient(error: Exception) -> None:
    with pytest.raises(LLMError, match="endpoint unreachable") as excinfo:
        BedrockClient(MODEL, client=_FailingRuntime(error)).chat(ChatRequest(user="x"))

    assert excinfo.value.transient


def test_missing_credentials_are_permanent() -> None:
    failing = _FailingRuntime(botocore.exceptions.NoCredentialsError())

    with pytest.raises(LLMError, match="Unable to locate credentials") as excinfo:
        BedrockClient(MODEL, client=failing).chat(ChatRequest(user="x"))

    assert not excinfo.value.transient


def test_refuses_a_prompt_that_would_overflow_the_context_without_sending_it(
    runtime: Any, stubber: Stubber
) -> None:
    client = BedrockClient(MODEL, context_tokens=1000, client=runtime)

    with pytest.raises(PromptTooLongError, match="the context is 1000") as excinfo:
        client.chat(ChatRequest(user="x" * 3000, max_output_tokens=10))

    assert not excinfo.value.transient
    assert client.context_tokens == 1000


def test_builds_a_runtime_client_for_the_region_without_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_CONFIG_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "does-not-exist")

    client = BedrockClient(MODEL, region="eu-west-1", timeout_seconds=42)

    runtime = client._client
    assert runtime.meta.region_name == "eu-west-1"
    assert runtime.meta.config.read_timeout == 42
    assert runtime.meta.config.retries["total_max_attempts"] == 1


def test_an_unknown_profile_is_a_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_CONFIG_FILE", "does-not-exist")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "does-not-exist")

    with pytest.raises(LLMError, match="no-such-profile") as excinfo:
        BedrockClient(MODEL, region="us-east-1", profile="no-such-profile")

    assert not excinfo.value.transient
