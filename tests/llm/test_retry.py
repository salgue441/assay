from dataclasses import dataclass, field

import pytest

from assay.llm import ChatRequest, ChatResponse, LLMError, RetryingClient

REPLY = ChatResponse(
    text="ok",
    model_id="fake",
    model_revision="r1",
    prompt_tokens=1,
    output_tokens=1,
    latency_seconds=0.0,
    cost_usd=0.0,
    truncated=False,
)


@dataclass
class Flaky:
    """Fails with the given errors in order, then succeeds."""

    errors: list[LLMError]
    calls: int = field(default=0)

    @property
    def model_id(self) -> str:
        return "fake"

    context_tokens = 4096

    @property
    def model_revision(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "r1"

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return REPLY


def retrying(inner: Flaky, attempts: int = 3) -> RetryingClient:
    return RetryingClient(inner, attempts=attempts, initial_wait_seconds=0, max_wait_seconds=0)


def test_retries_transient_errors_until_success() -> None:
    inner = Flaky([LLMError("busy", transient=True), LLMError("busy", transient=True)])

    assert retrying(inner).chat(ChatRequest(user="x")) is REPLY
    assert inner.calls == 3


def test_gives_up_after_the_last_attempt_with_the_original_error() -> None:
    inner = Flaky([LLMError(f"busy {i}", transient=True) for i in range(5)])

    with pytest.raises(LLMError, match="busy 2"):
        retrying(inner, attempts=3).chat(ChatRequest(user="x"))
    assert inner.calls == 3


def test_does_not_retry_permanent_errors() -> None:
    inner = Flaky([LLMError("no such model", transient=False)])

    with pytest.raises(LLMError, match="no such model"):
        retrying(inner).chat(ChatRequest(user="x"))
    assert inner.calls == 1


def test_exposes_the_inner_model_id_and_context() -> None:
    client = retrying(Flaky([]))

    assert (client.model_id, client.context_tokens) == ("fake", 4096)


def test_retries_a_transient_failure_to_look_up_the_revision() -> None:
    inner = Flaky([LLMError("server starting", transient=True)])

    assert retrying(inner).model_revision == "r1"
    assert inner.calls == 2


def test_requires_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetryingClient(Flaky([]), attempts=0)
