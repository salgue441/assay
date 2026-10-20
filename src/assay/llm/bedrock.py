"""
Chat completions from Amazon Bedrock through the Converse API.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

import boto3
import botocore.config
import botocore.exceptions

from assay.llm.base import ChatRequest, ChatResponse, LLMError, PromptTooLongError
from assay.llm.pricing import TokenPrices, cost_usd

DEFAULT_CONTEXT_TOKENS = 128_000
REPLY_TOOL_NAME = "reply"
"""The one tool a structured request forces the model to call."""

# Overload, throttling and service faults pass; ModelNotReadyException is a
# model still loading (AWS documents it as retryable).
_TRANSIENT_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
        "ModelNotReadyException",
    }
)
_PROMPT_TOO_LONG = re.compile(r"too long|context (length|window)|maximum.*tokens", re.IGNORECASE)


class ConverseClient(Protocol):
    """The part of a boto3 `bedrock-runtime` client this module uses."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...


class BedrockClient:
    """
    One Bedrock model or inference profile.

    Converse takes no seed, so `ChatRequest.seed` is ignored and Bedrock runs
    are not seed-reproducible: repeated calls at temperature 0 can differ.

    Bedrock model ids carry a version (`anthropic.claude-...-v1:0`), so the id
    is recorded as the revision. Use versioned ids: an unversioned alias could
    be re-pointed to different weights behind the same cache key.

    `cost_usd` comes from `prices`; without them it is 0.0, the spend is
    unknown, and a run budget cannot cap it.

    The AWS SDK's own retries are turned off: `RetryingClient` retries
    transient errors for every provider alike.
    """

    def __init__(
        self,
        model_id: str,
        *,
        region: str | None = None,
        profile: str | None = None,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        prices: TokenPrices | None = None,
        timeout_seconds: float = 300,
        client: ConverseClient | None = None,
    ) -> None:
        self._model_id = model_id
        self._context_tokens = context_tokens
        self._prices = prices
        if client is None:
            with self._translated_errors():
                client = _runtime_client(region, profile, timeout_seconds)
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    @property
    def model_revision(self) -> str:
        """The model id itself: Bedrock ids are versioned."""

        return self._model_id

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Raises `PromptTooLongError` without calling Bedrock if the prompt and
        reply budget may not fit `context_tokens`.

        With `request.json_schema`, the model is forced to call one tool whose
        input schema is that schema, and the tool input is returned as JSON
        text, so callers parse it like any other reply. The model must support
        forcing a specific tool (Anthropic Claude, Amazon Nova and Mistral
        Large do; check the Bedrock model's tool-use support).
        """

        if request.estimated_tokens() > self._context_tokens:
            msg = (
                f"bedrock {self._model_id}: prompt and reply need about "
                f"{request.estimated_tokens()} tokens; the context is {self._context_tokens}"
            )
            raise PromptTooLongError(msg)

        started = time.perf_counter()
        with self._translated_errors():
            response = self._client.converse(**self._converse_arguments(request))
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("inputTokens", 0))
        output_tokens = int(usage.get("outputTokens", 0))
        return ChatResponse(
            text=_reply_text(response),
            model_id=self._model_id,
            model_revision=self.model_revision,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
            cost_usd=cost_usd(self._prices, prompt_tokens, output_tokens),
            truncated=response.get("stopReason") == "max_tokens",
        )

    def _converse_arguments(self, request: ChatRequest) -> dict[str, Any]:
        inference: dict[str, Any] = {"temperature": request.temperature}
        if request.max_output_tokens is not None:
            inference["maxTokens"] = request.max_output_tokens
        arguments: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": request.user}]}],
            "inferenceConfig": inference,
        }
        if request.system is not None:
            arguments["system"] = [{"text": request.system}]
        if request.json_schema is not None:
            arguments["toolConfig"] = _forced_tool(request.json_schema)
        return arguments

    @contextmanager
    def _translated_errors(self) -> Iterator[None]:
        """Turn AWS SDK failures into `LLMError`s, marking transient ones."""

        try:
            yield
        except botocore.exceptions.ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "Unknown"))
            message = str(error.get("Message", exc))
            text = f"bedrock {self._model_id}: {code}: {message}"
            if code == "ValidationException" and _PROMPT_TOO_LONG.search(message):
                raise PromptTooLongError(text) from exc
            raise LLMError(text, transient=code in _TRANSIENT_CODES) from exc
        except (botocore.exceptions.ConnectionError, botocore.exceptions.HTTPClientError) as exc:
            # Connect and read timeouts, refused or dropped connections.
            msg = f"bedrock {self._model_id}: endpoint unreachable ({exc})"
            raise LLMError(msg, transient=True) from exc
        except botocore.exceptions.BotoCoreError as exc:
            # Missing credentials or region, unknown profile, invalid parameters.
            msg = f"bedrock {self._model_id}: {exc}"
            raise LLMError(msg, transient=False) from exc


def _runtime_client(region: str | None, profile: str | None, timeout_seconds: float) -> Any:
    session = boto3.Session(profile_name=profile, region_name=region)
    config = botocore.config.Config(
        read_timeout=timeout_seconds,
        retries={"total_max_attempts": 1},
    )
    return session.client("bedrock-runtime", config=config)


def _forced_tool(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": REPLY_TOOL_NAME,
                    "description": "Give your reply. Its input is the reply.",
                    "inputSchema": {"json": dict(schema)},
                }
            }
        ],
        "toolChoice": {"tool": {"name": REPLY_TOOL_NAME}},
    }


def _reply_text(response: Mapping[str, Any]) -> str:
    """
    The forced tool's input as JSON if the model called it, otherwise the
    reply's text blocks joined. A reply cut off at the token limit may have
    neither.
    """

    content = response.get("output", {}).get("message", {}).get("content", [])
    for block in content:
        tool_use = block.get("toolUse")
        if tool_use is not None and tool_use.get("name") == REPLY_TOOL_NAME:
            return json.dumps(tool_use.get("input", {}))
    return "".join(block["text"] for block in content if "text" in block)
