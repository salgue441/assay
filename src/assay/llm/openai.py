"""
Chat completions from the OpenAI API.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from assay.llm.base import ChatRequest, ChatResponse, LLMError, PromptTooLongError
from assay.llm.pricing import TokenPrices, cost_usd

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000
_TRANSIENT_STATUS = frozenset({408, 409, 429})
_CONTEXT_LENGTH_CODE = "context_length_exceeded"
_QUOTA_CODE = "insufficient_quota"

# Strict structured outputs reject these keywords. They only bound sizes and
# counts, so they are dropped and the caller enforces them when validating.
_STRICT_DROPPED = frozenset(
    {"minLength", "maxLength", "minProperties", "maxProperties", "uniqueItems"}
)
# These change what a reply may look like, so a schema using them is sent
# without strict mode rather than silently loosened.
_STRICT_BLOCKERS = frozenset(
    {
        "allOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "patternProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
    }
)
_SCHEMA_MAPS = frozenset({"properties", "$defs", "definitions"})
_SCHEMA_LISTS = frozenset({"anyOf", "prefixItems"})


class OpenAIClient:
    """
    One OpenAI model through the Chat Completions API.

    The API key comes from `client`, or the SDK's own `OPENAI_API_KEY`
    environment variable. `ChatRequest.seed` is passed on, but OpenAI only
    makes a best effort to honour it.

    The requested model id is recorded as the revision, so use dated
    snapshot ids (`gpt-4o-2024-08-06`): an alias such as `gpt-4o` is
    re-pointed to new snapshots over time. Each response names the snapshot
    that served it; a warning is logged when it differs from the id.

    `cost_usd` comes from `prices`; without them it is 0.0, the spend is
    unknown, and a run budget cannot cap it.

    The SDK's own retries are turned off: `RetryingClient` retries transient
    errors for every provider alike.
    """

    def __init__(
        self,
        model_id: str,
        *,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        prices: TokenPrices | None = None,
        timeout_seconds: float = 300,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._model_id = model_id
        self._context_tokens = context_tokens
        self._prices = prices
        self._served: set[str] = set()
        if client is None:
            with self._translated_errors():
                client = openai.OpenAI(timeout=timeout_seconds, max_retries=0)
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def context_tokens(self) -> int:
        return self._context_tokens

    @property
    def model_revision(self) -> str:
        """The requested model id; see the class notes on snapshot ids."""

        return self._model_id

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Raises `PromptTooLongError` without calling OpenAI if the prompt and
        reply budget may not fit `context_tokens`.

        With `request.json_schema`, the reply is constrained by a JSON schema
        response format. It is strict when the schema allows: every object
        closed with all properties required. Size and count keywords strict
        mode rejects (`maxLength`, `minLength`, ...) are dropped, so callers
        must still validate the reply. A schema strict mode cannot express is
        sent as is without strict mode, as guidance the model may not follow.
        """

        if request.estimated_tokens() > self._context_tokens:
            msg = (
                f"openai {self._model_id}: prompt and reply need about "
                f"{request.estimated_tokens()} tokens; the context is {self._context_tokens}"
            )
            raise PromptTooLongError(msg)

        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": request.user}]
        if request.system is not None:
            messages.insert(0, {"role": "system", "content": request.system})
        options: dict[str, Any] = {}
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_output_tokens is not None:
            options["max_completion_tokens"] = request.max_output_tokens
        if request.json_schema is not None:
            options["response_format"] = response_format(request.json_schema)

        started = time.perf_counter()
        with self._translated_errors():
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=request.temperature,
                **options,
            )
        return self._response(completion, time.perf_counter() - started)

    def _response(self, completion: ChatCompletion, latency_seconds: float) -> ChatResponse:
        self._note_served_model(completion.model)
        if not completion.choices:
            msg = f"openai {self._model_id}: the response has no choices"
            raise LLMError(msg, transient=False)
        choice = completion.choices[0]
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        output_tokens = usage.completion_tokens if usage is not None else 0
        return ChatResponse(
            # A refusal comes instead of content; callers see it as the reply.
            text=choice.message.content or choice.message.refusal or "",
            model_id=self._model_id,
            model_revision=self.model_revision,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            cost_usd=cost_usd(self._prices, prompt_tokens, output_tokens),
            truncated=choice.finish_reason == "length",
        )

    def _note_served_model(self, served: str) -> None:
        if served and served != self._model_id and served not in self._served:
            self._served.add(served)
            logger.warning(
                "openai %s was served by %s; pin a dated snapshot id so results stay comparable",
                self._model_id,
                served,
            )

    @contextmanager
    def _translated_errors(self) -> Iterator[None]:
        """Turn OpenAI SDK failures into `LLMError`s, marking transient ones."""

        try:
            yield
        except openai.APIConnectionError as exc:
            # Includes APITimeoutError.
            msg = f"openai {self._model_id}: API unreachable ({exc})"
            raise LLMError(msg, transient=True) from exc
        except openai.APIStatusError as exc:
            msg = f"openai {self._model_id}: {_error_message(exc)} (HTTP {exc.status_code})"
            if exc.code == _CONTEXT_LENGTH_CODE:
                raise PromptTooLongError(msg) from exc
            raise LLMError(msg, transient=_is_transient(exc)) from exc
        except openai.OpenAIError as exc:
            # A missing API key, or a response the SDK could not read.
            msg = f"openai {self._model_id}: {exc}"
            raise LLMError(msg, transient=False) from exc


def _error_message(exc: openai.APIStatusError) -> str:
    """The API's own error message, without the SDK's restatement of the body."""

    body = exc.body
    if isinstance(body, Mapping) and isinstance(body.get("message"), str):
        return str(body["message"])
    return exc.message


def _is_transient(exc: openai.APIStatusError) -> bool:
    """Rate limits, conflicts, timeouts and server faults; an exhausted quota is not."""

    if exc.code == _QUOTA_CODE:
        return False
    return exc.status_code in _TRANSIENT_STATUS or exc.status_code >= 500


def response_format(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The `response_format` that constrains a reply to `schema`, strict when possible."""

    format_: dict[str, Any] = {"name": _schema_name(schema)}
    try:
        if schema.get("type") != "object":
            raise _NotStrictError  # strict mode needs an object at the root
        format_.update(schema=_strict_schema(schema), strict=True)
    except _NotStrictError:
        format_.update(schema=dict(schema), strict=False)
    return {"type": "json_schema", "json_schema": format_}


class _NotStrictError(Exception):
    """The schema uses something strict mode cannot express."""


def _strict_schema(node: Any) -> Any:
    if not isinstance(node, Mapping):
        return node
    if _STRICT_BLOCKERS & node.keys():
        raise _NotStrictError
    if node.get("type") == "object" or "properties" in node:
        closed = node.get("additionalProperties") is False
        if not closed or set(node.get("required", ())) != set(node.get("properties", {})):
            raise _NotStrictError
    stripped: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRICT_DROPPED:
            continue
        if key in _SCHEMA_MAPS:
            stripped[key] = {name: _strict_schema(sub) for name, sub in value.items()}
        elif key in _SCHEMA_LISTS:
            stripped[key] = [_strict_schema(sub) for sub in value]
        elif key == "items":
            stripped[key] = _strict_schema(value)
        else:
            stripped[key] = value
    return stripped


def _schema_name(schema: Mapping[str, Any]) -> str:
    title = str(schema.get("title", ""))
    return re.sub(r"[^A-Za-z0-9_-]", "_", title)[:64] or "reply"
