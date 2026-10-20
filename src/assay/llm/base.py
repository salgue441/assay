"""
The provider-neutral contract for chat completions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

CHARS_PER_TOKEN_FLOOR = 3.0
"""A low bound on characters per token for code and English prose. Pilot
prompts measured 3.8 to 5.0 for both subject models, so dividing by 3.0
overestimates their token counts by at least a quarter."""

_MESSAGE_OVERHEAD_TOKENS = 16


def estimate_tokens(text: str) -> int:
    """A conservative (high) token count for `text`, without a tokenizer."""

    return math.ceil(len(text) / CHARS_PER_TOKEN_FLOOR)


class LLMError(RuntimeError):
    """
    A failed completion. `transient` errors (timeouts, overload, a server
    that is still starting) may succeed if retried; others will not.
    """

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class PromptTooLongError(LLMError):
    """
    The prompt would not fit the model's context alongside the reply. Some
    providers silently drop the start of an oversized prompt, instructions
    included, so it is refused before sending.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, transient=False)


@dataclass(frozen=True, slots=True)
class ChatRequest:
    user: str
    system: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    max_output_tokens: int | None = None
    json_schema: Mapping[str, Any] | None = field(default=None, hash=False)
    """Constrain the reply to this JSON schema, where the provider supports it."""

    def estimated_tokens(self) -> int:
        """A conservative token count for the prompt plus the reply budget."""

        messages = [self.user] if self.system is None else [self.system, self.user]
        prompt = sum(estimate_tokens(m) + _MESSAGE_OVERHEAD_TOKENS for m in messages)
        return prompt + (self.max_output_tokens or 0)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    model_id: str
    model_revision: str
    """Exact weights used, such as a digest; a model id alone can be re-pointed."""
    prompt_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    truncated: bool
    """The reply stopped at the output token limit."""


class LLMClient(Protocol):
    """One configured model behind one provider."""

    @property
    def model_id(self) -> str: ...

    @property
    def context_tokens(self) -> int:
        """The context window: prompt and reply together must fit in it."""
        ...

    @property
    def model_revision(self) -> str:
        """
        The exact weights behind `model_id`, known before any call, so callers
        can key cached results on it. Raises `LLMError` if it cannot be found.
        """
        ...

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Raise `LLMError` if the completion fails."""
        ...
