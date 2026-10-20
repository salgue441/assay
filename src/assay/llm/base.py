"""
The provider-neutral contract for chat completions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """
    A failed completion. `transient` errors (timeouts, overload, a server
    that is still starting) may succeed if retried; others will not.
    """

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


@dataclass(frozen=True, slots=True)
class ChatRequest:
    user: str
    system: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    max_output_tokens: int | None = None
    json_schema: Mapping[str, Any] | None = field(default=None, hash=False)
    """Constrain the reply to this JSON schema, where the provider supports it."""


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

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Raise `LLMError` if the completion fails."""
        ...
