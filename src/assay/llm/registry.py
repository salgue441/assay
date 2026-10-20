"""
Create a client for a configured provider.

Adding a provider means implementing `LLMClient` and adding one entry to
`_PROVIDERS`; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType

from assay.llm.base import LLMClient
from assay.llm.ollama import OllamaClient
from assay.llm.retry import RetryingClient


class Provider(StrEnum):
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    OPENAI = "openai"


class UnsupportedProviderError(ValueError):
    """Raised when no client exists for a provider yet."""


_PROVIDERS: Mapping[Provider, Callable[[str], LLMClient]] = MappingProxyType(
    {
        Provider.OLLAMA: OllamaClient,
    }
)


def create_client(provider: Provider | str, model_id: str, *, retries: int = 4) -> LLMClient:
    """
    A client for `model_id` on `provider`, retrying transient failures.

    Raises `UnsupportedProviderError` for a provider with no client yet.
    """

    resolved = Provider(provider)
    factory = _PROVIDERS.get(resolved)
    if factory is None:
        supported = ", ".join(sorted(_PROVIDERS))
        msg = f"no client for provider {resolved.value!r} yet (supported: {supported})"
        raise UnsupportedProviderError(msg)
    return RetryingClient(factory(model_id), attempts=retries)
