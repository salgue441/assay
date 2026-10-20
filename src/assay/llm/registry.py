"""
Create a client for a configured provider.

Adding a provider means implementing `LLMClient` and adding one entry to
`_PROVIDERS`; nothing else changes. Providers whose SDK is an optional extra
import it only when a client is created, so the core install still works.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from assay.llm.base import LLMClient
from assay.llm.ollama import OllamaClient
from assay.llm.retry import RetryingClient


class Provider(StrEnum):
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    OPENAI = "openai"


class UnsupportedProviderError(ValueError):
    """Raised when no client exists for a provider yet."""


class MissingProviderExtraError(UnsupportedProviderError):
    """Raised when a provider's SDK, an optional extra, is not installed."""


@contextmanager
def _requires_extra(provider: Provider, extra: str, *packages: str) -> Iterator[None]:
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] not in packages:
            raise
        msg = (
            f"the {provider.value} provider needs the {extra!r} extra, which is not "
            f"installed: run `uv sync --extra {extra}` or `pip install 'assay[{extra}]'`"
        )
        raise MissingProviderExtraError(msg) from exc


def _bedrock(model_id: str, **options: Any) -> LLMClient:
    with _requires_extra(Provider.BEDROCK, "bedrock", "boto3", "botocore"):
        from assay.llm.bedrock import BedrockClient  # noqa: PLC0415 - optional extra
    return BedrockClient(model_id, **options)


def _openai(model_id: str, **options: Any) -> LLMClient:
    with _requires_extra(Provider.OPENAI, "openai", "openai"):
        from assay.llm.openai import OpenAIClient  # noqa: PLC0415 - optional extra
    return OpenAIClient(model_id, **options)


_PROVIDERS: Mapping[Provider, Callable[..., LLMClient]] = MappingProxyType(
    {
        Provider.OLLAMA: OllamaClient,
        Provider.BEDROCK: _bedrock,
        Provider.OPENAI: _openai,
    }
)


def create_client(
    provider: Provider | str,
    model_id: str,
    *,
    retries: int = 4,
    options: Mapping[str, Any] | None = None,
) -> LLMClient:
    """
    A client for `model_id` on `provider`, retrying transient failures.

    `options` are keyword arguments for the provider's client, such as
    `context_tokens`, `prices` (a `TokenPrices`) or Bedrock's `region` and
    `profile`; an option the client does not take raises `TypeError`.

    Raises `UnsupportedProviderError` for a provider with no client yet, and
    `MissingProviderExtraError` (a subclass) when its SDK is not installed.
    Creating a client can raise `LLMError`, such as for a missing API key.
    """

    resolved = Provider(provider)
    factory = _PROVIDERS.get(resolved)
    if factory is None:
        supported = ", ".join(sorted(_PROVIDERS))
        msg = f"no client for provider {resolved.value!r} yet (supported: {supported})"
        raise UnsupportedProviderError(msg)
    return RetryingClient(factory(model_id, **(options or {})), attempts=retries)
