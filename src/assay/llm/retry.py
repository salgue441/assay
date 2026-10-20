"""
Retry transient failures for any LLM client.
"""

from __future__ import annotations

import logging

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from assay.llm.base import ChatRequest, ChatResponse, LLMClient, LLMError

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.transient


class RetryingClient:
    """
    Wraps a client with the same interface and retries transient
    `LLMError`s with exponential backoff and jitter. Permanent errors and the
    last transient one are raised to the caller.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        attempts: int = 4,
        initial_wait_seconds: float = 1.0,
        max_wait_seconds: float = 30.0,
    ) -> None:
        if attempts < 1:
            msg = "attempts must be at least 1"
            raise ValueError(msg)
        self._inner = inner
        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(multiplier=initial_wait_seconds, max=max_wait_seconds),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def context_tokens(self) -> int:
        return self._inner.context_tokens

    @property
    def model_revision(self) -> str:
        return self._retrying(lambda: self._inner.model_revision)

    def chat(self, request: ChatRequest) -> ChatResponse:
        return self._retrying(self._inner.chat, request)
