"""
Chat completions from a local Ollama server.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import Any

import httpx
import ollama

from assay.llm.base import ChatRequest, ChatResponse, LLMError

_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_CONTEXT_TOKENS = 8192


class OllamaClient:
    """
    One Ollama model. Local inference costs nothing, so `cost_usd` is 0.

    The server address comes from `host`, or Ollama's own `OLLAMA_HOST`
    environment variable, or `localhost:11434`. The model's digest is
    recorded as its revision: a tag such as `phi4-mini` can be re-pulled to
    different weights, a digest cannot.
    """

    def __init__(
        self,
        model_id: str,
        *,
        host: str | None = None,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        timeout_seconds: float = 300,
        client: ollama.Client | None = None,
    ) -> None:
        self._model_id = model_id
        self._context_tokens = context_tokens
        self._client = client or ollama.Client(host=host, timeout=timeout_seconds)

    @property
    def model_id(self) -> str:
        return self._model_id

    @cached_property
    def model_revision(self) -> str:
        """The installed model's digest; raises `LLMError` if it is not pulled."""

        wanted = _with_tag(self._model_id)
        with self._translated_errors():
            installed = self._client.list().models
        for model in installed:
            if model.model is not None and _with_tag(model.model) == wanted and model.digest:
                return model.digest
        msg = f"model {self._model_id!r} is not pulled; run `ollama pull {self._model_id}`"
        raise LLMError(msg, transient=False)

    def chat(self, request: ChatRequest) -> ChatResponse:
        revision = self.model_revision
        messages = [{"role": "user", "content": request.user}]
        if request.system is not None:
            messages.insert(0, {"role": "system", "content": request.system})

        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_ctx": self._context_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens

        started = time.perf_counter()
        with self._translated_errors():
            response = self._client.chat(
                model=self._model_id,
                messages=messages,
                options=options,
                format=dict(request.json_schema) if request.json_schema is not None else None,
                think=False,
            )
        return ChatResponse(
            text=response.message.content or "",
            model_id=self._model_id,
            model_revision=revision,
            prompt_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            latency_seconds=time.perf_counter() - started,
            cost_usd=0.0,
            truncated=response.done_reason == "length",
        )

    @contextmanager
    def _translated_errors(self) -> Iterator[None]:
        """Turn Ollama and transport failures into `LLMError`s, marking transient ones."""

        try:
            yield
        except ollama.ResponseError as exc:
            transient = exc.status_code in _TRANSIENT_STATUS
            msg = f"ollama {self._model_id}: {exc.error} (HTTP {exc.status_code})"
            raise LLMError(msg, transient=transient) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            msg = f"ollama {self._model_id}: server unreachable ({exc})"
            raise LLMError(msg, transient=True) from exc


def _with_tag(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"
