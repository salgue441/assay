"""A scripted LLMClient for tests: records requests, replies or fails on cue."""

from collections.abc import Callable
from dataclasses import dataclass, field

from assay.llm import ChatRequest, ChatResponse, LLMError


@dataclass
class FakeLLM:
    model_id: str = "fake-model:1b"
    model_revision: str = "rev-1"
    context_tokens: int = 8192
    reply: Callable[[ChatRequest], str] = lambda request: "## doc"
    fail_when: Callable[[ChatRequest], bool] = lambda request: False
    cost_usd: float = 0.0
    requests: list[ChatRequest] = field(default_factory=list)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.fail_when(request):
            raise LLMError("scripted failure", transient=True)
        text = self.reply(request)
        return ChatResponse(
            text=text,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_tokens=len(request.user.split()),
            output_tokens=len(text.split()),
            latency_seconds=0.01,
            cost_usd=self.cost_usd,
            truncated=False,
        )
