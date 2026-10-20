"""
Generate documentation for units with one or more subject models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from assay.corpus import Unit
from assay.generate.documents import Document, document_id
from assay.generate.prompt import PromptTemplate
from assay.llm import ChatRequest, LLMClient, LLMError, Provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subject:
    """A configured writer: a named model with its sampling settings."""

    name: str
    provider: Provider
    client: LLMClient
    temperature: float = 0.0


class Budget:
    """
    Tracks spend against an optional cap. Generation stops before a call
    once the cap is reached; a single call can overshoot it.
    """

    def __init__(self, limit_usd: float | None) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    @property
    def exhausted(self) -> bool:
        return self.limit_usd is not None and self.spent_usd >= self.limit_usd

    def charge(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd


@dataclass(frozen=True)
class Failure:
    unit_id: str
    subject: str
    error: str


@dataclass
class GenerationReport:
    documents: list[Document] = field(default_factory=list)
    skipped: int = 0
    """Documents that already existed."""
    failures: list[Failure] = field(default_factory=list)
    """Units whose generation failed after retries; they are retried on the next run."""
    stopped_by_budget: bool = False


class DocumentGenerator:
    """
    Turns units into documents, one per unit and subject.

    Work already done (a document id in `existing`) is skipped. A failed
    call is logged and recorded, and generation moves on, so one bad unit
    does not lose the rest of the run. A model that cannot be found at all
    fails immediately instead.
    """

    def __init__(
        self,
        subjects: Sequence[Subject],
        prompt: PromptTemplate,
        *,
        seed: int,
        max_output_tokens: int,
        budget: Budget | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not subjects:
            msg = "at least one subject model is required"
            raise ValueError(msg)
        self._subjects = subjects
        self._prompt = prompt
        self._seed = seed
        self._max_output_tokens = max_output_tokens
        self._budget = budget or Budget(None)
        self._clock = clock

    def fits(self, unit: Unit, *, language: str) -> bool:
        """
        Whether the unit's prompt and reply fit every subject's context, by a
        conservative estimate. Units that do not fit cannot be documented in
        one prompt and should be left out before sampling.
        """

        return self._request(unit, language, temperature=0.0).estimated_tokens() <= min(
            subject.client.context_tokens for subject in self._subjects
        )

    def generate(
        self, units: Sequence[Unit], *, language: str, existing: Collection[str] = ()
    ) -> GenerationReport:
        report = GenerationReport()
        for subject in self._subjects:
            revision = subject.client.model_revision
            for unit in units:
                identity = document_id(
                    unit,
                    prompt_hash=self._prompt.hash,
                    provider=subject.provider,
                    model_id=subject.client.model_id,
                    model_revision=revision,
                    seed=self._seed,
                    temperature=subject.temperature,
                    max_output_tokens=self._max_output_tokens,
                )
                if identity in existing:
                    report.skipped += 1
                    continue
                if self._budget.exhausted:
                    report.stopped_by_budget = True
                    return report

                try:
                    report.documents.append(
                        self._generate_one(unit, subject, identity, revision, language)
                    )
                except LLMError as exc:
                    logger.warning("%s on %s failed: %s", subject.name, unit.unit_id, exc)
                    report.failures.append(Failure(unit.unit_id, subject.name, str(exc)))
        return report

    def _generate_one(
        self, unit: Unit, subject: Subject, identity: str, revision: str, language: str
    ) -> Document:
        response = subject.client.chat(
            self._request(unit, language, temperature=subject.temperature)
        )
        self._budget.charge(response.cost_usd)
        return Document(
            document_id=identity,
            unit_id=unit.unit_id,
            unit_hash=unit.unit_hash,
            repo=unit.repo,
            commit=unit.commit,
            qualified_name=unit.qualified_name,
            kind=unit.kind,
            subject=subject.name,
            provider=subject.provider,
            model_id=response.model_id,
            model_revision=revision,
            prompt_version=self._prompt.version,
            prompt_hash=self._prompt.hash,
            seed=self._seed,
            temperature=subject.temperature,
            max_output_tokens=self._max_output_tokens,
            text=response.text,
            truncated=response.truncated,
            prompt_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds,
            cost_usd=response.cost_usd,
            generated_at=self._clock(),
        )

    def _request(self, unit: Unit, language: str, *, temperature: float) -> ChatRequest:
        system, user = self._prompt.render(unit, language=language)
        return ChatRequest(
            user=user,
            system=system,
            temperature=temperature,
            seed=self._seed,
            max_output_tokens=self._max_output_tokens,
        )
