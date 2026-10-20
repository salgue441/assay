"""
Grade documents with one or more judge models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from assay.corpus import Unit
from assay.generate import Budget, Document
from assay.judge.judgments import Judgment, judgment_id
from assay.judge.rubric import REPLY_SCHEMA, REPLY_SCHEMA_HASH, JudgeReply, Rubric
from assay.llm import ChatRequest, LLMClient, LLMError, Provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeModel:
    """A configured judge: a named model with its sampling temperature."""

    name: str
    provider: Provider
    client: LLMClient
    temperature: float = 0.0


@dataclass(frozen=True)
class JudgingFailure:
    document_id: str
    judge: str
    criterion: str
    run_index: int
    error: str


@dataclass
class JudgingReport:
    judgments: list[Judgment] = field(default_factory=list)
    skipped: int = 0
    """Judgments that already existed."""
    failures: list[JudgingFailure] = field(default_factory=list)
    """Calls that failed after retries or returned an invalid reply; retried next run."""
    stopped_by_budget: bool = False


@dataclass(frozen=True)
class _Task:
    """One grade to produce: a judge, a document, a criterion and a run."""

    judge: JudgeModel
    revision: str
    document: Document
    unit: Unit
    criterion: str
    run_index: int
    seed: int
    judgment_id: str


class DocumentJudge:
    """
    Grades each document on each criterion, `runs` times, with each judge.

    Every criterion is a separate call so one impression of a document
    cannot carry over to the other criteria. Run `i` uses seed `seed + i`,
    so repeated runs can differ when the temperature is above zero.
    """

    def __init__(
        self,
        judges: Sequence[JudgeModel],
        rubric: Rubric,
        *,
        criteria: Sequence[str],
        runs: int,
        seed: int,
        max_output_tokens: int,
        budget: Budget | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not judges:
            msg = "at least one judge model is required"
            raise ValueError(msg)
        if runs < 1:
            msg = "runs must be at least 1"
            raise ValueError(msg)
        unknown = [c for c in criteria if c not in rubric.criteria]
        if not criteria or unknown:
            msg = f"criteria must be non-empty and in rubric {rubric.version}; unknown: {unknown}"
            raise ValueError(msg)
        self._judges = judges
        self._rubric = rubric
        self._criteria = tuple(criteria)
        self._runs = runs
        self._seed = seed
        self._max_output_tokens = max_output_tokens
        self._budget = budget or Budget(None)
        self._clock = clock

    def fits(self, document: Document, unit: Unit, *, language: str) -> bool:
        """Whether every criterion's prompt fits every judge's context."""

        smallest = min(judge.client.context_tokens for judge in self._judges)
        return all(
            self._request(c, document, unit, language, temperature=0.0, seed=0).estimated_tokens()
            <= smallest
            for c in self._criteria
        )

    def judge(
        self,
        documents: Sequence[Document],
        units: Mapping[tuple[str, str], Unit],
        *,
        language: str,
        existing: Collection[str] = (),
    ) -> JudgingReport:
        """
        Grade every document. `units` maps (unit_id, commit) to each
        document's unit; judgment ids in `existing` are skipped.
        """

        report = JudgingReport()
        for task in self._tasks(documents, units):
            if task.judgment_id in existing:
                report.skipped += 1
                continue
            if self._budget.exhausted:
                report.stopped_by_budget = True
                break
            try:
                report.judgments.append(self._grade(task, language))
            except (LLMError, ValueError) as exc:
                logger.warning(
                    "%s on %s/%s run %d failed: %s",
                    task.judge.name,
                    task.document.document_id[:12],
                    task.criterion,
                    task.run_index,
                    exc,
                )
                report.failures.append(
                    JudgingFailure(
                        task.document.document_id,
                        task.judge.name,
                        task.criterion,
                        task.run_index,
                        str(exc),
                    )
                )
        return report

    def _tasks(
        self, documents: Sequence[Document], units: Mapping[tuple[str, str], Unit]
    ) -> Iterator[_Task]:
        for judge in self._judges:
            revision = judge.client.model_revision
            for document in documents:
                unit = units[(document.unit_id, document.commit)]
                for criterion in self._criteria:
                    for run in range(self._runs):
                        yield _Task(
                            judge=judge,
                            revision=revision,
                            document=document,
                            unit=unit,
                            criterion=criterion,
                            run_index=run,
                            seed=self._seed + run,
                            judgment_id=judgment_id(
                                document_id=document.document_id,
                                rubric_hash=self._rubric.hash,
                                criterion=criterion,
                                provider=judge.provider,
                                model_id=judge.client.model_id,
                                model_revision=revision,
                                run_index=run,
                                seed=self._seed + run,
                                temperature=judge.temperature,
                                max_output_tokens=self._max_output_tokens,
                                reply_schema_hash=REPLY_SCHEMA_HASH,
                            ),
                        )

    def _grade(self, task: _Task, language: str) -> Judgment:
        request = self._request(
            task.criterion,
            task.document,
            task.unit,
            language,
            temperature=task.judge.temperature,
            seed=task.seed,
        )
        response = task.judge.client.chat(request)
        self._budget.charge(response.cost_usd)
        if response.truncated:
            msg = f"reply cut off at the {self._max_output_tokens}-token output limit"
            raise ValueError(msg)
        reply = JudgeReply.parse(response.text)
        return Judgment(
            judgment_id=task.judgment_id,
            document_id=task.document.document_id,
            unit_id=task.document.unit_id,
            repo=task.document.repo,
            commit=task.document.commit,
            subject=task.document.subject,
            judge=task.judge.name,
            provider=task.judge.provider,
            model_id=response.model_id,
            model_revision=task.revision,
            rubric_version=self._rubric.version,
            rubric_hash=self._rubric.hash,
            criterion=task.criterion,
            run_index=task.run_index,
            seed=task.seed,
            temperature=task.judge.temperature,
            max_output_tokens=self._max_output_tokens,
            reply_schema_hash=REPLY_SCHEMA_HASH,
            score=reply.score,
            issues=tuple(reply.issues),
            rationale=reply.rationale,
            prompt_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds,
            cost_usd=response.cost_usd,
            judged_at=self._clock(),
        )

    def _request(
        self,
        criterion: str,
        document: Document,
        unit: Unit,
        language: str,
        *,
        temperature: float,
        seed: int,
    ) -> ChatRequest:
        system, user = self._rubric.render(
            criterion, unit=unit, document=document.text, language=language
        )
        return ChatRequest(
            user=user,
            system=system,
            temperature=temperature,
            seed=seed,
            max_output_tokens=self._max_output_tokens,
            json_schema=REPLY_SCHEMA,
        )
