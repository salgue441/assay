"""
The Dagster resource that turns configuration into pipeline objects.
"""

from __future__ import annotations

from pathlib import Path

import dagster as dg

from assay.corpus import CorpusBuilder, GitSnapshotFetcher
from assay.generate import Budget, DocumentGenerator, Subject
from assay.judge import DocumentJudge, JudgeModel
from assay.llm import create_client
from assay.metrics import Metric, static_metrics
from assay.scoring import ScorerConfig
from assay.settings import Settings, load_settings
from assay.storage import ParquetStore


class AssayResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    """
    Where configuration becomes objects: assets ask this resource for a
    builder or a store instead of constructing them, so tests and other
    deployments only change the config file it points at.
    """

    config_path: str
    base_dir: str | None = None
    """Directory relative config paths resolve against; the current one if unset."""

    def settings(self) -> Settings:
        base_dir = Path(self.base_dir) if self.base_dir is not None else None
        return load_settings(Path(self.config_path), base_dir=base_dir)

    def corpus_builder(self, settings: Settings) -> CorpusBuilder:
        corpus = settings.corpus
        return CorpusBuilder(
            GitSnapshotFetcher(corpus.cache_dir),
            policy=corpus.units.policy(),
            excluded_dirs=corpus.excluded_dirs,
            max_file_bytes=corpus.max_file_bytes,
        )

    def store(self, settings: Settings) -> ParquetStore:
        return ParquetStore(settings.storage.root)

    def static_metrics(self, settings: Settings) -> tuple[Metric, ...]:
        critical = settings.scoring.critical_failures
        return static_metrics(integrity_critical_below=critical.referential_integrity_below)

    def scorer_config(self, settings: Settings) -> ScorerConfig:
        scoring = settings.scoring
        return ScorerConfig(
            weights=scoring.weights,
            judge=scoring.judge,
            publish_at_or_above=scoring.routing.publish_at_or_above,
            escalate_below=scoring.routing.escalate_below,
        )

    def document_judge(
        self, settings: Settings, *, only: str | None = None, temperature: float | None = None
    ) -> DocumentJudge:
        """
        Every configured judge, or `only` the named one, each at its own
        temperature unless `temperature` overrides it (the variance study).
        """

        judging = settings.judge
        judges = [
            JudgeModel(
                name=model.name,
                provider=model.provider,
                client=create_client(model.provider, model.model_id),
                temperature=model.temperature if temperature is None else temperature,
            )
            for model in judging.models
            if only is None or model.name == only
        ]
        return DocumentJudge(
            judges,
            judging.rubric(),
            criteria=judging.criteria,
            runs=judging.runs_per_criterion,
            seed=settings.run.seed,
            max_output_tokens=judging.max_output_tokens,
            budget=Budget(settings.run.budget_usd),
        )

    def document_generator(self, settings: Settings) -> DocumentGenerator:
        generation = settings.generation
        subjects = [
            Subject(
                name=subject.name,
                provider=subject.provider,
                client=create_client(subject.provider, subject.model_id),
                temperature=subject.temperature,
            )
            for subject in generation.subjects
        ]
        return DocumentGenerator(
            subjects,
            generation.prompt(),
            seed=settings.run.seed,
            max_output_tokens=generation.max_output_tokens,
            budget=Budget(settings.run.budget_usd),
        )
