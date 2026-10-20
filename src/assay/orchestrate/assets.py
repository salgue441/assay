"""
Dagster assets for the pipeline's tables.

No `from __future__ import annotations` here: Dagster reads the asset
functions' annotations at definition time to inject the context and
resources, and cannot resolve them as strings.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping

import dagster as dg

from assay.corpus import SOURCES, SYMBOLS, UNITS, store_snapshot_corpus
from assay.generate import DOCUMENTS, Document, sample_units
from assay.judge import JUDGMENTS, REPLY_SCHEMA_HASH, JudgingReport, Judgment
from assay.metrics import METRICS_TABLE, Metric, MetricRow, get_rules, measure
from assay.orchestrate.partitions import CorpusPartitions
from assay.orchestrate.resources import AssayResource
from assay.records import Record
from assay.scoring import SCORES, CompositeScorer, RoutingDecision
from assay.settings import Settings
from assay.storage import ContractError, ParquetStore, Partition, TableSpec

CORPUS_GROUP = "corpus"
GENERATION_GROUP = "generation"
SCORING_GROUP = "scoring"
CONTRACT_CHECK = "contract"
COMPLETE_CHECK = "complete"

_DESCRIPTIONS = {
    SOURCES.name: "One row per repository epoch: commit, tree hash, commit time, license.",
    UNITS.name: "Documentation units: modules and public top-level classes and functions.",
    SYMBOLS.name: "Every extracted symbol with its signature, docstring and file.",
    DOCUMENTS.name: "Documentation written by each subject model for the sampled units.",
    METRICS_TABLE.name: "Deterministic metrics per document, each with an explanation.",
    JUDGMENTS.name: "Rubric grades per document, criterion, judge and run, with the issues found.",
    SCORES.name: "Composite score and routing decision per document and scorer configuration.",
}

CORPUS_TABLES: tuple[TableSpec[Record], ...] = (SOURCES, UNITS, SYMBOLS)  # type: ignore[assignment]

AssetEvents = Iterator[dg.MaterializeResult[None] | dg.AssetCheckResult]


def _contract_check_spec(table: TableSpec[Record]) -> dg.AssetCheckSpec:
    return dg.AssetCheckSpec(
        CONTRACT_CHECK,
        asset=table.name,
        blocking=True,
        description="The stored partition reloads through the table's data contract.",
    )


def corpus_assets(partitions: CorpusPartitions) -> dg.AssetsDefinition:
    """
    One multi-asset for the three corpus tables of a repository epoch.

    They come from a single build of the snapshot, so they are materialized
    together. Each stored table is read back through its contract as a
    blocking asset check: a partition that does not reload cleanly stops
    downstream assets.
    """

    @dg.multi_asset(
        name="corpus",
        specs=[
            dg.AssetSpec(table.name, group_name=CORPUS_GROUP, description=_DESCRIPTIONS[table.name])
            for table in CORPUS_TABLES
        ],
        check_specs=[_contract_check_spec(table) for table in CORPUS_TABLES],
        partitions_def=partitions.definition,
    )
    def corpus(context: dg.AssetExecutionContext, assay: AssayResource) -> AssetEvents:
        repo, epoch = partitions.resolve(context.partition_key)
        settings = assay.settings()

        built = assay.corpus_builder(settings).build(repo, epoch)
        store = assay.store(settings)
        partition = store_snapshot_corpus(store, built)
        context.log.info(
            "stored %s epoch %d at %s: %d units",
            repo.name,
            epoch,
            built.snapshot.commit[:12],
            len(built.units),
        )

        for table in CORPUS_TABLES:
            yield from _materialize_and_check(
                store, table, partition, metadata={"commit": built.snapshot.commit}
            )

    return corpus


def documents_assets(partitions: CorpusPartitions) -> dg.AssetsDefinition:
    """
    Documentation for a repository epoch's sampled units, by every subject.

    Units too large for a subject's context are left out before sampling,
    since an oversized prompt cannot be sent without losing its instructions.

    New documents are added to those already stored, so documents from
    earlier prompts or model revisions stay as history. A failed unit or an
    exhausted budget fails the non-blocking `complete` check; rerunning
    picks up only what is missing.
    """

    @dg.multi_asset(
        name="generation",
        specs=[
            dg.AssetSpec(
                DOCUMENTS.name,
                group_name=GENERATION_GROUP,
                description=_DESCRIPTIONS[DOCUMENTS.name],
                deps=[UNITS.name],
            )
        ],
        check_specs=[
            _contract_check_spec(DOCUMENTS),  # type: ignore[arg-type]
            dg.AssetCheckSpec(
                COMPLETE_CHECK,
                asset=DOCUMENTS.name,
                description="Every sampled unit has a document from every subject model.",
            ),
        ],
        partitions_def=partitions.definition,
    )
    def generation(context: dg.AssetExecutionContext, assay: AssayResource) -> AssetEvents:
        repo, epoch = partitions.resolve(context.partition_key)
        settings = assay.settings()
        store = assay.store(settings)
        partition = Partition(repo.name, epoch)

        units = store.read(UNITS, repo=repo.name, epoch=epoch)
        if not units:
            msg = f"no units stored for {context.partition_key}; run build-corpus first"
            raise dg.Failure(msg)

        generator = assay.document_generator(settings)
        language = repo.language.value
        fitting = [unit for unit in units if generator.fits(unit, language=language)]
        sample = sample_units(
            fitting, settings.generation.sample_per_partition, seed=settings.run.seed
        )
        existing = store.read(DOCUMENTS, repo=repo.name, epoch=epoch)
        report = generator.generate(
            sample, language=language, existing={d.document_id for d in existing}
        )
        store.write(DOCUMENTS, [*existing, *report.documents], partition)
        context.log.info(
            "%s: %d new documents, %d already stored, %d failed",
            context.partition_key,
            len(report.documents),
            report.skipped,
            len(report.failures),
        )

        yield from _materialize_and_check(
            store,
            DOCUMENTS,  # type: ignore[arg-type]
            partition,
            metadata={
                "new": len(report.documents),
                "too_large": len(units) - len(fitting),
                "skipped": report.skipped,
                "failed": len(report.failures),
                "output_tokens": sum(d.output_tokens for d in report.documents),
            },
        )
        yield dg.AssetCheckResult(
            asset_key=DOCUMENTS.name,
            check_name=COMPLETE_CHECK,
            passed=not report.failures and not report.stopped_by_budget,
            severity=dg.AssetCheckSeverity.WARN,
            metadata={
                "failed": len(report.failures),
                "stopped_by_budget": report.stopped_by_budget,
                "failures": "\n".join(
                    f"{f.subject} {f.unit_id}: {f.error}" for f in report.failures
                ),
            },
        )

    return generation


def metrics_assets(partitions: CorpusPartitions) -> dg.AssetsDefinition:
    """
    Every static metric for every document of a repository epoch.

    Metrics are deterministic and cheap, so the partition is recomputed in
    full on every run; a new metric version simply adds its rows.
    """

    @dg.multi_asset(
        name="static_metrics",
        specs=[
            dg.AssetSpec(
                METRICS_TABLE.name,
                group_name=SCORING_GROUP,
                description=_DESCRIPTIONS[METRICS_TABLE.name],
                deps=[DOCUMENTS.name, UNITS.name, SYMBOLS.name],
            )
        ],
        check_specs=[_contract_check_spec(METRICS_TABLE)],  # type: ignore[arg-type]
        partitions_def=partitions.definition,
    )
    def static_metrics(context: dg.AssetExecutionContext, assay: AssayResource) -> AssetEvents:
        repo, epoch = partitions.resolve(context.partition_key)
        settings = assay.settings()
        store = assay.store(settings)

        documents = store.read(DOCUMENTS, repo=repo.name, epoch=epoch)
        if not documents:
            msg = f"no documents stored for {context.partition_key}; run generate first"
            raise dg.Failure(msg)

        rows = measure(
            documents,
            store.read(UNITS, repo=repo.name, epoch=epoch),
            store.read(SYMBOLS, repo=repo.name, epoch=epoch),
            metrics=assay.static_metrics(settings),
            rules=get_rules(repo.language),
        )
        partition = Partition(repo.name, epoch)
        store.write(METRICS_TABLE, rows, partition)

        values: dict[str, list[float]] = {}
        for row in rows:
            values.setdefault(row.metric, []).append(row.value)
        yield from _materialize_and_check(
            store,
            METRICS_TABLE,  # type: ignore[arg-type]
            partition,
            metadata={
                "documents": len(documents),
                "critical": sum(row.critical for row in rows),
                **{f"mean_{name}": round(sum(v) / len(v), 4) for name, v in values.items()},
            },
        )

    return static_metrics


def judgments_assets(partitions: CorpusPartitions) -> dg.AssetsDefinition:
    """
    Grades of a repository epoch's documents by every judge, on every
    criterion, in every run.

    New judgments are added to those already stored. Documents whose prompt
    would not fit a judge's context are left out and counted as too large;
    failed calls and invalid replies fail the non-blocking `complete` check
    and are retried on the next run.
    """

    @dg.multi_asset(
        name="judging",
        specs=[
            dg.AssetSpec(
                JUDGMENTS.name,
                group_name=SCORING_GROUP,
                description=_DESCRIPTIONS[JUDGMENTS.name],
                deps=[DOCUMENTS.name, UNITS.name],
            )
        ],
        check_specs=[
            _contract_check_spec(JUDGMENTS),  # type: ignore[arg-type]
            dg.AssetCheckSpec(
                COMPLETE_CHECK,
                asset=JUDGMENTS.name,
                description="Every document has every grade from every judge.",
            ),
        ],
        partitions_def=partitions.definition,
    )
    def judging(context: dg.AssetExecutionContext, assay: AssayResource) -> AssetEvents:
        repo, epoch = partitions.resolve(context.partition_key)
        settings = assay.settings()
        store = assay.store(settings)
        partition = Partition(repo.name, epoch)

        documents = store.read(DOCUMENTS, repo=repo.name, epoch=epoch)
        if not documents:
            msg = f"no documents stored for {context.partition_key}; run generate first"
            raise dg.Failure(msg)
        units = {(u.unit_id, u.commit): u for u in store.read(UNITS, repo=repo.name, epoch=epoch)}

        judge = assay.document_judge(settings)
        language = repo.language.value
        fitting = [
            d for d in documents if judge.fits(d, units[(d.unit_id, d.commit)], language=language)
        ]
        existing = store.read(JUDGMENTS, repo=repo.name, epoch=epoch)
        report = judge.judge(
            fitting, units, language=language, existing={j.judgment_id for j in existing}
        )
        variance = JudgingReport()
        if epoch == 0:
            subset = variance_subset(
                store.read(DOCUMENTS, epoch=0),
                size=settings.judge.variance_study.subset_size,
                seed=settings.run.seed,
            )
            studied = [d for d in fitting if d.document_id in subset]
            for temperature in _variance_temperatures(settings):
                repeated = assay.document_judge(
                    settings, only=settings.scoring.judge, temperature=temperature
                ).judge(
                    studied,
                    units,
                    language=language,
                    existing={j.judgment_id for j in existing},
                )
                variance.judgments += repeated.judgments
                variance.skipped += repeated.skipped
                variance.failures += repeated.failures
                variance.stopped_by_budget |= repeated.stopped_by_budget
        store.write(JUDGMENTS, [*existing, *report.judgments, *variance.judgments], partition)
        context.log.info(
            "%s: %d new judgments, %d already stored, %d failed; variance study %d new",
            context.partition_key,
            len(report.judgments),
            report.skipped,
            len(report.failures),
            len(variance.judgments),
        )

        configured = {model.name: model.temperature for model in settings.judge.models}
        scores: dict[str, list[int]] = {}
        for judgment in [*existing, *report.judgments]:
            if judgment.temperature == configured.get(judgment.judge):
                scores.setdefault(judgment.criterion, []).append(judgment.score)
        yield from _materialize_and_check(
            store,
            JUDGMENTS,  # type: ignore[arg-type]
            partition,
            metadata={
                "new": len(report.judgments),
                "skipped": report.skipped,
                "failed": len(report.failures),
                "too_large": len(documents) - len(fitting),
                "contradictions": sum(j.contradicts_rubric for j in report.judgments),
                "variance_new": len(variance.judgments),
                **{f"mean_{name}": round(sum(v) / len(v), 3) for name, v in scores.items()},
            },
        )
        yield dg.AssetCheckResult(
            asset_key=JUDGMENTS.name,
            check_name=COMPLETE_CHECK,
            passed=not (report.failures or variance.failures)
            and not (report.stopped_by_budget or variance.stopped_by_budget),
            severity=dg.AssetCheckSeverity.WARN,
            metadata={
                "failed": len(report.failures) + len(variance.failures),
                "stopped_by_budget": report.stopped_by_budget or variance.stopped_by_budget,
                "failures": "\n".join(
                    f"{f.judge} {f.document_id[:12]} {f.criterion}#{f.run_index}: {f.error}"
                    for f in [*report.failures, *variance.failures]
                ),
            },
        )

    return judging


def scores_assets(partitions: CorpusPartitions) -> dg.AssetsDefinition:
    """
    The composite score and routing decision of each document of a
    repository epoch, under the configured scorer.

    Only the current metric versions and the configured judge's grades from
    the main judging configuration are used, so old metric definitions,
    variance-study runs and earlier rubrics never mix in. Scores of other
    scorer configurations are kept; those of the current one are replaced.
    """

    @dg.multi_asset(
        name="scoring",
        specs=[
            dg.AssetSpec(
                SCORES.name,
                group_name=SCORING_GROUP,
                description=_DESCRIPTIONS[SCORES.name],
                deps=[DOCUMENTS.name, METRICS_TABLE.name, JUDGMENTS.name],
            )
        ],
        check_specs=[_contract_check_spec(SCORES)],  # type: ignore[arg-type]
        partitions_def=partitions.definition,
    )
    def scoring(context: dg.AssetExecutionContext, assay: AssayResource) -> AssetEvents:
        repo, epoch = partitions.resolve(context.partition_key)
        settings = assay.settings()
        store = assay.store(settings)
        partition = Partition(repo.name, epoch)

        documents = store.read(DOCUMENTS, repo=repo.name, epoch=epoch)
        if not documents:
            msg = f"no documents stored for {context.partition_key}; run generate first"
            raise dg.Failure(msg)

        config = assay.scorer_config(settings)
        report = CompositeScorer(config).score(
            documents,
            _current_metric_rows(
                store.read(METRICS_TABLE, repo=repo.name, epoch=epoch),
                assay.static_metrics(settings),
            ),
            scoring_judgments(store.read(JUDGMENTS, repo=repo.name, epoch=epoch), settings),
        )
        kept = [
            score
            for score in store.read(SCORES, repo=repo.name, epoch=epoch)
            if score.scorer_version != config.version
        ]
        store.write(SCORES, [*kept, *report.scores], partition)

        decisions = Counter(score.decision for score in report.scores)
        context.log.info(
            "%s: %d scored, %d unscored under scorer %s",
            context.partition_key,
            len(report.scores),
            len(report.unscored),
            config.version,
        )
        yield from _materialize_and_check(
            store,
            SCORES,  # type: ignore[arg-type]
            partition,
            metadata={
                "scored": len(report.scores),
                "unscored": len(report.unscored),
                **{decision.value: decisions[decision] for decision in RoutingDecision},
            },
        )

    return scoring


def _current_metric_rows(rows: Iterable[MetricRow], metrics: Iterable[Metric]) -> list[MetricRow]:
    """The rows of the metric versions computed today."""

    current = {(metric.name, metric.version) for metric in metrics}
    return [row for row in rows if (row.metric, row.metric_version) in current]


def scoring_judgments(
    judgments: Iterable[Judgment], settings: Settings, *, any_temperature: bool = False
) -> list[Judgment]:
    """The scoring judge's grades from the main judging configuration: see `judge_judgments`."""

    return judge_judgments(
        judgments, settings, settings.scoring.judge, any_temperature=any_temperature
    )


def judge_judgments(
    judgments: Iterable[Judgment],
    settings: Settings,
    judge: str,
    *,
    any_temperature: bool = False,
) -> list[Judgment]:
    """
    The grades of the configured judge named `judge` from the main judging
    configuration: its configured model and temperature, the current rubric
    and reply schema, the configured output limit and the run seed (run `i`
    is judged with seed `run.seed + i`). `any_temperature` also keeps grades
    at other temperatures, such as the variance study's.
    """

    judging = settings.judge
    (model,) = (m for m in judging.models if m.name == judge)
    rubric_hash = judging.rubric().hash
    return [
        judgment
        for judgment in judgments
        if judgment.judge == model.name
        and judgment.model_id == model.model_id
        and (any_temperature or judgment.temperature == model.temperature)
        and judgment.rubric_hash == rubric_hash
        and judgment.reply_schema_hash == REPLY_SCHEMA_HASH
        and judgment.max_output_tokens == judging.max_output_tokens
        and judgment.seed == settings.run.seed + judgment.run_index
    ]


def _materialize_and_check(
    store: ParquetStore,
    table: TableSpec[Record],
    partition: Partition,
    *,
    metadata: Mapping[str, str | int | float],
) -> AssetEvents:
    try:
        rows = store.read(table, repo=partition.repo, epoch=partition.epoch)
    except ContractError as exc:
        yield dg.MaterializeResult(asset_key=table.name, metadata=dict(metadata))
        yield dg.AssetCheckResult(
            asset_key=table.name,
            check_name=CONTRACT_CHECK,
            passed=False,
            metadata={"error": str(exc)},
        )
        return

    yield dg.MaterializeResult(asset_key=table.name, metadata={"rows": len(rows), **metadata})
    yield dg.AssetCheckResult(
        asset_key=table.name, check_name=CONTRACT_CHECK, passed=True, metadata={"rows": len(rows)}
    )


def variance_subset(documents: Iterable[Document], *, size: int, seed: int) -> frozenset[str]:
    """
    The ids of the documents the variance study judges again: the first
    `size` of `documents` in an order fixed by `seed` and each document's id,
    so the subset is reproducible and never depends on reading order.
    """

    def rank(document: Document) -> str:
        return hashlib.sha256(f"{seed}:variance:{document.document_id}".encode()).hexdigest()

    return frozenset(d.document_id for d in sorted(documents, key=rank)[:size])


def _variance_temperatures(settings: Settings) -> list[float]:
    """The variance study's temperatures other than the scoring judge's own,
    whose runs the main judging already produces."""

    (model,) = (m for m in settings.judge.models if m.name == settings.scoring.judge)
    return [t for t in settings.judge.variance_study.temperatures if t != model.temperature]
