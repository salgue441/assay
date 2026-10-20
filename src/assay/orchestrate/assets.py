"""
Dagster assets for the pipeline's tables.

No `from __future__ import annotations` here: Dagster reads the asset
functions' annotations at definition time to inject the context and
resources, and cannot resolve them as strings.
"""

from collections.abc import Iterator, Mapping

import dagster as dg

from assay.corpus import SOURCES, SYMBOLS, UNITS, store_snapshot_corpus
from assay.generate import DOCUMENTS, sample_units
from assay.orchestrate.partitions import CorpusPartitions
from assay.orchestrate.resources import AssayResource
from assay.records import Record
from assay.storage import ContractError, ParquetStore, Partition, TableSpec

CORPUS_GROUP = "corpus"
GENERATION_GROUP = "generation"
CONTRACT_CHECK = "contract"
COMPLETE_CHECK = "complete"

_DESCRIPTIONS = {
    SOURCES.name: "One row per repository epoch: commit, tree hash, commit time, license.",
    UNITS.name: "Documentation units: modules and public top-level classes and functions.",
    SYMBOLS.name: "Every extracted symbol with its signature, docstring and file.",
    DOCUMENTS.name: "Documentation written by each subject model for the sampled units.",
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

        sample = sample_units(
            units, settings.generation.sample_per_partition, seed=settings.run.seed
        )
        existing = store.read(DOCUMENTS, repo=repo.name, epoch=epoch)
        report = assay.document_generator(settings).generate(
            sample, language=repo.language.value, existing={d.document_id for d in existing}
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


def _materialize_and_check(
    store: ParquetStore,
    table: TableSpec[Record],
    partition: Partition,
    *,
    metadata: Mapping[str, str | int],
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
