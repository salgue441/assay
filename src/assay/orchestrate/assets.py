"""
Dagster assets for the corpus tables.

No `from __future__ import annotations` here: Dagster reads the asset
function's annotations at definition time to inject the context and
resources, and cannot resolve them as strings.
"""

from collections.abc import Iterator

import dagster as dg

from assay.corpus import SOURCES, SYMBOLS, UNITS, store_snapshot_corpus
from assay.orchestrate.partitions import CorpusPartitions
from assay.orchestrate.resources import AssayResource
from assay.records import Record
from assay.storage import ContractError, ParquetStore, Partition, TableSpec

CORPUS_GROUP = "corpus"
CONTRACT_CHECK = "contract"

_DESCRIPTIONS = {
    SOURCES.name: "One row per repository epoch: commit, tree hash, commit time, license.",
    UNITS.name: "Documentation units: modules and public top-level classes and functions.",
    SYMBOLS.name: "Every extracted symbol with its signature, docstring and file.",
}

CORPUS_TABLES: tuple[TableSpec[Record], ...] = (SOURCES, UNITS, SYMBOLS)  # type: ignore[assignment]


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
        check_specs=[
            dg.AssetCheckSpec(
                CONTRACT_CHECK,
                asset=table.name,
                blocking=True,
                description="The stored partition reloads through the table's data contract.",
            )
            for table in CORPUS_TABLES
        ],
        partitions_def=partitions.definition,
    )
    def corpus(
        context: dg.AssetExecutionContext, assay: AssayResource
    ) -> Iterator[dg.MaterializeResult[None] | dg.AssetCheckResult]:
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
            yield from _materialize_and_check(store, table, partition, commit=built.snapshot.commit)

    return corpus


def _materialize_and_check(
    store: ParquetStore, table: TableSpec[Record], partition: Partition, *, commit: str
) -> Iterator[dg.MaterializeResult[None] | dg.AssetCheckResult]:
    try:
        rows = store.read(table, repo=partition.repo, epoch=partition.epoch)
    except ContractError as exc:
        yield dg.MaterializeResult(asset_key=table.name, metadata={"commit": commit})
        yield dg.AssetCheckResult(
            asset_key=table.name,
            check_name=CONTRACT_CHECK,
            passed=False,
            metadata={"error": str(exc)},
        )
        return

    yield dg.MaterializeResult(asset_key=table.name, metadata={"rows": len(rows), "commit": commit})
    yield dg.AssetCheckResult(
        asset_key=table.name, check_name=CONTRACT_CHECK, passed=True, metadata={"rows": len(rows)}
    )
