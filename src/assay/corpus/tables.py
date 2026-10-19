"""
The corpus tables (`sources`, `units`, `symbols`) and how a built epoch is stored.
"""

from __future__ import annotations

from assay.corpus.builder import SnapshotCorpus
from assay.corpus.manifest import CommitSha
from assay.corpus.snapshot import Snapshot
from assay.corpus.units import Unit
from assay.extract import SourceLanguage, Symbol
from assay.storage import ParquetStore, Partition, TableSpec


class SymbolRow(Symbol):
    """
    A symbol with the snapshot and file it was extracted from: a `symbols` row.
    """

    repo: str
    commit: CommitSha
    path: str
    module: str
    language: SourceLanguage
    file_has_syntax_errors: bool


SOURCES = TableSpec(name="sources", model=Snapshot, key=("repo", "epoch"))
UNITS = TableSpec(name="units", model=Unit, key=("unit_id", "commit"))
SYMBOLS = TableSpec(
    name="symbols", model=SymbolRow, key=("repo", "commit", "path", "qualified_name")
)


def symbol_rows(corpus: SnapshotCorpus) -> tuple[SymbolRow, ...]:
    snapshot = corpus.snapshot
    return tuple(
        SymbolRow(
            **symbol.model_dump(exclude=set(Symbol.model_computed_fields)),
            repo=snapshot.repo,
            commit=snapshot.commit,
            path=inventory.path,
            module=inventory.module,
            language=inventory.language,
            file_has_syntax_errors=inventory.has_syntax_errors,
        )
        for inventory in corpus.inventories
        for symbol in inventory.symbols
    )


def store_snapshot_corpus(store: ParquetStore, corpus: SnapshotCorpus) -> Partition:
    """
    Write one built epoch to the `sources`, `units` and `symbols` tables,
    replacing any earlier write of the same partition.
    """

    partition = Partition(repo=corpus.snapshot.repo, epoch=corpus.snapshot.epoch)
    store.write(SOURCES, [corpus.snapshot], partition)
    store.write(UNITS, corpus.units, partition)
    store.write(SYMBOLS, symbol_rows(corpus), partition)
    return partition
