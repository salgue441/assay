from collections.abc import Callable
from pathlib import Path

import duckdb

from assay.corpus import (
    SOURCES,
    SYMBOLS,
    UNITS,
    CorpusBuilder,
    GitSnapshotFetcher,
    UnitPolicy,
    store_snapshot_corpus,
    symbol_rows,
)
from assay.storage import ParquetStore, Partition

from ..local_repo import LocalRepo

MakeRepo = Callable[..., LocalRepo]

MODULE = '''"""Shapes."""


class Square:
    """A square."""

    def area(self) -> int:
        return 4
'''


def build(make_repo: MakeRepo, tmp_path: Path) -> tuple[LocalRepo, CorpusBuilder]:
    repo = make_repo("owner/shapes")
    repo.commit({"shapes/__init__.py": "", "shapes/square.py": MODULE})
    repo.commit({"shapes/square.py": MODULE.replace("return 4", "return 16")})
    builder = CorpusBuilder(GitSnapshotFetcher(tmp_path / "cache"), policy=UnitPolicy(min_lines=1))
    return repo, builder


def test_symbol_rows_carry_their_snapshot_and_file(make_repo: MakeRepo, tmp_path: Path) -> None:
    repo, builder = build(make_repo, tmp_path)

    rows = {r.qualified_name: r for r in symbol_rows(builder.build(repo.spec(), 0))}

    area = rows["shapes.square.Square.area"]
    assert (area.repo, area.commit, area.path, area.module) == (
        "owner/shapes",
        repo.commits[0],
        "shapes/square.py",
        "shapes.square",
    )
    assert area.signature == "def area(self) -> int"
    assert not area.file_has_syntax_errors


def test_stores_every_epoch_and_reads_it_back(make_repo: MakeRepo, tmp_path: Path) -> None:
    repo, builder = build(make_repo, tmp_path)
    store = ParquetStore(tmp_path / "data")
    built = [builder.build(repo.spec(), epoch) for epoch in (0, 1)]

    partitions = [store_snapshot_corpus(store, corpus) for corpus in built]

    assert partitions == [Partition("owner/shapes", 0), Partition("owner/shapes", 1)]
    assert store.read(SOURCES) == [c.snapshot for c in built]
    assert store.read(UNITS, epoch=1) == list(built[1].units)
    assert len(store.read(SYMBOLS)) == 2 * len(symbol_rows(built[0]))


def test_stored_tables_are_queryable_with_duckdb(make_repo: MakeRepo, tmp_path: Path) -> None:
    repo, builder = build(make_repo, tmp_path)
    store = ParquetStore(tmp_path / "data")
    for epoch in (0, 1):
        store_snapshot_corpus(store, builder.build(repo.spec(), epoch))
    root = (tmp_path / "data").as_posix()

    changed = duckdb.sql(
        f"""
        select now.qualified_name
        from read_parquet('{root}/units/**/*.parquet') as now
        left join read_parquet('{root}/units/**/*.parquet') as before
          on before.unit_id = now.unit_id and before.epoch = now.epoch - 1
        where now.epoch = 1 and before.unit_hash is distinct from now.unit_hash
        order by 1
        """
    ).fetchall()

    assert changed == [("shapes.square",), ("shapes.square.Square",)]
