from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from assay.storage import ContractError, ParquetStore, Partition

from .records import READINGS, reading

OWNER_REPO = Partition(repo="owner/repo", epoch=0)


def test_writes_to_a_hive_partition_path(tmp_path: Path) -> None:
    path = ParquetStore(tmp_path).write(READINGS, [reading("a")], OWNER_REPO)

    assert path == tmp_path / "readings" / "repo_slug=owner__repo" / "epoch=0" / "part-0.parquet"
    assert pq.read_table(path).num_rows == 1
    assert not path.with_name("part-0.parquet.tmp").exists()


def test_reads_back_what_was_written(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    rows = [reading("a"), reading("b", note="x")]
    store.write(READINGS, rows, OWNER_REPO)

    assert store.read(READINGS) == rows


def test_rewriting_a_partition_replaces_it(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    store.write(READINGS, [reading("a"), reading("b")], OWNER_REPO)
    store.write(READINGS, [reading("c")], OWNER_REPO)

    assert [r.sensor for r in store.read(READINGS)] == ["c"]


def test_read_filters_by_repo_and_epoch(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    for repo in ("owner/one", "owner/two"):
        for epoch in (0, 1):
            store.write(
                READINGS, [reading(repo=repo, epoch=epoch)], Partition(repo=repo, epoch=epoch)
            )

    def pairs(**filters: object) -> list[tuple[str, int]]:
        rows = store.read(READINGS, **filters)  # type: ignore[arg-type]
        return [(r.repo, r.epoch) for r in rows]

    assert len(pairs()) == 4
    assert pairs(repo="owner/two") == [("owner/two", 0), ("owner/two", 1)]
    assert pairs(epoch=1) == [("owner/one", 1), ("owner/two", 1)]
    assert pairs(repo="owner/one", epoch=0) == [("owner/one", 0)]


def test_reading_a_missing_table_returns_no_records(tmp_path: Path) -> None:
    assert ParquetStore(tmp_path).read(READINGS) == []


def test_an_empty_partition_can_be_written(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    path = store.write(READINGS, [], OWNER_REPO)

    assert pq.read_table(path).schema == READINGS.schema
    assert store.read(READINGS) == []


def test_reading_a_file_with_an_outdated_schema_is_rejected(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    path = store.write(READINGS, [reading("a")], OWNER_REPO)
    pq.write_table(pq.read_table(path).drop_columns(["note"]), path)

    with pytest.raises(ContractError, match="schema differs"):
        store.read(READINGS)


@pytest.mark.parametrize(
    ("row", "column"),
    [
        (reading(repo="someone/else"), "repo someone/else"),
        (reading(epoch=3), "epoch 3"),
    ],
)
def test_rows_must_match_their_partition(tmp_path: Path, row: object, column: str) -> None:
    with pytest.raises(ContractError, match=column):
        ParquetStore(tmp_path).write(READINGS, [row], OWNER_REPO)  # type: ignore[list-item]


def test_contract_violations_are_never_written(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        ParquetStore(tmp_path).write(READINGS, [reading("a"), reading("a")], OWNER_REPO)

    assert list(tmp_path.rglob("*.parquet*")) == []


def test_reading_detects_a_tampered_file(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    path = store.write(READINGS, [reading("a")], OWNER_REPO)
    rows = pq.read_table(path).to_pylist()
    rows[0]["high"] = -10
    pq.write_table(pa.Table.from_pylist(rows, schema=READINGS.schema), path)

    with pytest.raises(ContractError, match="high is below low"):
        store.read(READINGS)


def test_duckdb_reads_partitions_without_shadowing_columns(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    store.write(READINGS, [reading("a", epoch=2)], Partition(repo="owner/repo", epoch=2))

    files = f"{tmp_path.as_posix()}/readings/**/*.parquet"
    rows = duckdb.sql(f"select repo, repo_slug, epoch from read_parquet('{files}')").fetchall()

    assert rows == [("owner/repo", "owner__repo", 2)]


def test_partition_rejects_negative_epochs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Partition(repo="owner/repo", epoch=-1)
