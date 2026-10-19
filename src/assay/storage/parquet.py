"""
Partitioned Parquet storage for record tables.

Layout: `{root}/{table}/repo_slug={owner__repo}/epoch={n}/part-0.parquet`.

Partition keys deliberately differ from column names: DuckDB's Hive
partitioning replaces a file's column with the directory value when the
names match, which would turn `owner/repo` into `owner__repo` on read.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from assay.records import Record
from assay.storage.table import TableSpec, require_partition_values

_PART_FILE = "part-0.parquet"


@dataclass(frozen=True, slots=True)
class Partition:
    """One repository epoch: the unit a pipeline run writes."""

    repo: str
    epoch: int

    def __post_init__(self) -> None:
        if self.epoch < 0:
            msg = f"epoch must be non-negative, got {self.epoch}"
            raise ValueError(msg)

    @property
    def path(self) -> Path:
        return Path(f"repo_slug={repo_slug(self.repo)}", f"epoch={self.epoch}")


class ParquetStore:
    """
    Writes and reads record tables under one root directory.

    Writes are contract-checked and atomic: a partition is written to a
    temporary file and renamed over the previous version, so rerunning a
    partition replaces it and an interrupted write leaves the old one intact.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def write[R: Record](self, spec: TableSpec[R], rows: Sequence[R], partition: Partition) -> Path:
        require_partition_values(spec, rows, repo=partition.repo, epoch=partition.epoch)
        table = spec.to_arrow(rows)
        spec.validate(table)

        target = self._root / spec.name / partition.path / _PART_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        pq.write_table(table, temporary)
        os.replace(temporary, target)
        return target

    def read[R: Record](
        self, spec: TableSpec[R], *, repo: str | None = None, epoch: int | None = None
    ) -> list[R]:
        """
        Read a table's records, optionally limited to one repository and/or
        epoch, in partition order. A table with no matching partitions reads as
        empty.

        Each file is contract-checked as stored, never cast to the expected
        schema, so a tampered or outdated file raises `ContractError`.
        """

        slug = "*" if repo is None else repo_slug(repo)
        pattern = f"repo_slug={slug}/epoch={'*' if epoch is None else epoch}/{_PART_FILE}"
        files = sorted((self._root / spec.name).glob(pattern), key=_partition_order)

        return [record for file in files for record in spec.validate(pq.read_table(file))]


def repo_slug(repo: str) -> str:
    """`owner/repo` made safe for one path segment."""

    return repo.replace("/", "__")


def _partition_order(file: Path) -> tuple[str, int]:
    epoch_dir, slug_dir = file.parent, file.parent.parent
    return slug_dir.name, int(epoch_dir.name.removeprefix("epoch="))
