"""
Table specifications and the data contract every stored table must meet.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import pyarrow as pa
from pydantic import ValidationError

from assay.records import Record
from assay.storage.schema import arrow_schema

_MAX_REPORTED_PROBLEMS = 20


class ContractError(ValueError):
    """Raised when rows break a table's data contract."""


@dataclass(frozen=True)
class TableSpec[R: Record]:
    """
    A stored table: its name, the record model for one row, and the columns
    that identify a row.

    The contract is derived from the model, so each rule is written once: the
    Arrow schema must match the model, every row must validate as the model
    (field types, constraints and cross-field validators), and `key` must be
    unique. Rows are validated straight from Arrow; a pandas round trip would
    turn null strings into NaN.
    """

    name: str
    model: type[R]
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not all(self.has_column(column) for column in self.key):
            msg = f"{self.name}: key {self.key} must be non-empty columns of {self.model.__name__}"
            raise ValueError(msg)

        unloadable = [
            name
            for name, info in self.model.model_fields.items()
            if info.exclude and info.is_required()
        ]
        if unloadable:
            msg = (
                f"{self.name}: {self.model.__name__} has required fields that are never stored "
                f"({', '.join(unloadable)}), so stored rows could not be loaded back"
            )
            raise ValueError(msg)

    @cached_property
    def schema(self) -> pa.Schema:
        return arrow_schema(self.model)

    def has_column(self, name: str) -> bool:
        return name in self.schema.names

    def to_arrow(self, rows: Iterable[R]) -> pa.Table:
        return pa.Table.from_pylist([row.model_dump() for row in rows], schema=self.schema)

    def validate(self, table: pa.Table) -> list[R]:
        """
        Return the table's rows as records, or raise `ContractError` listing
        every way the table breaks the contract.
        """

        if not table.schema.equals(self.schema):
            msg = f"{self.name}: schema differs from {self.model.__name__}\n{table.schema}"
            raise ContractError(msg)

        records: list[R] = []
        problems: list[str] = []
        for index, row in enumerate(table.to_pylist()):
            try:
                records.append(self.model.model_validate(row))
            except ValidationError as exc:
                problems.extend(f"row {index}: {error['msg']}" for error in exc.errors())

        problems.extend(self._duplicate_key_problems(table))
        if problems:
            shown = problems[:_MAX_REPORTED_PROBLEMS]
            more = len(problems) - len(shown)
            summary = "\n".join(shown) + (f"\n... and {more} more" if more else "")
            msg = f"{self.name}: contract violated\n{summary}"
            raise ContractError(msg)

        return records

    def _duplicate_key_problems(self, table: pa.Table) -> list[str]:
        keys = zip(*(table.column(column).to_pylist() for column in self.key), strict=True)
        return [
            f"key {dict(zip(self.key, key, strict=True))} appears {count} times"
            for key, count in Counter(keys).items()
            if count > 1
        ]


def require_partition_values[R: Record](
    spec: TableSpec[R], rows: Sequence[R], **expected: object
) -> None:
    """
    Raise `ContractError` if a row's column disagrees with the partition it is
    written to. Columns the table does not have are not checked.
    """

    for column, value in expected.items():
        if not spec.has_column(column):
            continue
        wrong = sorted({str(getattr(row, column)) for row in rows} - {str(value)})
        if wrong:
            msg = f"{spec.name}: rows for {column}={value} contain {column} {', '.join(wrong)}"
            raise ContractError(msg)
