"""
Documentation units: the pieces of source a subject model is asked to document.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Self

from pydantic import Field, computed_field, model_validator

from assay.corpus.discovery import SourceFile
from assay.corpus.manifest import CommitSha
from assay.extract import Symbol, SymbolInventory, SymbolKind
from assay.records import Record

UNIT_KINDS = frozenset({SymbolKind.MODULE, SymbolKind.CLASS, SymbolKind.FUNCTION})
"""Methods are documented as part of their class, never as units of their own."""


class Unit(Record):
    """
    One documentation unit in one snapshot: a `units` row.

    `unit_id` names the unit's position (repository, file, symbol) and stays the
    same across epochs; `unit_hash` covers its path and source text and changes
    whenever either does.
    """

    unit_id: str
    unit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo: str
    commit: CommitSha
    path: str
    qualified_name: str
    kind: SymbolKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source: str
    symbols: tuple[str, ...]
    """Qualified names of every symbol inside the unit, itself included."""

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        if self.kind not in UNIT_KINDS:
            msg = f"{self.kind.value!r} is not a unit kind"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_lines(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass(frozen=True, slots=True)
class UnitPolicy:
    """Which symbols become documentation units."""

    kinds: frozenset[SymbolKind] = field(default=UNIT_KINDS)
    min_lines: int = 3
    include_private: bool = False

    def __post_init__(self) -> None:
        if not self.kinds <= UNIT_KINDS:
            msg = f"unit kinds must be a subset of {sorted(UNIT_KINDS)}"
            raise ValueError(msg)
        if self.min_lines < 1:
            msg = "min_lines must be at least 1"
            raise ValueError(msg)


DEFAULT_POLICY = UnitPolicy()


def select_units(
    inventory: SymbolInventory,
    source: SourceFile,
    *,
    repo: str,
    commit: str,
    policy: UnitPolicy = DEFAULT_POLICY,
) -> tuple[Unit, ...]:
    """
    Cut one file into units: the module itself plus its top-level classes and
    functions, filtered by `policy`.
    """

    lines = source.text.splitlines(keepends=True)
    return tuple(
        _to_unit(symbol, inventory, lines, repo=repo, commit=commit, path=str(source.path))
        for symbol in inventory.symbols
        if _is_selected(symbol, inventory.module, policy)
    )


def changed_units(previous: Iterable[Unit], current: Iterable[Unit]) -> tuple[Unit, ...]:
    """
    Units in `current` that are new or whose content changed since `previous`:
    the rolling sample of a drift epoch.
    """

    previous_hashes = {unit.unit_id: unit.unit_hash for unit in previous}
    return tuple(u for u in current if previous_hashes.get(u.unit_id) != u.unit_hash)


def _is_selected(symbol: Symbol, module: str, policy: UnitPolicy) -> bool:
    is_module_or_top_level = symbol.qualified_name in (module, f"{module}.{symbol.name}")
    return (
        symbol.kind in policy.kinds
        and is_module_or_top_level
        and (symbol.is_public or policy.include_private)
        and symbol.end_line - symbol.start_line + 1 >= policy.min_lines
    )


def _to_unit(
    symbol: Symbol,
    inventory: SymbolInventory,
    lines: list[str],
    *,
    repo: str,
    commit: str,
    path: str,
) -> Unit:
    text = "".join(lines[symbol.start_line - 1 : symbol.end_line])

    return Unit(
        unit_id=f"{repo}:{path}#{symbol.qualified_name}",
        unit_hash=hashlib.sha256(f"{path}\0{text}".encode()).hexdigest(),
        repo=repo,
        commit=commit,
        path=path,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        source=text,
        symbols=_contained_names(symbol, inventory),
    )


def _contained_names(unit: Symbol, inventory: SymbolInventory) -> tuple[str, ...]:
    if unit.kind is SymbolKind.MODULE:
        return tuple(s.qualified_name for s in inventory.symbols)

    nested_prefix = f"{unit.qualified_name}."
    return tuple(
        s.qualified_name
        for s in inventory.symbols
        if s.qualified_name == unit.qualified_name or s.qualified_name.startswith(nested_prefix)
    )
