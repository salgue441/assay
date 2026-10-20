"""
Typed records for the symbol inventory.

The inventory is the ground truth that static metrics (referential integrity,
symbol coverage) compare generated documentation against. It is built by
parsing source code only; no LLM is involved.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from assay.records import Record


class SourceLanguage(StrEnum):
    """Languages the corpus may contain; values match `corpus.languages` in config."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"


class SymbolKind(StrEnum):
    """What a symbol is; a function defined in a class body is a method."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class Symbol(Record):
    """
    One named, documentable thing in a source file.

    `is_public` says whether documentation is expected to cover the symbol.
    What counts as public is a language convention (a leading underscore in
    Python, an export in TypeScript, a capital letter in Go), so each
    extractor decides it.
    """

    qualified_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: SymbolKind
    signature: str | None
    docstring: str | None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    is_public: bool

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.end_line < self.start_line:
            msg = f"end_line {self.end_line} is before start_line {self.start_line}"
            raise ValueError(msg)

        if self.qualified_name != self.name and not self.qualified_name.endswith(f".{self.name}"):
            msg = f"qualified_name {self.qualified_name!r} does not end with name {self.name!r}"
            raise ValueError(msg)

        return self


class SymbolInventory(Record):
    """
    All symbols extracted from one source file.

    Qualified names are unique within an inventory: they key the `symbols`
    table together with the snapshot hash.
    """

    module: str = Field(min_length=1)
    path: str
    language: SourceLanguage
    has_syntax_errors: bool
    symbols: tuple[Symbol, ...]

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        seen: set[str] = set()
        for symbol in self.symbols:
            if symbol.qualified_name in seen:
                msg = f"duplicate qualified_name {symbol.qualified_name!r}"
                raise ValueError(msg)
            seen.add(symbol.qualified_name)

        return self

    def names(self) -> frozenset[str]:
        """
        Every simple and qualified name, for referential-integrity lookups.
        """

        return frozenset(
            name for symbol in self.symbols for name in (symbol.name, symbol.qualified_name)
        )

    def public(self) -> tuple[Symbol, ...]:
        """
        Symbols that documentation is expected to cover.
        """

        return tuple(s for s in self.symbols if s.is_public)
