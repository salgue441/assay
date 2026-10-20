"""
What a metric knows about the unit a document describes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from assay.corpus import Unit
from assay.extract import Symbol, SymbolKind
from assay.metrics.rules import LanguageRules

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class MetricContext:
    """
    The ground truth a document is checked against: the unit, the symbols
    inside it, and every name the document may legitimately use.
    """

    unit: Unit
    unit_symbols: tuple[Symbol, ...]
    """Symbols inside the unit, the unit itself first."""
    known_names: frozenset[str]
    """Identifiers in the unit's source, plus every symbol in the snapshot by
    simple and qualified name and each part of a qualified name."""
    rules: LanguageRules

    @classmethod
    def build(
        cls, unit: Unit, snapshot_symbols: Iterable[Symbol], rules: LanguageRules
    ) -> MetricContext:
        snapshot_symbols = tuple(snapshot_symbols)
        inside = set(unit.symbols)
        unit_symbols = sorted(
            (s for s in snapshot_symbols if s.qualified_name in inside),
            key=lambda s: s.qualified_name != unit.qualified_name,
        )

        known = set(_IDENTIFIER.findall(unit.source))
        for symbol in snapshot_symbols:
            known.update((symbol.name, symbol.qualified_name, *symbol.qualified_name.split(".")))
        return cls(
            unit=unit,
            unit_symbols=tuple(unit_symbols),
            known_names=frozenset(known),
            rules=rules,
        )

    @cached_property
    def symbol(self) -> Symbol | None:
        """The unit's own symbol."""

        return next(
            (s for s in self.unit_symbols if s.qualified_name == self.unit.qualified_name), None
        )

    @cached_property
    def parameters(self) -> tuple[str, ...]:
        """A function's parameters; empty for classes and modules."""

        is_function = self.unit.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
        if not is_function or self.symbol is None or self.symbol.signature is None:
            return ()
        return self.rules.parameters(self.symbol.signature)

    @cached_property
    def return_annotation(self) -> str | None:
        if self.symbol is None or self.symbol.signature is None:
            return None
        return self.rules.return_annotation(self.symbol.signature)

    @cached_property
    def public_members(self) -> tuple[Symbol, ...]:
        """Public symbols directly inside the unit: a class's methods and
        nested classes, or a module's top-level classes and functions. A
        constructor is not a member: its parameters are covered instead."""

        prefix = f"{self.unit.qualified_name}."
        return tuple(
            s
            for s in self.unit_symbols
            if s.qualified_name == f"{prefix}{s.name}"
            and s.is_public
            and s is not self._constructor
        )

    @cached_property
    def constructor_parameters(self) -> tuple[str, ...]:
        """A class's constructor parameters (`__init__` in Python); empty for
        anything else."""

        constructor = self._constructor
        if constructor is None or constructor.signature is None:
            return ()
        return self.rules.parameters(constructor.signature)

    @cached_property
    def _constructor(self) -> Symbol | None:
        name = self.rules.constructor_name
        if self.unit.kind is not SymbolKind.CLASS or name is None:
            return None
        qualified_name = f"{self.unit.qualified_name}.{name}"
        return next((s for s in self.unit_symbols if s.qualified_name == qualified_name), None)
