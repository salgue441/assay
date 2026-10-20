"""
Language-independent steps shared by the extractors: putting a definition
header on one line, and merging symbols that share a qualified name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from assay.extract.models import Symbol

# Applied in order to a header with comments removed, before whitespace is collapsed.
_SIGNATURE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r",\s*\n\s*([)\]])"), r"\1"),  # trailing comma before a closing bracket
    (re.compile(r"([(\[])\s*\n\s*"), r"\1"),  # line break after an opening bracket
    (re.compile(r"\s*\n\s*([)\]])"), r"\1"),  # line break before a closing bracket
)


def normalize_signature(header: str) -> str:
    """
    Collapse a possibly multi-line definition header onto one line.
    """

    for pattern, replacement in _SIGNATURE_RULES:
        header = pattern.sub(replacement, header)

    return " ".join(header.split())


def merge_redefinitions(symbols: Iterable[Symbol]) -> tuple[Symbol, ...]:
    """
    Keep the first symbol of each qualified name, in order.

    A later definition contributes only its docstring, and only when the
    first has none: a property setter documented instead of its getter, or
    an overload documented on its implementation.
    """

    merged: dict[str, Symbol] = {}
    for symbol in symbols:
        first = merged.get(symbol.qualified_name)
        if first is None:
            merged[symbol.qualified_name] = symbol
        elif first.docstring is None and symbol.docstring is not None:
            merged[symbol.qualified_name] = first.model_copy(update={"docstring": symbol.docstring})

    return tuple(merged.values())
