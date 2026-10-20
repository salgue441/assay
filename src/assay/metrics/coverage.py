"""
Symbol coverage: does a document mention what it should?
"""

from __future__ import annotations

import re

from assay.extract import SymbolKind
from assay.metrics.base import MetricResult, listed, ratio
from assay.metrics.context import MetricContext


def expected_names(context: MetricContext) -> tuple[str, ...]:
    """
    The names a complete document mentions: a function's parameters; a
    class's constructor parameters and public members; a module's public
    top-level classes and functions. Dunder methods are not expected.
    """

    if context.unit.kind is SymbolKind.MODULE:
        return tuple(s.name for s in context.public_members)
    if context.unit.kind is SymbolKind.CLASS:
        return (*context.constructor_parameters, *(s.name for s in context.public_members))
    return context.parameters


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text) is not None


class SymbolCoverage:
    """
    The share of expected names the document mentions anywhere, as whole
    words. A unit with nothing to expect scores 1.0.
    """

    name = "symbol_coverage"
    version = "v1"

    def score(self, text: str, context: MetricContext) -> MetricResult:
        expected = list(dict.fromkeys(expected_names(context)))
        missing = [name for name in expected if not _mentions(text, name)]
        value = ratio(len(expected) - len(missing), len(expected))

        explanation = f"{len(expected) - len(missing)}/{len(expected)} expected names mentioned"
        if missing:
            explanation += f"; missing: {listed(missing)}"
        return MetricResult(value=value, explanation=explanation)
