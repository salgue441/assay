"""
Referential integrity: do the names a document cites exist?
"""

from __future__ import annotations

import re

from assay.metrics.base import MetricResult, listed, ratio
from assay.metrics.context import MetricContext
from assay.metrics.markdown import parse_markdown

_STRING_LITERAL = re.compile(r"""[rbuf]{0,2}("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""", re.I)
_DOTTED_NAME = re.compile(r"(?<![\w.\-])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_PATH_LIKE = re.compile(r"[/\\]")


def code_references(spans: tuple[str, ...]) -> frozenset[str]:
    """
    The dotted names cited in inline code spans.

    String literals are ignored, as are names preceded by `-` (command-line
    options such as `--verbose`) and spans that look like file paths.
    """

    names: set[str] = set()
    for span in spans:
        if _PATH_LIKE.search(span):
            continue
        names.update(_DOTTED_NAME.findall(_STRING_LITERAL.sub(" ", span)))
    return frozenset(names)


class ReferentialIntegrity:
    """
    The share of distinct names cited in inline code spans that resolve.

    A name resolves if it is a known qualified name, or if every dotted part
    is a known or ambient name (`self.items`, `os.path.join`). Known names
    are the unit's source identifiers and the snapshot's symbols, so a
    hallucinated `click.utils.fake_fn` fails while parameters resolve. A
    document that cites nothing scores 1.0; symbol coverage measures what is
    missing.

    Falling below `critical_below` is a critical failure.
    """

    name = "referential_integrity"
    version = "v1"

    def __init__(self, *, critical_below: float | None = None) -> None:
        self._critical_below = critical_below

    def score(self, text: str, context: MetricContext) -> MetricResult:
        references = code_references(parse_markdown(text).inline_code)
        unresolved = sorted(name for name in references if not _resolves(name, context))
        value = ratio(len(references) - len(unresolved), len(references))

        explanation = f"{len(references) - len(unresolved)}/{len(references)} cited names resolve"
        if unresolved:
            explanation += f"; unresolved: {listed(unresolved)}"
        critical = self._critical_below is not None and value < self._critical_below
        return MetricResult(value=value, explanation=explanation, critical=critical)


def _resolves(name: str, context: MetricContext) -> bool:
    if name in context.known_names:
        return True
    usable = context.known_names | context.rules.ambient_names
    return all(part in usable for part in name.split("."))
