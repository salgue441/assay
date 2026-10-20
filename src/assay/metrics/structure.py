"""
Structural completeness: does a document have the sections it needs?
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

from assay.extract import SymbolKind
from assay.metrics.base import MetricResult, ratio
from assay.metrics.context import MetricContext
from assay.metrics.markdown import Heading, parse_markdown

SECTION_NAMES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "summary": frozenset({"summary", "overview", "description"}),
        "parameters": frozenset({"parameters", "arguments", "args"}),
        "returns": frozenset({"returns", "return value", "return"}),
        "constructor": frozenset({"constructor", "initialization", "__init__"}),
        "methods": frozenset({"methods"}),
        "contents": frozenset({"contents", "members", "classes", "functions"}),
    }
)
"""Accepted heading texts per section, compared case-insensitively."""

_NOT_WORD = re.compile(r"[^\w ]+")


def required_sections(context: MetricContext) -> tuple[str, ...]:
    """
    The sections a unit needs, following the generation prompt: always a
    summary; parameters and a declared non-None return for functions; a
    constructor with parameters and public members for classes; public
    members for modules.
    """

    sections = ["summary"]
    kind = context.unit.kind
    if kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
        if context.parameters:
            sections.append("parameters")
        if context.return_annotation not in (None, "None"):
            sections.append("returns")
    elif kind is SymbolKind.CLASS:
        if context.constructor_parameters:
            sections.append("constructor")
        if any(s.kind is SymbolKind.METHOD for s in context.public_members):
            sections.append("methods")
    elif context.public_members:
        sections.append("contents")
    return tuple(sections)


def _normalized(heading: Heading) -> str:
    return " ".join(_NOT_WORD.sub(" ", heading.text.lower()).split())


def _has_section(headings: tuple[Heading, ...], section: str) -> bool:
    accepted = SECTION_NAMES[section]
    return any(
        heading.level >= 3 and any(_normalized(heading).startswith(name) for name in accepted)
        for heading in headings
    )


class StructuralCompleteness:
    """
    The share of structural requirements met: a level-2 title naming the
    unit first, then each required section as a level-3 or deeper heading.
    """

    name = "structural_completeness"
    version = "v1"

    def score(self, text: str, context: MetricContext) -> MetricResult:
        headings = parse_markdown(text).headings
        title = headings[0] if headings else None
        missing: list[str] = []
        if (
            title is None
            or title.level != 2
            or context.unit.qualified_name.rsplit(".", 1)[-1] not in title.text
        ):
            missing.append("level-2 title naming the unit")
        missing += [s for s in required_sections(context) if not _has_section(headings, s)]

        total = 1 + len(required_sections(context))
        explanation = f"{total - len(missing)}/{total} structural requirements met"
        if missing:
            explanation += f"; missing: {', '.join(missing)}"
        return MetricResult(value=ratio(total - len(missing), total), explanation=explanation)
