"""
Split generated Markdown into prose, code and headings.

A small, predictable subset of Markdown is enough here: fenced code blocks,
inline code spans and ATX headings (`## Title`). Anything else is prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

# Fences may be indented: models nest code blocks inside list items.
_FENCE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$", re.M | re.S
)
_INLINE_CODE = re.compile(r"(?P<ticks>`+)(?P<body>.+?)(?P=ticks)(?!`)", re.S)
_HEADING = re.compile(r"^(?P<level>#{1,6})[ \t]+(?P<text>.+?)[ \t#]*$", re.M)


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True, slots=True)
class MarkdownParts:
    prose: str
    """The text with code blocks, inline code and headings removed."""
    inline_code: tuple[str, ...]
    """Contents of inline code spans outside code blocks, in order."""
    code_blocks: tuple[str, ...]
    headings: tuple[Heading, ...]
    """Headings outside code blocks, in order."""


@cache
def parse_markdown(text: str) -> MarkdownParts:
    code_blocks = tuple(match["body"] for match in _FENCE.finditer(text))
    without_blocks = _FENCE.sub("\n", text)

    headings = tuple(
        Heading(level=len(match["level"]), text=match["text"].strip())
        for match in _HEADING.finditer(without_blocks)
    )
    inline_code = tuple(match["body"].strip() for match in _INLINE_CODE.finditer(without_blocks))

    prose = _INLINE_CODE.sub(" ", _HEADING.sub("\n", without_blocks))
    return MarkdownParts(
        prose=prose, inline_code=inline_code, code_blocks=code_blocks, headings=headings
    )
