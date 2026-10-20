"""
Small helpers over tree-sitter nodes, shared by the language extractors.

tree-sitter reports 0-based rows and byte offsets into the UTF-8 encoded
source; these helpers convert to 1-based lines and decoded text.
"""

from __future__ import annotations

from collections.abc import Iterator

from tree_sitter import Node


def node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def start_line(node: Node) -> int:
    return node.start_point.row + 1


def end_line(node: Node) -> int:
    """
    Last line the node occupies.

    A node that ends at column 0 (for example a module whose source ends with
    a newline) stops before that row, so the row itself is not counted.
    """

    row, column = node.end_point
    return max(row if column == 0 else row + 1, 1)


def descendants_before(node: Node, end_byte: int) -> Iterator[Node]:
    """
    Yield every descendant of `node` that ends at or before `end_byte`.

    Subtrees that start at or after `end_byte` are skipped without descending.
    """

    stack = list(reversed(node.children))
    while stack:
        current = stack.pop()
        if current.start_byte >= end_byte:
            continue

        if current.end_byte <= end_byte:
            yield current

        stack.extend(reversed(current.children))


def text_without_comments(node: Node, source: bytes, *, start_byte: int, end_byte: int) -> str:
    """
    The source of `node` between two byte offsets, with comments removed.

    Every grammar the extractors use names its comment nodes `comment`.
    """

    pieces: list[bytes] = []
    cursor = start_byte
    for descendant in descendants_before(node, end_byte):
        if descendant.type == "comment" and descendant.start_byte >= cursor:
            pieces.append(source[cursor : descendant.start_byte])
            cursor = descendant.end_byte
    pieces.append(source[cursor:end_byte])

    return b"".join(pieces).decode("utf-8")
