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
