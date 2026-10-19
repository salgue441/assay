import pytest
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from assay.extract._treesitter import descendants_before, end_line, node_text, start_line

_PARSER = Parser(Language(tspython.language()))


def parse(source: str) -> tuple[Node, bytes]:
    data = source.encode("utf-8")
    return _PARSER.parse(data).root_node, data


@pytest.mark.parametrize(
    ("source", "expected"),
    [("", 1), ("x = 1", 1), ("x = 1\n", 1), ("x = 1\ny = 2\n", 2), ("x = 1\n\n\n", 3)],
)
def test_end_line_does_not_count_a_trailing_empty_row(source: str, expected: int) -> None:
    root, _ = parse(source)

    assert end_line(root) == expected


def test_lines_are_one_based() -> None:
    root, _ = parse("\n\ndef f(): pass\n")
    function = root.named_children[0]

    assert (start_line(function), end_line(function)) == (3, 3)


def test_node_text_slices_utf8_bytes() -> None:
    root, data = parse('s = "ñandú"\n')

    assert node_text(root.named_children[0], data) == 's = "ñandú"'


def test_descendants_before_skips_nodes_that_cross_or_follow_the_boundary() -> None:
    root, data = parse("a = 1\nb = 2\n")
    boundary = data.index(b"2")

    texts = [node_text(n, data) for n in descendants_before(root, boundary)]

    assert "a = 1" in texts
    assert "b" in texts
    assert "b = 2" not in texts  # crosses the boundary: descended into, not yielded
    assert "2" not in texts  # starts at the boundary
