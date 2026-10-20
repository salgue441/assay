import pytest

from assay.extract import Symbol, SymbolKind
from assay.extract._symbols import merge_redefinitions, normalize_signature


def make_symbol(name: str, *, signature: str, docstring: str | None) -> Symbol:
    return Symbol(
        qualified_name=f"m.{name}",
        name=name,
        kind=SymbolKind.FUNCTION,
        signature=signature,
        docstring=docstring,
        start_line=1,
        end_line=1,
        is_public=True,
    )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("def f(\n    a,\n    b,\n)", "def f(a, b)"),
        ("func f(\n\ta int,\n) error", "func f(a int) error"),
        ("x[\n  1\n]", "x[1]"),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_normalize_signature(header: str, expected: str) -> None:
    assert normalize_signature(header) == expected


def test_merge_keeps_the_first_and_borrows_a_missing_docstring() -> None:
    merged = merge_redefinitions(
        [
            make_symbol("f", signature="first", docstring=None),
            make_symbol("g", signature="g", docstring="G."),
            make_symbol("f", signature="second", docstring="Second."),
            make_symbol("f", signature="third", docstring="Third."),
        ]
    )

    assert [(s.name, s.signature, s.docstring) for s in merged] == [
        ("f", "first", "Second."),
        ("g", "g", "G."),
    ]
