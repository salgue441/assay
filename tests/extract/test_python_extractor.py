import subprocess
import sys
from textwrap import dedent

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from assay.extract import PythonExtractor, SourceLanguage, Symbol, SymbolInventory, SymbolKind


@pytest.fixture(scope="module")
def extractor() -> PythonExtractor:
    return PythonExtractor()


def extract(extractor: PythonExtractor, source: str, module: str = "pkg.mod") -> SymbolInventory:
    return extractor.extract(dedent(source).lstrip("\n"), module=module, path="pkg/mod.py")


def by_name(inventory: SymbolInventory) -> dict[str, Symbol]:
    return {s.qualified_name: s for s in inventory.symbols}


def test_module_symbol_covers_the_whole_file(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        '''
        """Module docstring."""

        x = 1
        ''',
    )

    module = inventory.symbols[0]
    assert module.qualified_name == "pkg.mod"
    assert module.name == "mod"
    assert module.kind is SymbolKind.MODULE
    assert module.docstring == "Module docstring."
    assert (module.start_line, module.end_line) == (1, 3)
    assert inventory.language is SourceLanguage.PYTHON
    assert inventory.path == "pkg/mod.py"
    assert not inventory.has_syntax_errors


def test_empty_source_yields_only_the_module(extractor: PythonExtractor) -> None:
    inventory = extractor.extract("", module="empty")

    assert [s.kind for s in inventory.symbols] == [SymbolKind.MODULE]
    assert inventory.symbols[0].end_line == 1


def test_extracts_every_top_level_definition(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        def first() -> None: ...
        def second() -> None: ...
        class Third: ...
        """,
    )

    assert list(by_name(inventory)) == [
        "pkg.mod",
        "pkg.mod.first",
        "pkg.mod.second",
        "pkg.mod.Third",
    ]


def test_classes_methods_and_nested_classes(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        class Outer:
            def method(self) -> None: ...

            class Inner:
                def deep(self) -> None: ...
        """,
    )

    symbols = by_name(inventory)
    assert symbols["pkg.mod.Outer"].kind is SymbolKind.CLASS
    assert symbols["pkg.mod.Outer.method"].kind is SymbolKind.METHOD
    assert symbols["pkg.mod.Outer.Inner"].kind is SymbolKind.CLASS
    assert symbols["pkg.mod.Outer.Inner.deep"].kind is SymbolKind.METHOD


def test_skips_definitions_inside_functions_and_control_flow(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        def outer() -> None:
            def helper() -> None: ...
            class Local: ...

        if True:
            def conditional() -> None: ...
        """,
    )

    assert list(by_name(inventory)) == ["pkg.mod", "pkg.mod.outer"]


def test_decorators_are_unwrapped_and_excluded_from_signature(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        @cache
        @other(arg=1)
        def cached(x: int) -> int:
            return x
        """,
    )

    symbol = by_name(inventory)["pkg.mod.cached"]
    assert symbol.signature == "def cached(x: int) -> int"
    assert symbol.start_line == 3


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("async def fetch(url: str) -> bytes: ...", "async def fetch(url: str) -> bytes"),
        ("class Plain: ...", "class Plain"),
        ("class Child(Base, metaclass=Meta): ...", "class Child(Base, metaclass=Meta)"),
        ("def f(x: int):  # trailing comment\n    pass", "def f(x: int)"),
        (
            "def f(\n    a: int,  # first\n    b: dict[str, int],\n) -> None:\n    pass",
            "def f(a: int, b: dict[str, int]) -> None",
        ),
        ("def f(x=(1, )): ...", "def f(x=(1, ))"),
    ],
)
def test_signature_is_normalized(extractor: PythonExtractor, source: str, expected: str) -> None:
    symbol = extractor.extract(source, module="m").symbols[1]

    assert symbol.signature == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('"""Summary.\n\n    Detail line.\n    """', "Summary.\n\nDetail line."),
        ('"Single quotes."', "Single quotes."),
        ('r"""Raw \\d+ pattern."""', "Raw \\d+ pattern."),
        ('"""Caf\\u00e9 escapes decode."""', "Café escapes decode."),
        ('"Implicit " "concatenation."', "Implicit concatenation."),
        ('""""""', None),
        ('f"""Not a {docstring}."""', None),
        ('b"""Bytes are not docstrings."""', None),
        ("x = 1", None),
    ],
)
def test_docstring_follows_python_rules(
    extractor: PythonExtractor, body: str, expected: str | None
) -> None:
    source = f"def f():\n    {body}\n    pass\n"

    assert extractor.extract(source, module="m").symbols[1].docstring == expected


def test_docstring_must_be_the_first_statement(extractor: PythonExtractor) -> None:
    source = 'def f():\n    x = 1\n    """Too late."""\n'

    assert extractor.extract(source, module="m").symbols[1].docstring is None


def test_redefinitions_keep_the_first_with_a_docstring_fallback(
    extractor: PythonExtractor,
) -> None:
    inventory = extract(
        extractor,
        '''
        class Config:
            @property
            def timeout(self) -> int:
                return self._timeout

            @timeout.setter
            def timeout(self, value: int) -> None:
                """Seconds before a request is abandoned."""
                self._timeout = value
        ''',
    )

    names = [s.qualified_name for s in inventory.symbols]
    assert names.count("pkg.mod.Config.timeout") == 1

    timeout = by_name(inventory)["pkg.mod.Config.timeout"]
    assert timeout.signature == "def timeout(self) -> int"
    assert timeout.docstring == "Seconds before a request is abandoned."


def test_syntax_errors_are_flagged_and_valid_parts_still_extracted(
    extractor: PythonExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        def good() -> None: ...

        def broken(:
        """,
    )

    assert inventory.has_syntax_errors
    assert "pkg.mod.good" in by_name(inventory)


def test_line_numbers_are_one_based_and_inclusive(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        import os


        def f() -> None:
            pass
        """,
    )

    f = by_name(inventory)["pkg.mod.f"]
    assert (f.start_line, f.end_line) == (4, 5)


def test_non_ascii_source_is_sliced_by_bytes_correctly(extractor: PythonExtractor) -> None:
    source = '"""Módulo — ünïcode."""\n\ndef saludo(nombre: str = "señor") -> str: ...\n'

    inventory = extractor.extract(source, module="m")

    assert inventory.symbols[0].docstring == "Módulo — ünïcode."
    assert inventory.symbols[1].signature == 'def saludo(nombre: str = "señor") -> str'


@settings(max_examples=200, deadline=None)
@given(source=st.text())
def test_never_raises_and_lines_stay_in_range(source: str) -> None:
    inventory = PythonExtractor().extract(source, module="fuzz")

    module = inventory.symbols[0]
    for symbol in inventory.symbols:
        assert 1 <= symbol.start_line <= symbol.end_line <= module.end_line


def test_large_files_extract_without_crashing() -> None:
    """
    Regression: py-tree-sitter 0.26.0 corrupts node memory on files of a few
    hundred definitions and the interpreter segfaults. Run in a subprocess so
    a crash fails this test instead of killing the whole test run.
    """

    source = "".join(f"def f{i}(x: int) -> int:\n    return x\n\n" for i in range(1000))
    count_symbols = (
        "import sys\n"
        "from assay.extract import PythonExtractor\n"
        "print(len(PythonExtractor().extract(sys.stdin.read(), module='big').symbols))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", count_symbols],
        input=source,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"extraction crashed (exit {result.returncode}): {result.stderr}"
    assert result.stdout.strip() == "1001"


def test_public_means_no_leading_underscore(extractor: PythonExtractor) -> None:
    inventory = extract(
        extractor,
        """
        class Client:
            def __init__(self) -> None: ...
            def send(self) -> None: ...
            def _retry(self) -> None: ...

        def _helper() -> None: ...
        """,
        module="pkg._internal",
    )

    assert {s.qualified_name: s.is_public for s in inventory.symbols} == {
        "pkg._internal": False,
        "pkg._internal.Client": True,
        "pkg._internal.Client.__init__": False,
        "pkg._internal.Client.send": True,
        "pkg._internal.Client._retry": False,
        "pkg._internal._helper": False,
    }
