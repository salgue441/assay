import subprocess
import sys
from textwrap import dedent

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from assay.extract import GoExtractor, SourceLanguage, Symbol, SymbolInventory, SymbolKind


@pytest.fixture(scope="module")
def extractor() -> GoExtractor:
    return GoExtractor()


def extract(extractor: GoExtractor, source: str) -> SymbolInventory:
    return extractor.extract(
        dedent(source).lstrip("\n"), module="internal.server", path="internal/server/server.go"
    )


def by_name(inventory: SymbolInventory) -> dict[str, Symbol]:
    return {s.qualified_name: s for s in inventory.symbols}


def overview(inventory: SymbolInventory) -> list[tuple[str, SymbolKind, bool]]:
    return [(s.qualified_name, s.kind, s.is_public) for s in inventory.symbols]


def test_module_symbol_is_the_package(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        // Package server serves requests.
        //
        // It is small.
        package server

        import "fmt"
        """,
    )

    module = inventory.symbols[0]
    assert module.qualified_name == "internal.server"
    assert module.name == "server"
    assert module.kind is SymbolKind.MODULE
    assert module.signature is None
    assert module.docstring == "Package server serves requests.\n\nIt is small."
    assert module.is_public
    assert (module.start_line, module.end_line) == (1, 6)
    assert inventory.language is SourceLanguage.GO
    assert inventory.path == "internal/server/server.go"
    assert not inventory.has_syntax_errors


def test_empty_source_yields_only_the_module(extractor: GoExtractor) -> None:
    inventory = extractor.extract("", module="empty")

    assert overview(inventory) == [("empty", SymbolKind.MODULE, True)]
    assert inventory.symbols[0].end_line == 1
    assert inventory.symbols[0].docstring is None


def test_extracts_functions_methods_and_types(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        type Server struct{ addr string }

        type handler interface {
            Serve(ctx context.Context) error
            close()
            io.Reader
        }

        type ID int
        type Alias = Server
        type Func func() error

        func New() *Server { return nil }
        func helper() {}
        func (s *Server) Start() error { return nil }
        func (s Server) stop() {}
        func (Server) Addr() string { return "" }
        func (b *Box[K, V]) Get(key K) V { return b.m[key] }
        func (o *Outside) Run() {}
        """,
    )

    assert overview(inventory) == [
        ("internal.server", SymbolKind.MODULE, True),
        ("internal.server.Server", SymbolKind.CLASS, True),
        ("internal.server.handler", SymbolKind.CLASS, False),
        ("internal.server.handler.Serve", SymbolKind.METHOD, True),
        ("internal.server.handler.close", SymbolKind.METHOD, False),
        ("internal.server.New", SymbolKind.FUNCTION, True),
        ("internal.server.helper", SymbolKind.FUNCTION, False),
        ("internal.server.Server.Start", SymbolKind.METHOD, True),
        ("internal.server.Server.stop", SymbolKind.METHOD, False),
        ("internal.server.Server.Addr", SymbolKind.METHOD, True),
        ("internal.server.Box.Get", SymbolKind.METHOD, True),
        ("internal.server.Outside.Run", SymbolKind.METHOD, True),
    ]


def test_grouped_type_declarations(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        // Shared types.
        type (
            // Request is a request.
            Request struct {
                Path string
            }

            Response struct{}

            Status int
        )

        // Lonely is alone in its group.
        type (
            Lonely struct{}
        )
        """,
    )

    symbols = by_name(inventory)
    assert list(symbols) == [
        "internal.server",
        "internal.server.Request",
        "internal.server.Response",
        "internal.server.Lonely",
    ]
    assert symbols["internal.server.Request"].docstring == "Request is a request."
    assert symbols["internal.server.Response"].docstring is None
    assert symbols["internal.server.Lonely"].docstring == "Lonely is alone in its group."
    request = symbols["internal.server.Request"]
    assert (request.start_line, request.end_line) == (6, 8)
    lonely = symbols["internal.server.Lonely"]
    assert (lonely.start_line, lonely.end_line) == (16, 18)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("func F() {}", "func F()"),
        (
            "func F(a, b int, rest ...string) (int, error) {}",
            "func F(a, b int, rest ...string) (int, error)",
        ),
        (
            "func F[T any, K comparable](m map[K]T) []T {}",
            "func F[T any, K comparable](m map[K]T) []T",
        ),
        (
            "func (s *Server) Start(ctx context.Context) (err error) {}",
            "func (s *Server) Start(ctx context.Context) (err error)",
        ),
        ("func F(\n\ta int, // first\n\tb string,\n) error {}", "func F(a int, b string) error"),
        ("func F() /* note */ error {}", "func F() error"),
        ("func asm(x int) int", "func asm(x int) int"),
        ("type S struct {\n\tA int\n}", "type S struct"),
        ("type Box[T any] struct{ v T }", "type Box[T any] struct"),
        ("type I interface {\n\tM()\n}", "type I interface"),
    ],
)
def test_signature_is_the_header_on_one_line(
    extractor: GoExtractor, source: str, expected: str
) -> None:
    symbol = extractor.extract(f"package p\n\n{source}\n", module="p").symbols[1]

    assert symbol.signature == expected


def test_interface_method_signature(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        type Store interface {
            // Get returns the value.
            Get(ctx context.Context, key string) (value []byte, err error)
        }
        """,
    )

    get = by_name(inventory)["internal.server.Store.Get"]
    assert get.signature == "Get(ctx context.Context, key string) (value []byte, err error)"
    assert get.docstring == "Get returns the value."
    assert (get.start_line, get.end_line) == (5, 5)


def test_doc_comment_is_the_contiguous_line_comment_block(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        // Unrelated note.

        // Start starts the server.
        //
        //	indented example
        //go:noinline
        func Start() {}

        // Detached by a blank line.

        func Detached() {}

        /* A block comment. */
        func Block() {}

        var x = 1 // trailing note
        func Trailing() {}

        //nolint:errcheck
        func OnlyDirective() {}
        """,
    )

    symbols = by_name(inventory)
    assert symbols["internal.server.Start"].docstring == (
        "Start starts the server.\n\n\tindented example"
    )
    assert symbols["internal.server.Detached"].docstring is None
    assert symbols["internal.server.Block"].docstring is None
    assert symbols["internal.server.Trailing"].docstring is None
    assert symbols["internal.server.OnlyDirective"].docstring is None


def test_line_comments_above_a_block_comment_are_not_joined(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        // Far above.
        /* block */
        // Close.
        func F() {}
        """,
    )

    assert by_name(inventory)["internal.server.F"].docstring == "Close."


def test_repeated_init_functions_merge(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        func init() {}

        // Second init registers handlers.
        func init() {}
        """,
    )

    init = by_name(inventory)["internal.server.init"]
    assert [s.qualified_name for s in inventory.symbols].count("internal.server.init") == 1
    assert (init.start_line, init.docstring) == (3, "Second init registers handlers.")
    assert not init.is_public


@pytest.mark.parametrize(
    ("name", "public"),
    [("Serve", True), ("serve", False), ("_serve", False), ("Ünicode", True), ("été", False)],
)
def test_public_means_exported(extractor: GoExtractor, name: str, public: bool) -> None:
    inventory = extractor.extract(f"package p\n\nfunc {name}() {{}}\n", module="p")

    assert inventory.symbols[1].is_public is public


def test_line_numbers_are_one_based_and_inclusive(extractor: GoExtractor) -> None:
    inventory = extract(
        extractor,
        """
        package server

        // Run runs.
        func Run() {
            return
        }
        """,
    )

    run = by_name(inventory)["internal.server.Run"]
    assert (run.start_line, run.end_line) == (4, 6)


def test_syntax_errors_are_flagged_and_valid_parts_still_extracted(
    extractor: GoExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        package server

        func Good() {}

        func Broken(
        """,
    )

    assert inventory.has_syntax_errors
    assert "internal.server.Good" in by_name(inventory)


def test_non_ascii_source_is_sliced_by_bytes_correctly(extractor: GoExtractor) -> None:
    source = (
        '// Paquete — ünïcode.\npackage p\n\nfunc Saludo(nombre string) string { return "ñ" }\n'
    )

    inventory = extractor.extract(source, module="p")

    assert inventory.symbols[0].docstring == "Paquete — ünïcode."
    assert inventory.symbols[1].signature == "func Saludo(nombre string) string"


@settings(max_examples=200, deadline=None)
@given(source=st.text())
def test_never_raises_and_lines_stay_in_range(source: str) -> None:
    inventory = GoExtractor().extract(source, module="fuzz")

    module = inventory.symbols[0]
    for symbol in inventory.symbols:
        assert 1 <= symbol.start_line <= symbol.end_line <= module.end_line


@settings(max_examples=100, deadline=None)
@given(
    parts=st.lists(
        st.sampled_from(
            [
                "package p\n",
                "func ",
                "(s *S) ",
                "(r ",
                "F",
                "() ",
                "{ ",
                "}\n",
                "type ",
                "T struct ",
                "I interface ",
                "M()\n",
                "// doc\n",
                "(\n",
                ")\n",
            ]
        ),
        max_size=40,
    )
)
def test_never_raises_on_go_like_fragments(parts: list[str]) -> None:
    inventory = GoExtractor().extract("".join(parts), module="fuzz")

    assert inventory.symbols[0].kind is SymbolKind.MODULE


def test_large_files_extract_without_crashing() -> None:
    """
    Regression: py-tree-sitter 0.26.0 corrupts node memory on files of a few
    hundred definitions and the interpreter segfaults. Run in a subprocess so
    a crash fails this test instead of killing the whole test run.
    """

    source = "package big\n\n" + "".join(
        f"// F{i} returns x.\nfunc F{i}(x int) int {{\n\treturn x\n}}\n\n" for i in range(1000)
    )
    count_symbols = (
        "import sys\n"
        "from assay.extract import GoExtractor\n"
        "print(len(GoExtractor().extract(sys.stdin.read(), module='big').symbols))\n"
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
