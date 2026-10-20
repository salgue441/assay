import subprocess
import sys
from textwrap import dedent

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from assay.extract import SourceLanguage, Symbol, SymbolInventory, SymbolKind, TypeScriptExtractor


@pytest.fixture(scope="module")
def extractor() -> TypeScriptExtractor:
    return TypeScriptExtractor()


def extract(
    extractor: TypeScriptExtractor, source: str, path: str = "src/lib/mod.ts"
) -> SymbolInventory:
    return extractor.extract(dedent(source).lstrip("\n"), module="lib.mod", path=path)


def by_name(inventory: SymbolInventory) -> dict[str, Symbol]:
    return {s.qualified_name: s for s in inventory.symbols}


def overview(inventory: SymbolInventory) -> list[tuple[str, SymbolKind, bool]]:
    return [(s.qualified_name, s.kind, s.is_public) for s in inventory.symbols]


def test_module_symbol_covers_the_whole_file(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        /** Helpers for strings. */
        import { x } from "y";

        export const n = 1;
        """,
    )

    module = inventory.symbols[0]
    assert module.qualified_name == "lib.mod"
    assert module.name == "mod"
    assert module.kind is SymbolKind.MODULE
    assert module.signature is None
    assert module.docstring == "Helpers for strings."
    assert module.is_public
    assert (module.start_line, module.end_line) == (1, 4)
    assert inventory.language is SourceLanguage.TYPESCRIPT
    assert inventory.path == "src/lib/mod.ts"
    assert not inventory.has_syntax_errors


def test_empty_source_yields_only_the_module(extractor: TypeScriptExtractor) -> None:
    inventory = extractor.extract("", module="empty")

    assert overview(inventory) == [("empty", SymbolKind.MODULE, True)]
    assert inventory.symbols[0].end_line == 1


def test_extracts_top_level_declarations(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        export function add(a: number, b: number): number { return a + b; }
        function local(): void {}
        export function* count(): Generator<number> {}
        export const arrow = (x: string): string => x;
        const expression = function (y: number) { return y; };
        export class Box {}
        export abstract class Shape {}
        export interface Named { name: string; }
        export type Alias = string;
        export enum Color { Red }
        export namespace Space {}
        export const value = 42;
        """,
    )

    assert overview(inventory) == [
        ("lib.mod", SymbolKind.MODULE, True),
        ("lib.mod.add", SymbolKind.FUNCTION, True),
        ("lib.mod.local", SymbolKind.FUNCTION, False),
        ("lib.mod.count", SymbolKind.FUNCTION, True),
        ("lib.mod.arrow", SymbolKind.FUNCTION, True),
        ("lib.mod.expression", SymbolKind.FUNCTION, False),
        ("lib.mod.Box", SymbolKind.CLASS, True),
        ("lib.mod.Shape", SymbolKind.CLASS, True),
        ("lib.mod.Named", SymbolKind.CLASS, True),
    ]


def test_class_members(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        export class Client {
          name: string;
          constructor(private readonly url: string) {}
          send(body: string): void {}
          protected retry(): void {}
          private reset(): void {}
          #secret(): void {}
          _internal(): void {}
          static create(): Client { return new Client(""); }
          get size(): number { return 0; }
          set size(value: number) {}
          handle = (event: Event): void => {};
          [Symbol.iterator]() {}
        }
        """,
    )

    assert overview(inventory)[1:] == [
        ("lib.mod.Client", SymbolKind.CLASS, True),
        ("lib.mod.Client.constructor", SymbolKind.METHOD, True),
        ("lib.mod.Client.send", SymbolKind.METHOD, True),
        ("lib.mod.Client.retry", SymbolKind.METHOD, True),
        ("lib.mod.Client.reset", SymbolKind.METHOD, False),
        ("lib.mod.Client.#secret", SymbolKind.METHOD, False),
        ("lib.mod.Client._internal", SymbolKind.METHOD, False),
        ("lib.mod.Client.create", SymbolKind.METHOD, True),
        ("lib.mod.Client.size", SymbolKind.METHOD, True),
        ("lib.mod.Client.handle", SymbolKind.METHOD, True),
    ]


def test_abstract_class_and_interface_members(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        export abstract class Base {
          abstract run(input: string): Promise<void>;
          stop(): void {}
        }
        export interface Store {
          new (seed: number): Store;
          (key: string): string;
          readonly size: number;
          get(key: string): string | undefined;
          clear?(): void;
        }
        """,
    )

    symbols = by_name(inventory)
    assert overview(inventory)[1:] == [
        ("lib.mod.Base", SymbolKind.CLASS, True),
        ("lib.mod.Base.run", SymbolKind.METHOD, True),
        ("lib.mod.Base.stop", SymbolKind.METHOD, True),
        ("lib.mod.Store", SymbolKind.CLASS, True),
        ("lib.mod.Store.get", SymbolKind.METHOD, True),
        ("lib.mod.Store.clear", SymbolKind.METHOD, True),
    ]
    assert symbols["lib.mod.Base"].signature == "abstract class Base"
    assert symbols["lib.mod.Base.run"].signature == "abstract run(input: string): Promise<void>"
    assert symbols["lib.mod.Store"].signature == "interface Store"
    assert symbols["lib.mod.Store.get"].signature == "get(key: string): string | undefined"
    assert symbols["lib.mod.Store.clear"].signature == "clear?(): void"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("export function f(a: number, b = 2): number {}", "function f(a: number, b = 2): number"),
        ("export async function f(): Promise<void> {}", "async function f(): Promise<void>"),
        ("function f<T extends object>(x: T): T {}", "function f<T extends object>(x: T): T"),
        ("function f(): void;", "function f(): void"),
        ("export default function f() {}", "function f()"),
        ("const f = async <T,>(x: T): Promise<T> => x;", "const f = async <T,>(x: T): Promise<T>"),
        ("let f = x => x;", "let f = x"),
        ("var f = function named(y: string) { return y; };", "var f = function named(y: string)"),
        ("class C<T> extends B<T> implements I {}", "class C<T> extends B<T> implements I"),
        ("interface I extends J, K {}", "interface I extends J, K"),
        (
            "function f(a: number, // first\n  b: string,\n): void {}",
            "function f(a: number, b: string): void",
        ),
        ("function f(\n  a: number,\n  b: string,\n) {}", "function f(a: number, b: string)"),
        ("@sealed\nexport class C {}", "class C"),
        ("@sealed\nclass C {}", "class C"),
    ],
)
def test_signature_is_the_header_on_one_line(
    extractor: TypeScriptExtractor, source: str, expected: str
) -> None:
    symbol = extractor.extract(source, module="m").symbols[1]

    assert symbol.signature == expected


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (
            "constructor(private a: number, public b?: string) {}",
            "constructor(private a: number, public b?: string)",
        ),
        (
            "static async load(id: string): Promise<C> {}",
            "static async load(id: string): Promise<C>",
        ),
        ("get size(): number { return 0; }", "get size(): number"),
        (
            "private readonly handle = (e: Event): void => {};",
            "private readonly handle = (e: Event): void",
        ),
        ("@HostListener('click')\nonClick(e: Event): void {}", "onClick(e: Event): void"),
        ("overloaded(a: string): void;", "overloaded(a: string): void"),
    ],
)
def test_member_signatures(extractor: TypeScriptExtractor, member: str, expected: str) -> None:
    symbol = extractor.extract(f"class C {{\n{member}\n}}\n", module="m").symbols[2]

    assert symbol.signature == expected


def test_jsdoc_is_cleaned(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        /**
         * Adds two numbers.
         *
         * @param a - the first
         *   continued
         * @returns the sum
         */
        export function add(a: number, b: number): number { return a + b; }
        """,
    )

    assert by_name(inventory)["lib.mod.add"].docstring == (
        "Adds two numbers.\n\n@param a - the first\n  continued\n@returns the sum"
    )


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("/** One line. */", "One line."),
        ("/**One line.*/", "One line."),
        ("/** */", None),
        ("/**/", None),
        ("/* Plain block. */", None),
        ("// Line comment.", None),
    ],
)
def test_only_jsdoc_blocks_are_docstrings(
    extractor: TypeScriptExtractor, comment: str, expected: str | None
) -> None:
    source = f"const x = 1;\n{comment}\nfunction f() {{}}\n"

    assert by_name(extractor.extract(source, module="m"))["m.f"].docstring == expected


def test_jsdoc_attaches_through_export_and_decorators(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        import { x } from "y";

        /** The widget. */
        @Component({ selector: "w" })
        export class Widget {
          /** Builds it. */
          constructor() {}

          /** Handles clicks. */
          @HostListener("click")
          onClick(): void {}

          /** The handler. */
          handle = () => {};
        }

        /** Doubles. */
        export const double = (n: number) => n * 2;
        """,
    )

    symbols = by_name(inventory)
    assert symbols["lib.mod"].docstring is None
    assert symbols["lib.mod.Widget"].docstring == "The widget."
    assert symbols["lib.mod.Widget.constructor"].docstring == "Builds it."
    assert symbols["lib.mod.Widget.onClick"].docstring == "Handles clicks."
    assert symbols["lib.mod.Widget.handle"].docstring == "The handler."
    assert symbols["lib.mod.double"].docstring == "Doubles."


def test_jsdoc_must_immediately_precede_the_declaration(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        /** Not for f. */
        const unrelated = 1;
        function f() {}

        /** Separated by a line comment. */
        // note
        function g() {}
        """,
    )

    symbols = by_name(inventory)
    assert symbols["lib.mod.f"].docstring is None
    assert symbols["lib.mod.g"].docstring is None


def test_leading_jsdoc_documents_the_first_declaration_not_the_module(
    extractor: TypeScriptExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        /** Adds. */
        export function add() {}
        """,
    )

    assert inventory.symbols[0].docstring is None
    assert inventory.symbols[1].docstring == "Adds."


def test_module_docstring_before_a_declaration_doc(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        /** The module. */

        /** Adds. */
        export function add() {}
        """,
    )

    assert [s.docstring for s in inventory.symbols] == ["The module.", "Adds."]


def test_export_lists_make_local_declarations_public(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        function a() {}
        function b() {}
        const c = () => 1;
        class D {}
        function e() {}
        function _f() {}
        export { a, b as renamed, c, _f };
        export { e } from "./other";
        export default D;
        """,
    )

    assert {s.name: s.is_public for s in inventory.symbols[1:]} == {
        "a": True,
        "b": True,
        "c": True,
        "D": True,
        "e": False,
        "_f": False,
    }


def test_anonymous_default_exports_are_skipped(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        export default function () {}
        export default class {}
        """,
    )

    assert [s.kind for s in inventory.symbols] == [SymbolKind.MODULE]


def test_definitions_inside_functions_and_namespaces_are_skipped(
    extractor: TypeScriptExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        export function outer() {
          function helper() {}
          class Local {}
        }
        namespace Space { export function inner() {} }
        if (true) { function conditional() {} }
        """,
    )

    assert list(by_name(inventory)) == ["lib.mod", "lib.mod.outer"]


def test_overloads_merge_into_the_first_with_a_docstring_fallback(
    extractor: TypeScriptExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        export function parse(text: string): number;
        export function parse(text: string, radix: number): number;
        /** Parses a number. */
        export function parse(text: string, radix = 10): number { return 0; }

        export class Box {
          get size(): number { return 0; }
          /** The size. */
          set size(value: number) {}
        }
        """,
    )

    symbols = by_name(inventory)
    assert list(symbols) == ["lib.mod", "lib.mod.parse", "lib.mod.Box", "lib.mod.Box.size"]
    assert symbols["lib.mod.parse"].signature == "function parse(text: string): number"
    assert symbols["lib.mod.parse"].docstring == "Parses a number."
    assert (symbols["lib.mod.parse"].start_line, symbols["lib.mod.parse"].end_line) == (1, 1)
    assert symbols["lib.mod.Box.size"].signature == "get size(): number"
    assert symbols["lib.mod.Box.size"].docstring == "The size."


def test_line_numbers_exclude_decorators(extractor: TypeScriptExtractor) -> None:
    inventory = extract(
        extractor,
        """
        import { x } from "y";

        @Injectable()
        export class Service {
          @Input()
          handle = () => {};

          run(): void {
            x();
          }
        }

        export const multi = () => {
          return 1;
        };
        """,
    )

    lines = {s.qualified_name: (s.start_line, s.end_line) for s in inventory.symbols}
    assert lines == {
        "lib.mod": (1, 15),
        "lib.mod.Service": (4, 11),
        "lib.mod.Service.handle": (6, 6),
        "lib.mod.Service.run": (8, 10),
        "lib.mod.multi": (13, 15),
    }


def test_tsx_files_use_the_tsx_grammar(extractor: TypeScriptExtractor) -> None:
    source = """
        /** Renders the title. */
        export const Title = ({ text }: Props) => <h1 className="t">{text}</h1>;

        export function App(): JSX.Element {
          return <div><Title text="hi" /></div>;
        }
        """

    tsx = extract(extractor, source, path="src/App.tsx")
    ts = extract(extractor, source, path="src/App.ts")

    assert not tsx.has_syntax_errors
    assert overview(tsx)[1:] == [
        ("lib.mod.Title", SymbolKind.FUNCTION, True),
        ("lib.mod.App", SymbolKind.FUNCTION, True),
    ]
    assert by_name(tsx)["lib.mod.Title"].signature == "const Title = ({ text }: Props)"
    assert by_name(tsx)["lib.mod.Title"].docstring == "Renders the title."
    assert ts.has_syntax_errors


def test_angle_bracket_casts_parse_as_typescript(extractor: TypeScriptExtractor) -> None:
    inventory = extract(extractor, "export const n = <number>value;\nexport function f() {}\n")

    assert not inventory.has_syntax_errors
    assert "lib.mod.f" in by_name(inventory)


def test_syntax_errors_are_flagged_and_valid_parts_still_extracted(
    extractor: TypeScriptExtractor,
) -> None:
    inventory = extract(
        extractor,
        """
        export function good(): void {}

        function broken(
        """,
    )

    assert inventory.has_syntax_errors
    assert "lib.mod.good" in by_name(inventory)


def test_non_ascii_source_is_sliced_by_bytes_correctly(extractor: TypeScriptExtractor) -> None:
    source = (
        '/** Módulo — ünïcode. */\nimport "ñ";\n\n'
        'export function saludo(nombre = "señor"): string {}\n'
    )

    inventory = extractor.extract(source, module="m")

    assert inventory.symbols[0].docstring == "Módulo — ünïcode."
    assert inventory.symbols[1].signature == 'function saludo(nombre = "señor"): string'


@settings(max_examples=200, deadline=None)
@given(source=st.text(), tsx=st.booleans())
def test_never_raises_and_lines_stay_in_range(source: str, tsx: bool) -> None:
    path = "fuzz.tsx" if tsx else "fuzz.ts"
    inventory = TypeScriptExtractor().extract(source, module="fuzz", path=path)

    module = inventory.symbols[0]
    for symbol in inventory.symbols:
        assert 1 <= symbol.start_line <= symbol.end_line <= module.end_line


@settings(max_examples=100, deadline=None)
@given(
    parts=st.lists(
        st.sampled_from(
            [
                "export ",
                "class C ",
                "{ ",
                "} ",
                "function f",
                "(a: number) ",
                "=> ",
                "const g = ",
                "/** doc */\n",
                "@dec ",
                "private ",
                "constructor() {}\n",
                "interface I ",
                "m(): void;\n",
                "\n",
            ]
        ),
        max_size=40,
    )
)
def test_never_raises_on_typescript_like_fragments(parts: list[str]) -> None:
    inventory = TypeScriptExtractor().extract("".join(parts), module="fuzz", path="f.ts")

    assert inventory.symbols[0].kind is SymbolKind.MODULE


def test_large_files_extract_without_crashing() -> None:
    """
    Regression: py-tree-sitter 0.26.0 corrupts node memory on files of a few
    hundred definitions and the interpreter segfaults. Run in a subprocess so
    a crash fails this test instead of killing the whole test run.
    """

    source = "".join(
        f"/** Doc {i}. */\nexport function f{i}(x: number): number {{\n  return x;\n}}\n\n"
        for i in range(1000)
    )
    count_symbols = (
        "import sys\n"
        "from assay.extract import TypeScriptExtractor\n"
        "inventory = TypeScriptExtractor().extract(sys.stdin.read(), module='big', path='big.ts')\n"
        "print(len(inventory.symbols))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", count_symbols],
        input=source,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"extraction crashed (exit {result.returncode}): {result.stderr}"
    assert result.stdout.strip() == "1001"
