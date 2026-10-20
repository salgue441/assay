import pytest

from assay.metrics.rules import TypeScriptRules

RULES = TypeScriptRules()


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("function add(a: number, b = 2): number", ("a", "b")),
        ("async function load(id: string, ...rest: unknown[]): Promise<void>", ("id", "rest")),
        ("function f<T extends object>(x: T, y?: T): T", ("x", "y")),
        ("function* count(limit: number): Generator<number>", ("limit",)),
        ("function f(this: Window, event: Event)", ("event",)),
        ("function f({ a, b }: Options, [first]: string[], c: number)", ("c",)),
        ("const f = async <T,>(x: T): Promise<T>", ("x",)),
        ("const f = function named(y: string)", ("y",)),
        ("let f = x", ("x",)),
        ("constructor(private readonly url: string, public retries?: number)", ("url", "retries")),
        ("static async create(id: string): Promise<Client>", ("id",)),
        ("get size(): number", ()),
        ("set size(value: number)", ("value",)),
        ("send(body)", ("body",)),
        ("abstract run(input: string): void", ("input",)),
        ("clear?(): void", ()),
        ("private readonly handle = (event: Event): void", ("event",)),
        ("function f(cb: (inner: number) => void)", ("cb",)),
        ("class Box<T> extends Base", ()),
        ("interface Store", ()),
        ("function broken(", ()),
        ("", ()),
    ],
)
def test_parameters(signature: str, expected: tuple[str, ...]) -> None:
    assert RULES.parameters(signature) == expected


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("function add(a: number, b: number): number", "number"),
        ("async function load(): Promise<Map<string, number[]>>", "Promise<Map<string, number[]>>"),
        ("function isText(x: unknown): x is string", "x is string"),
        ("const f = (x: number): string | undefined", "string | undefined"),
        ("get(key: string): string | undefined", "string | undefined"),
        ("function f(cb: (inner: number) => void)", None),
        ("function f(): void", None),
        ("abstract run(): void", None),
        ("async function f(): Promise<void>", "Promise<void>"),
        ("constructor(a: number)", None),
        ("class Box", None),
        ("function broken(", None),
    ],
)
def test_return_annotation(signature: str, expected: str | None) -> None:
    assert RULES.return_annotation(signature) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Array",
        "Promise",
        "console",
        "log",
        "Map",
        "string",
        "number",
        "undefined",
        "map",
        "filter",
        "then",
        "JSON",
        "stringify",
        "Record",
        "this",
        "async",
        "length",
    ],
)
def test_ambient_names(name: str) -> None:
    assert name in RULES.ambient_names


def test_ambient_names_exclude_arbitrary_words() -> None:
    assert "wrapText" not in RULES.ambient_names


def test_constructor_name() -> None:
    assert RULES.constructor_name == "constructor"
