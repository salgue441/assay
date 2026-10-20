"""
TypeScript and Go end to end: a real repository through discovery,
extraction and unit selection, and its units through the static metrics.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from assay.corpus import CorpusBuilder, GitSnapshotFetcher, SnapshotCorpus, Unit
from assay.extract import SourceLanguage, Symbol
from assay.metrics import MetricContext, get_rules, static_metrics

from ..local_repo import LocalRepo

MakeRepo = Callable[..., LocalRepo]

GREETER_TS = """/** Greeting helpers. */
import { shout } from "./shout";

/**
 * Greets people.
 */
export class Greeter {
  constructor(private readonly greeting: string) {}

  /** Greets one person. */
  greet(name: string): string {
    return `${this.greeting}, ${name}`;
  }

  private whisper(text: string): string {
    return text.toLowerCase();
  }
}

/** Says hello. */
export function hello(name: string, excited = false): string {
  return excited ? shout(name) : `Hello, ${name}`;
}

function helper(): void {
  return;
}
"""

APP_TSX = """export const App = ({ title }: Props) => (
  <h1>{title}</h1>
);
"""

SERVER_GO = """// Package server serves greetings.
package server

import "fmt"

// Server greets over the network.
type Server struct {
\taddr string
}

// New makes a server listening on addr.
func New(addr string, port int) *Server {
\treturn &Server{addr: fmt.Sprintf("%s:%d", addr, port)}
}

// Start starts serving.
func (s *Server) Start(ctx context.Context) error {
\treturn nil
}

func (s *Server) stop() {
\ts.addr = ""
}
"""

MAIN_GO = """package main

func main() {
\tprintln("hi")
}
"""


def build(
    make_repo: MakeRepo, tmp_path: Path, language: SourceLanguage, files: dict[str, str]
) -> SnapshotCorpus:
    repo = make_repo(f"owner/{language.value}")
    repo.commit(files)
    spec = repo.spec().model_copy(update={"language": language})
    return CorpusBuilder(GitSnapshotFetcher(tmp_path / "cache")).build(spec, 0)


def contexts(corpus: SnapshotCorpus, language: SourceLanguage) -> dict[str, MetricContext]:
    symbols: list[Symbol] = [s for i in corpus.inventories for s in i.symbols]
    rules = get_rules(language)
    return {u.qualified_name: MetricContext.build(u, symbols, rules) for u in corpus.units}


def units(corpus: SnapshotCorpus) -> list[tuple[str, str]]:
    return [(u.path, u.qualified_name) for u in corpus.units]


def scores(text: str, context: MetricContext) -> dict[str, float]:
    return {m.name: m.score(text, context).value for m in static_metrics()}


@pytest.fixture
def typescript(make_repo: MakeRepo, tmp_path: Path) -> SnapshotCorpus:
    return build(
        make_repo,
        tmp_path,
        SourceLanguage.TYPESCRIPT,
        {
            "package.json": "{}\n",
            "src/greeter.ts": GREETER_TS,
            "src/greeter.test.ts": "test('greets', () => {});\n",
            "src/types.d.ts": "declare const x: number;\n",
            "src/components/App.tsx": APP_TSX,
            "src/components/index.ts": 'export { App } from "./App";\n',
            "node_modules/dep/index.ts": "export function dep() {}\n",
        },
    )


@pytest.fixture
def go(make_repo: MakeRepo, tmp_path: Path) -> SnapshotCorpus:
    return build(
        make_repo,
        tmp_path,
        SourceLanguage.GO,
        {
            "go.mod": "module example.com/greet\n",
            "internal/server/server.go": SERVER_GO,
            "internal/server/server_test.go": "package server\n\nfunc TestNew() {}\n",
            "cmd/greet/main.go": MAIN_GO,
        },
    )


def test_typescript_files_are_discovered_extracted_and_cut_into_units(
    typescript: SnapshotCorpus,
) -> None:
    assert [(i.path, i.module, i.has_syntax_errors) for i in typescript.inventories] == [
        ("src/greeter.ts", "greeter", False),
        ("src/components/App.tsx", "components.App", False),
        ("src/components/index.ts", "components", False),
    ]
    assert units(typescript) == [
        ("src/greeter.ts", "greeter"),
        ("src/greeter.ts", "greeter.Greeter"),
        ("src/greeter.ts", "greeter.hello"),
        ("src/components/App.tsx", "components.App"),
        ("src/components/App.tsx", "components.App.App"),
    ]

    greeter = next(u for u in typescript.units if u.qualified_name == "greeter.Greeter")
    assert greeter.symbols == (
        "greeter.Greeter",
        "greeter.Greeter.constructor",
        "greeter.Greeter.greet",
        "greeter.Greeter.whisper",
    )
    assert (greeter.start_line, greeter.end_line) == (7, 18)


def test_typescript_units_feed_the_static_metrics(typescript: SnapshotCorpus) -> None:
    facts = contexts(typescript, SourceLanguage.TYPESCRIPT)

    greeter = facts["greeter.Greeter"]
    assert greeter.constructor_parameters == ("greeting",)
    assert [s.name for s in greeter.public_members] == ["greet"]
    hello = facts["greeter.hello"]
    assert (hello.parameters, hello.return_annotation) == (("name", "excited"), "string")
    assert [s.name for s in facts["greeter"].public_members] == ["Greeter", "hello"]

    document = (
        "## `hello`\n\n"
        "### Summary\n\nGreets `name`, louder when `excited`; see `console.log` "
        "and `Greeter.greet`, not `fakeHelper`.\n\n"
        "### Parameters\n\n- `name`\n- `excited`\n\n"
        "### Returns\n\nThe greeting.\n"
    )
    result = scores(document, hello)
    assert result["referential_integrity"] == pytest.approx(5 / 6)
    assert result["symbol_coverage"] == 1.0
    assert result["structural_completeness"] == 1.0


def test_go_files_are_discovered_extracted_and_cut_into_units(go: SnapshotCorpus) -> None:
    assert [(i.path, i.module, i.has_syntax_errors) for i in go.inventories] == [
        ("cmd/greet/main.go", "cmd.greet", False),
        ("internal/server/server.go", "internal.server", False),
    ]
    assert units(go) == [
        ("cmd/greet/main.go", "cmd.greet"),
        ("internal/server/server.go", "internal.server"),
        ("internal/server/server.go", "internal.server.Server"),
        ("internal/server/server.go", "internal.server.New"),
    ]

    server = next(u for u in go.units if u.qualified_name == "internal.server.Server")
    assert server.symbols == (
        "internal.server.Server",
        "internal.server.Server.Start",
        "internal.server.Server.stop",
    )


def test_go_units_feed_the_static_metrics(go: SnapshotCorpus) -> None:
    facts = contexts(go, SourceLanguage.GO)

    server = facts["internal.server.Server"]
    assert server.constructor_parameters == ()
    assert [s.name for s in server.public_members] == ["Start"]
    new = facts["internal.server.New"]
    assert (new.parameters, new.return_annotation) == (("addr", "port"), "*Server")

    document = (
        "## `New`\n\n"
        "### Summary\n\nMakes a `Server` on `addr` and `port` with `fmt.Sprintf`; "
        "call `Server.Start` next, not `Server.Listen`.\n\n"
        "### Parameters\n\n- `addr`\n- `port`\n"
    )
    result = scores(document, new)
    assert result["referential_integrity"] == pytest.approx(6 / 7)
    assert result["symbol_coverage"] == 1.0
    assert result["structural_completeness"] == pytest.approx(3 / 4)


def unit_ids(units: tuple[Unit, ...]) -> list[str]:
    return [u.unit_id for u in units]


def test_files_sharing_a_go_package_keep_distinct_unit_ids(
    make_repo: MakeRepo, tmp_path: Path
) -> None:
    corpus = build(
        make_repo,
        tmp_path,
        SourceLanguage.GO,
        {
            "pkg/a.go": "package pkg\n\n// A does a.\nfunc A() {\n\treturn\n}\n",
            "pkg/b.go": "package pkg\n\n// B does b.\nfunc B() {\n\treturn\n}\n",
        },
    )

    assert unit_ids(corpus.units) == [
        "owner/go:pkg/a.go#pkg",
        "owner/go:pkg/a.go#pkg.A",
        "owner/go:pkg/b.go#pkg",
        "owner/go:pkg/b.go#pkg.B",
    ]
