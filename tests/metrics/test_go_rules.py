import pytest

from assay.metrics.rules import GoRules

RULES = GoRules()


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("func New(addr string, port int) *Server", ("addr", "port")),
        ("func F(a, b int, rest ...string) (int, error)", ("a", "b", "rest")),
        ("func (s *Server) Start(ctx context.Context) error", ("ctx",)),
        ("func (Server) Addr() string", ()),
        ("func F[T any, K comparable](m map[K]T) []T", ("m",)),
        ("func F(int, string)", ()),
        ("func F(fn func(x int) int)", ("fn",)),
        ("Get(ctx context.Context, key string) (value []byte, err error)", ("ctx", "key")),
        ("type Server struct", ()),
        ("func Broken(", ()),
        ("", ()),
    ],
)
def test_parameters(signature: str, expected: tuple[str, ...]) -> None:
    assert RULES.parameters(signature) == expected


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("func New() *Server", "*Server"),
        ("func F() (int, error)", "(int, error)"),
        ("func (s *Server) Start() (err error)", "(err error)"),
        ("Close() error", "error"),
        ("func F[T any](x T) []T", "[]T"),
        ("func main()", None),
        ("type Server struct", None),
        ("func Broken(", None),
    ],
)
def test_return_annotation(signature: str, expected: str | None) -> None:
    assert RULES.return_annotation(signature) == expected


@pytest.mark.parametrize(
    "name",
    [
        "func",
        "nil",
        "len",
        "error",
        "string",
        "fmt",
        "Println",
        "strings",
        "context",
        "Context",
        "errors",
        "New",
        "http",
        "Error",
    ],
)
def test_ambient_names(name: str) -> None:
    assert name in RULES.ambient_names


def test_ambient_names_exclude_arbitrary_words() -> None:
    assert "NewServer" not in RULES.ambient_names


def test_go_has_no_constructor_methods() -> None:
    assert RULES.constructor_name is None
