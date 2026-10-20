import pytest

from assay.extract import UnsupportedLanguageError
from assay.metrics import PythonRules, get_rules

RULES = PythonRules()


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("def f(a, b: int = 1, *args, c, d=2, **kw) -> None", ("a", "b", "args", "c", "d", "kw")),
        ("def f(a, /, b)", ("a", "b")),
        ("def method(self, x)", ("x",)),
        ("def build(cls)", ()),
        ("async def fetch(url: str) -> bytes", ("url",)),
        ("def f()", ()),
        ("class Box(Base)", ()),
        ("def broken(", ()),
    ],
)
def test_parameters(signature: str, expected: tuple[str, ...]) -> None:
    assert RULES.parameters(signature) == expected


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("def f() -> list[str]", "list[str]"),
        ("def f() -> None", "None"),
        ("def f()", None),
        ("class C", None),
    ],
)
def test_return_annotation(signature: str, expected: str | None) -> None:
    assert RULES.return_annotation(signature) == expected


@pytest.mark.parametrize(
    "name", ["None", "len", "str", "os", "typing", "join", "items", "self", "match"]
)
def test_ambient_names(name: str) -> None:
    assert name in RULES.ambient_names


def test_ambient_names_exclude_arbitrary_words() -> None:
    assert "wrap_text" not in RULES.ambient_names


def test_get_rules() -> None:
    assert isinstance(get_rules("python"), PythonRules)
    with pytest.raises(UnsupportedLanguageError, match="no static-check rules for 'go'"):
        get_rules("go")
    with pytest.raises(UnsupportedLanguageError, match="unknown language"):
        get_rules("cobol")
