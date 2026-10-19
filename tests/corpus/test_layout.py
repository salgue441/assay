from pathlib import PurePosixPath

import pytest

from assay.corpus import PythonLayout, get_layout
from assay.extract import SourceLanguage, UnsupportedLanguageError

LAYOUT = PythonLayout()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/mod.py", True),
        ("pkg/test_mod.py", False),
        ("pkg/mod_test.py", False),
        ("conftest.py", False),
        ("pkg/mod.pyi", False),
        ("README.md", False),
    ],
)
def test_is_source_file(path: str, expected: bool) -> None:
    assert LAYOUT.is_source_file(PurePosixPath(path)) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/mod.py", "pkg.mod"),
        ("pkg/__init__.py", "pkg"),
        ("src/pkg/sub/mod.py", "pkg.sub.mod"),
        ("src/pkg/__init__.py", "pkg"),
        ("setup.py", "setup"),
        ("__init__.py", None),
        ("src/__init__.py", None),
        ("my-scripts/tool.py", None),
        ("pkg/2fast.py", None),
    ],
)
def test_module_name(path: str, expected: str | None) -> None:
    assert LAYOUT.module_name(PurePosixPath(path)) == expected


@pytest.mark.parametrize("language", [SourceLanguage.PYTHON, "python"])
def test_get_layout_returns_the_python_layout(language: SourceLanguage | str) -> None:
    assert get_layout(language).language is SourceLanguage.PYTHON


def test_get_layout_rejects_an_unimplemented_language() -> None:
    with pytest.raises(UnsupportedLanguageError, match="no source layout for 'go'"):
        get_layout("go")


def test_get_layout_rejects_an_unknown_language() -> None:
    with pytest.raises(UnsupportedLanguageError, match="unknown language 'cobol'"):
        get_layout("cobol")
