from pathlib import PurePosixPath
from types import MappingProxyType

import pytest

from assay.corpus import PythonLayout, get_layout, layout
from assay.corpus.layout import GoLayout, TypeScriptLayout
from assay.extract import SourceLanguage, UnsupportedLanguageError

LAYOUT = PythonLayout()
TYPESCRIPT = TypeScriptLayout()
GO = GoLayout()


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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/index.ts", True),
        ("src/App.tsx", True),
        ("src/types.d.ts", False),
        ("src/app.test.ts", False),
        ("src/app.spec.ts", False),
        ("src/App.test.tsx", False),
        ("src/App.spec.tsx", False),
        ("src/__tests__/app.ts", False),
        ("src/__mocks__/api.ts", False),
        ("src/testing.ts", True),
        ("src/app.js", False),
        ("src/app.mts", False),
    ],
)
def test_typescript_is_source_file(path: str, expected: bool) -> None:
    assert TYPESCRIPT.is_source_file(PurePosixPath(path)) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app/user.ts", "app.user"),
        ("lib/user.ts", "lib.user"),
        ("src/components/Button.tsx", "components.Button"),
        ("src/widgets/index.ts", "widgets"),
        ("src/index.ts", "index"),
        ("index.ts", "index"),
        ("src/app/user.service.ts", "app.user.service"),
        ("packages/core-utils/src/my-helper.ts", "packages.core-utils.src.my-helper"),
        ("src/pages/[id].tsx", "pages.[id]"),
        ("src/my dir/a.ts", None),
    ],
)
def test_typescript_module_name(path: str, expected: str | None) -> None:
    assert TYPESCRIPT.module_name(PurePosixPath(path)) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("server.go", True),
        ("internal/http/server.go", True),
        ("internal/http/server_test.go", False),
        ("internal/testdata/fixture.go", False),
        ("_examples/demo.go", False),
        ("internal/http/server.go.txt", False),
        ("README.md", False),
    ],
)
def test_go_is_source_file(path: str, expected: bool) -> None:
    assert GO.is_source_file(PurePosixPath(path)) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("internal/http/server.go", "internal.http"),
        ("internal/http/client.go", "internal.http"),
        ("cmd/my-tool/main.go", "cmd.my-tool"),
        ("errors.go", "errors"),
        ("doc.go", "doc"),
        ("my dir/a.go", None),
    ],
)
def test_go_module_name(path: str, expected: str | None) -> None:
    assert GO.module_name(PurePosixPath(path)) == expected


@pytest.mark.parametrize(
    ("language", "layout_type"),
    [
        (SourceLanguage.PYTHON, PythonLayout),
        ("python", PythonLayout),
        (SourceLanguage.TYPESCRIPT, TypeScriptLayout),
        ("typescript", TypeScriptLayout),
        (SourceLanguage.GO, GoLayout),
        ("go", GoLayout),
    ],
)
def test_get_layout_returns_the_layout_of_each_language(
    language: SourceLanguage | str, layout_type: type[object]
) -> None:
    found = get_layout(language)

    assert isinstance(found, layout_type)
    assert found.language == language


def test_get_layout_rejects_an_unimplemented_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layout, "_LAYOUTS", MappingProxyType({}))

    with pytest.raises(UnsupportedLanguageError, match="no source layout for 'go'"):
        get_layout("go")


def test_get_layout_rejects_an_unknown_language() -> None:
    with pytest.raises(UnsupportedLanguageError, match="unknown language 'cobol'"):
        get_layout("cobol")
