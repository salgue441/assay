"""
Language-specific rules for finding source files and naming their modules.

Adding a language means implementing `LanguageLayout` and adding one entry to
`_LAYOUTS`; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import ClassVar, Protocol

from assay.extract import SourceLanguage, UnsupportedLanguageError


class LanguageLayout(Protocol):
    """
    How one language lays out source on disk.

    Paths are relative to the snapshot root, in POSIX form.
    """

    @property
    def language(self) -> SourceLanguage: ...

    def is_source_file(self, path: PurePosixPath) -> bool:
        """Whether the file is non-test source in this language."""
        ...

    def module_name(self, path: PurePosixPath) -> str | None:
        """The module name for a source file, or None if it has no valid one."""
        ...


class PythonLayout:
    """
    Python files outside tests, named by their import path.

    A leading `src/` directory is stripped (the src layout), and a package's
    `__init__.py` is named after the package.
    """

    _SOURCE_ROOTS: ClassVar[frozenset[str]] = frozenset({"src"})

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.PYTHON

    def is_source_file(self, path: PurePosixPath) -> bool:
        if path.suffix != ".py":
            return False

        name = path.name
        is_test = name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
        return not is_test

    def module_name(self, path: PurePosixPath) -> str | None:
        parts = list(path.with_suffix("").parts)
        if parts and parts[0] in self._SOURCE_ROOTS:
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        if not parts or not all(part.isidentifier() for part in parts):
            return None
        return ".".join(parts)


class TypeScriptLayout:
    """
    TypeScript files outside tests and declaration files, named by their
    path as a dotted module name.

    The naming mirrors `PythonLayout`: a leading `src/` directory is
    stripped, and an `index` file is named after its directory, the way
    `import "./widgets"` resolves to `widgets/index.ts`. Dots inside a file
    name are kept (`app/user.service.ts` is `app.user.service`), so names
    stay readable at the cost of rare clashes such as `a/b.c.ts` and
    `a/b/c.ts`; unit ids include the path, so a clash never merges units.
    """

    _SOURCE_ROOTS: ClassVar[frozenset[str]] = frozenset({"src"})
    _SUFFIXES: ClassVar[frozenset[str]] = frozenset({".ts", ".tsx"})
    _TEST_MARKERS: ClassVar[frozenset[str]] = frozenset({".test", ".spec"})
    _TEST_DIRS: ClassVar[frozenset[str]] = frozenset({"__tests__", "__mocks__"})
    _INDEX = "index"

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.TYPESCRIPT

    def is_source_file(self, path: PurePosixPath) -> bool:
        if path.suffix not in self._SUFFIXES or path.name.endswith(".d.ts"):
            return False

        is_test = PurePosixPath(path.stem).suffix in self._TEST_MARKERS
        return not is_test and not self._TEST_DIRS.intersection(path.parts)

    def module_name(self, path: PurePosixPath) -> str | None:
        parts = list(path.with_suffix("").parts)
        if parts and parts[0] in self._SOURCE_ROOTS:
            parts = parts[1:]
        if len(parts) > 1 and parts[-1] == self._INDEX:
            parts = parts[:-1]

        if not parts or not all(_is_name_segment(part) for part in parts):
            return None
        return ".".join(parts)


class GoLayout:
    """
    Go files outside tests, named by their package directory.

    A Go package is a directory, so every file in it shares the dotted
    directory path as its module name (`internal/http/server.go` is
    `internal.http`), the import path with dots for slashes. Unit ids
    include the file path, so units in sibling files stay distinct. Files
    at the repository root have no directory to name them and are named by
    their file stem instead. Like the go tool, directories named
    `testdata` or starting with `_` are ignored.
    """

    _SUFFIX = ".go"
    _TEST_SUFFIX = "_test.go"
    _IGNORED_DIR = "testdata"

    @property
    def language(self) -> SourceLanguage:
        return SourceLanguage.GO

    def is_source_file(self, path: PurePosixPath) -> bool:
        if path.suffix != self._SUFFIX or path.name.endswith(self._TEST_SUFFIX):
            return False

        directories = path.parts[:-1]
        return not any(d == self._IGNORED_DIR or d.startswith("_") for d in directories)

    def module_name(self, path: PurePosixPath) -> str | None:
        parts = list(path.parent.parts) or [path.stem]
        if not all(_is_name_segment(part) for part in parts):
            return None
        return ".".join(parts)


def _is_name_segment(part: str) -> bool:
    """
    Whether a path segment can be part of a dotted module name.
    """

    return bool(part) and not part.startswith(".") and not any(c.isspace() for c in part)


_LAYOUTS: Mapping[SourceLanguage, LanguageLayout] = MappingProxyType(
    {
        SourceLanguage.PYTHON: PythonLayout(),
        SourceLanguage.TYPESCRIPT: TypeScriptLayout(),
        SourceLanguage.GO: GoLayout(),
    }
)


def get_layout(language: SourceLanguage | str) -> LanguageLayout:
    """
    Return the layout for `language`; layouts are stateless and shared.

    Raises `UnsupportedLanguageError` for an unknown or unimplemented language.
    """

    try:
        resolved = SourceLanguage(language)
    except ValueError:
        msg = f"unknown language {language!r}"
        raise UnsupportedLanguageError(msg) from None

    layout = _LAYOUTS.get(resolved)
    if layout is None:
        msg = f"no source layout for {resolved.value!r} yet"
        raise UnsupportedLanguageError(msg)
    return layout
