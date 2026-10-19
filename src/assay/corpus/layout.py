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


_LAYOUTS: Mapping[SourceLanguage, LanguageLayout] = MappingProxyType(
    {
        SourceLanguage.PYTHON: PythonLayout(),
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
