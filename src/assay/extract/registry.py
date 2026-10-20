"""
Look up the extractor for a source language.

Adding a language means implementing `SymbolExtractor` and adding one entry to
`_FACTORIES`; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from assay.extract.base import SymbolExtractor
from assay.extract.go import GoExtractor
from assay.extract.models import SourceLanguage
from assay.extract.python import PythonExtractor
from assay.extract.typescript import TypeScriptExtractor


class UnsupportedLanguageError(ValueError):
    """Raised when no extractor exists for a language."""


_FACTORIES: Mapping[SourceLanguage, Callable[[], SymbolExtractor]] = MappingProxyType(
    {
        SourceLanguage.PYTHON: PythonExtractor,
        SourceLanguage.TYPESCRIPT: TypeScriptExtractor,
        SourceLanguage.GO: GoExtractor,
    }
)


def supported_languages() -> frozenset[SourceLanguage]:
    return frozenset(_FACTORIES)


def get_extractor(language: SourceLanguage | str) -> SymbolExtractor:
    """
    Return a new extractor for `language`.

    Raises `UnsupportedLanguageError` for an unknown language or one whose
    extractor is not implemented yet.
    """

    try:
        resolved = SourceLanguage(language)
    except ValueError:
        msg = f"unknown language {language!r}"
        raise UnsupportedLanguageError(msg) from None

    factory = _FACTORIES.get(resolved)
    if factory is None:
        supported = ", ".join(sorted(supported_languages()))
        msg = f"no extractor for {resolved.value!r} yet (supported: {supported})"
        raise UnsupportedLanguageError(msg)

    return factory()
