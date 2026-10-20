from types import MappingProxyType

import pytest

from assay.extract import (
    GoExtractor,
    PythonExtractor,
    SourceLanguage,
    TypeScriptExtractor,
    UnsupportedLanguageError,
    get_extractor,
    registry,
    supported_languages,
)


@pytest.mark.parametrize(
    ("language", "extractor_type"),
    [
        (SourceLanguage.PYTHON, PythonExtractor),
        ("python", PythonExtractor),
        (SourceLanguage.TYPESCRIPT, TypeScriptExtractor),
        ("typescript", TypeScriptExtractor),
        (SourceLanguage.GO, GoExtractor),
        ("go", GoExtractor),
    ],
)
def test_returns_the_extractor_of_each_language(
    language: SourceLanguage | str, extractor_type: type[object]
) -> None:
    extractor = get_extractor(language)

    assert isinstance(extractor, extractor_type)
    assert extractor.language == language


def test_returns_a_new_instance_per_call() -> None:
    assert get_extractor("python") is not get_extractor("python")


def test_known_but_unimplemented_language_names_what_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories = MappingProxyType({SourceLanguage.PYTHON: PythonExtractor})
    monkeypatch.setattr(registry, "_FACTORIES", factories)

    with pytest.raises(UnsupportedLanguageError, match=r"no extractor for 'go' yet.*python"):
        get_extractor(SourceLanguage.GO)


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(UnsupportedLanguageError, match="unknown language 'cobol'"):
        get_extractor("cobol")


def test_every_language_is_supported() -> None:
    assert supported_languages() == set(SourceLanguage)
