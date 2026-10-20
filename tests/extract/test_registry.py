import pytest

from assay.extract import (
    PythonExtractor,
    SourceLanguage,
    TypeScriptExtractor,
    UnsupportedLanguageError,
    get_extractor,
    supported_languages,
)


@pytest.mark.parametrize("language", [SourceLanguage.PYTHON, "python"])
def test_returns_the_python_extractor(language: SourceLanguage | str) -> None:
    extractor = get_extractor(language)

    assert isinstance(extractor, PythonExtractor)
    assert extractor.language is SourceLanguage.PYTHON


@pytest.mark.parametrize("language", [SourceLanguage.TYPESCRIPT, "typescript"])
def test_returns_the_typescript_extractor(language: SourceLanguage | str) -> None:
    extractor = get_extractor(language)

    assert isinstance(extractor, TypeScriptExtractor)
    assert extractor.language is SourceLanguage.TYPESCRIPT


def test_returns_a_new_instance_per_call() -> None:
    assert get_extractor("python") is not get_extractor("python")


def test_known_but_unimplemented_language_names_what_is_supported() -> None:
    with pytest.raises(
        UnsupportedLanguageError, match=r"no extractor for 'go' yet.*python, typescript"
    ):
        get_extractor(SourceLanguage.GO)


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(UnsupportedLanguageError, match="unknown language 'cobol'"):
        get_extractor("cobol")


def test_supported_languages() -> None:
    assert supported_languages() == {SourceLanguage.PYTHON, SourceLanguage.TYPESCRIPT}
