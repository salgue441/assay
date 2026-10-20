import pytest

from assay.metrics import StructuralCompleteness, required_sections

from .fixtures import context

METRIC = StructuralCompleteness()


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("wrap_text", ("summary", "parameters", "returns")),
        ("log", ("summary", "parameters")),
        ("ping", ("summary",)),
        ("Wrapper", ("summary", "constructor", "methods")),
        ("module", ("summary", "contents")),
    ],
)
def test_required_sections_follow_the_unit(unit: str, expected: tuple[str, ...]) -> None:
    assert required_sections(context(unit)) == expected


def test_a_complete_document_scores_full() -> None:
    text = "## `textkit.core.wrap_text`\n### Summary\nx\n### Parameters\nx\n### Returns\nx\n"

    result = METRIC.score(text, context("wrap_text"))

    assert (result.value, result.explanation) == (1.0, "4/4 structural requirements met")


def test_synonyms_and_extra_punctuation_are_accepted() -> None:
    text = "## `wrap_text`\n### Overview:\nx\n#### Arguments\nx\n### Return value\nx\n"

    assert METRIC.score(text, context("wrap_text")).value == 1.0


def test_a_level_one_title_and_missing_sections_are_reported() -> None:
    text = "# `wrap_text`\nWraps text.\n### Parameters\nx\n"

    result = METRIC.score(text, context("wrap_text"))

    assert result.value == 0.25
    assert result.explanation == (
        "1/4 structural requirements met; missing: level-2 title naming the unit, summary, returns"
    )


def test_a_title_for_another_unit_does_not_count() -> None:
    assert METRIC.score("## `log`\n### Summary\nx\n", context("ping")).value == 0.5


def test_prose_without_headings_scores_zero() -> None:
    assert METRIC.score("This code defines a class.", context("Wrapper")).value == 0.0
