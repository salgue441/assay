import pytest

from assay.metrics import SymbolCoverage, expected_names

from .fixtures import context

METRIC = SymbolCoverage()


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("wrap_text", ("text", "width", "preserve")),
        ("log", ("message",)),
        ("ping", ()),
        ("Wrapper", ("width", "wrap")),
        ("module", ("Wrapper", "wrap_text", "log", "ping")),
    ],
)
def test_expected_names(unit: str, expected: tuple[str, ...]) -> None:
    assert expected_names(context(unit)) == expected


def test_mentions_anywhere_as_whole_words_count() -> None:
    result = METRIC.score("Wraps text to a given width.", context("wrap_text"))

    assert result.value == pytest.approx(2 / 3)
    assert result.explanation == "2/3 expected names mentioned; missing: `preserve`"


def test_partial_words_and_attributes_do_not_count() -> None:
    result = METRIC.score("Uses `texts` and `self.width`.", context("wrap_text"))

    assert result.value == 0.0


def test_a_unit_with_nothing_to_expect_scores_full() -> None:
    assert METRIC.score("Returns the process id.", context("ping")).value == 1.0


def test_private_and_dunder_members_are_not_expected() -> None:
    result = METRIC.score("`Wrapper(width)` with `wrap`.", context("Wrapper"))

    assert result.value == 1.0
