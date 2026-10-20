import pytest

from assay.metrics import ReferentialIntegrity, code_references

from .fixtures import context

METRIC = ReferentialIntegrity(critical_below=0.9)


def score(text: str, unit: str = "wrap_text") -> tuple[float, bool, str]:
    result = METRIC.score(text, context(unit))
    return result.value, result.critical, result.explanation


def test_names_from_source_snapshot_builtins_and_stdlib_resolve() -> None:
    text = (
        "`wrap_text` takes `text`, `width` and `preserve`, returns a `str`, "
        "and is like `textkit.core.Wrapper.wrap`, `self.width` or `os.path.join`."
    )

    assert score(text) == (1.0, False, "8/8 cited names resolve")


def test_hallucinated_names_do_not_resolve_and_can_be_critical() -> None:
    text = "`wrap_text` calls `textkit.core.fake_helper` and `shrink_text`, then `os.fake`."

    value, critical, explanation = score(text)

    assert value == 0.25
    assert critical
    assert explanation == (
        "1/4 cited names resolve; unresolved: `os.fake`, `shrink_text`, `textkit.core.fake_helper`"
    )


def test_each_name_counts_once() -> None:
    assert score("`fake` `fake` `fake` `text`")[0] == 0.5


def test_a_document_citing_nothing_scores_full_without_being_critical() -> None:
    assert score("Wraps text nicely.") == (1.0, False, "0/0 cited names resolve")


def test_threshold_is_optional() -> None:
    assert not ReferentialIntegrity().score("`made_up`", context("wrap_text")).critical


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        (("wrap_text(text, width=78)",), {"wrap_text", "text", "width"}),
        (('"replace"', "'strict'"), set()),
        (("errors='replace'",), {"errors"}),
        (("--verbose", "-v"), set()),
        (("config/default.yaml", "C:\\data"), set()),
        (("int | None",), {"int", "None"}),
        (("a.b.c()",), {"a.b.c"}),
        (("3.14",), set()),
    ],
)
def test_code_references(spans: tuple[str, ...], expected: set[str]) -> None:
    assert code_references(spans) == expected
