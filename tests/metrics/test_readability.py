import pytest

from assay.metrics import Readability

from .fixtures import context

METRIC = Readability()


def score(text: str) -> float:
    return METRIC.score(text, context("wrap_text")).value


def test_plain_short_sentences_score_full() -> None:
    assert score("This wraps text. It keeps lines short. You can set the width.") == 1.0


def test_dense_long_sentences_score_lower() -> None:
    dense = (
        "Notwithstanding considerable configurational heterogeneity, the implementation "
        "systematically reconciles typographical irregularities, accommodating "
        "internationalization considerations, indentation conventions, and paragraph "
        "segmentation characteristics simultaneously."
    )

    assert score(dense) < 0.3


def test_code_is_not_read_as_prose() -> None:
    plain = "This wraps text. It keeps lines short. You can set the width."
    code = "```python\nreconfigure_internationalization_parameters(x)\n```"
    with_code = f"{plain}\n{code}\n`a_long_identifier_name`"

    assert score(with_code) == score(plain)


def test_too_little_prose_scores_zero() -> None:
    result = METRIC.score(
        "## `wrap_text`\n```python\nwrap_text('x')\n```\nWraps text.", context("wrap_text")
    )

    assert (result.value, result.explanation) == (0.0, "too little prose to read (2 words)")


@pytest.mark.parametrize("text", ["", "### Summary"])
def test_empty_documents_score_zero(text: str) -> None:
    assert score(text) == 0.0
