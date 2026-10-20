import pytest

from assay.scoring.components import critical_failures, judged_components, static_components

from .fixtures import doc_id, judgment, metric_row

A, B = doc_id("a"), doc_id("b")


def test_static_components_are_named_after_their_metric() -> None:
    rows = [
        metric_row("a", "referential_integrity", 0.9),
        metric_row("a", "readability", 0.4),
        metric_row("b", "readability", 0.6),
    ]

    assert static_components(rows) == {
        A: {"referential_integrity": 0.9, "readability": 0.4},
        B: {"readability": 0.6},
    }


def test_a_metric_twice_for_one_document_is_rejected() -> None:
    rows = [metric_row("a", "readability", 0.4), metric_row("a", "readability", 0.5, version="2")]

    with pytest.raises(ValueError, match="'readability' more than once"):
        static_components(rows)


def test_critical_failures_are_the_sorted_failing_metrics() -> None:
    rows = [
        metric_row("a", "symbol_coverage", 0.1, critical=True),
        metric_row("a", "readability", 0.4),
        metric_row("a", "referential_integrity", 0.2, critical=True),
        metric_row("b", "readability", 0.6),
    ]

    assert critical_failures(rows) == {A: ("referential_integrity", "symbol_coverage")}


def test_judged_components_take_the_median_run_on_a_unit_scale() -> None:
    judgments = [
        judgment("a", "accuracy", 2, run=0),
        judgment("a", "accuracy", 5, run=1),
        judgment("a", "accuracy", 4, run=2),
        judgment("a", "clarity", 1, run=0),
        judgment("a", "clarity", 2, run=1),
        judgment("b", "accuracy", 5),
    ]

    assert judged_components(judgments, judge="grader") == {
        A: {"accuracy": 0.75, "clarity": 0.125},
        B: {"accuracy": 1.0},
    }


def test_other_judges_are_ignored() -> None:
    judgments = [
        judgment("a", "accuracy", 3),
        judgment("a", "accuracy", 5, run=1, judge="other"),
        judgment("b", "accuracy", 5, judge="other"),
    ]

    assert judged_components(judgments, judge="grader") == {A: {"accuracy": 0.5}}


def test_a_run_twice_for_one_criterion_is_rejected() -> None:
    judgments = [judgment("a", "accuracy", 3), judgment("a", "accuracy", 4)]

    with pytest.raises(ValueError, match="run 0 of 'accuracy' by 'grader' more than once"):
        judged_components(judgments, judge="grader")
