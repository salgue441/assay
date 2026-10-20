from pathlib import Path

import pytest
from pydantic import ValidationError

from assay.scoring import SCORES, Component, RoutingDecision, Score
from assay.storage import ParquetStore, Partition


def score(**changes: object) -> Score:
    values: dict[str, object] = {
        "document_id": "d" * 64,
        "unit_id": "pkg.mod.func",
        "repo": "owner/repo",
        "commit": "c" * 40,
        "subject": "writer",
        "scorer_version": "0123456789abcdef",
        "judge": "grader",
        "composite": 72.5,
        "components": (
            Component(name="accuracy", value=0.75, weight=0.5),
            Component(name="referential_integrity", value=0.7, weight=0.5),
        ),
        "missing": ("clarity",),
        "critical": False,
        "critical_reasons": (),
        "decision": RoutingDecision.REVISE,
    } | changes
    return Score.model_validate(values)


def test_round_trips_through_the_store(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    rows = [
        score(),
        score(
            document_id="e" * 64,
            critical=True,
            critical_reasons=("referential_integrity",),
            decision=RoutingDecision.REJECT,
        ),
    ]
    store.write(SCORES, rows, Partition(repo="owner/repo", epoch=0))

    assert store.read(SCORES) == rows


def test_the_same_document_may_be_stored_under_two_scorer_versions(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    rows = [score(), score(scorer_version="fedcba9876543210")]
    store.write(SCORES, rows, Partition(repo="owner/repo", epoch=0))

    assert len(store.read(SCORES)) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"critical": True},
        {"critical_reasons": ("referential_integrity",)},
        {"decision": RoutingDecision.REJECT},
        {
            "critical": True,
            "critical_reasons": ("referential_integrity",),
            "decision": RoutingDecision.PUBLISH,
        },
        {"composite": 100.5},
        {"components": ()},
        {"scorer_version": "short"},
    ],
)
def test_rejects_inconsistent_scores(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        score(**changes)
