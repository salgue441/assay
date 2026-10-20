import json
from collections.abc import Callable

import pytest
import yaml
from typer.testing import CliRunner

from assay.cli import app
from assay.judge import JUDGMENTS
from assay.llm import ChatRequest
from assay.metrics import METRICS_TABLE
from assay.orchestrate import build_pipeline
from assay.scoring import SCORES, RoutingDecision
from assay.storage import ParquetStore, Partition

from ..fake_llm import FakeLLM
from .project import Project

PARTITION = "owner__greeter@00"
STORED = Partition("owner/greeter", 0)
DOCUMENTS = 3  # see the project's config
CLEAN = "## `greet`\n### Summary\nSays hello."
"""A document whose only cited name resolves, so referential integrity is 1."""


def grade(request: ChatRequest, *, document: str = CLEAN, score: int = 4) -> str:
    """A document for generation requests, a grade for judge requests."""

    if request.json_schema is None:
        return document
    return json.dumps({"issues": [], "rationale": "fine", "score": score})


@pytest.fixture
def use_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def install(reply: Callable[[ChatRequest], str] = grade) -> None:
        monkeypatch.setattr(
            "assay.orchestrate.resources.create_client",
            lambda provider, model_id: FakeLLM(model_id=model_id, reply=reply),
        )

    install()
    return install


def set_weights(project: Project, weights: dict[str, float]) -> None:
    config = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    config["scoring"]["weights"] = weights
    project.config.write_text(yaml.safe_dump(config), encoding="utf-8")


def prepare(project: Project, *, judge: bool = True) -> None:
    """Materialize everything the scores depend on."""

    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded
    assert pipeline.materialize_documents(PARTITION, log_level="ERROR").succeeded
    assert pipeline.materialize_metrics(PARTITION, log_level="ERROR").succeeded
    if judge:
        assert pipeline.materialize_judgments(PARTITION, log_level="ERROR").succeeded


@pytest.fixture
def judged(project: Project, use_llm: Callable[..., None]) -> Project:
    prepare(project)
    return project


def score(project: Project) -> dict[str, int]:
    run = build_pipeline(project.config).materialize_scores(PARTITION, log_level="ERROR")
    assert run.succeeded, run.error
    return run.metadata["scores"]


def test_scores_and_routes_every_document(judged: Project) -> None:
    counts = score(judged)

    assert counts == {
        "rows": DOCUMENTS,
        "scored": DOCUMENTS,
        "unscored": 0,
        "publish": DOCUMENTS,
        "revise": 0,
        "escalate": 0,
        "reject": 0,
    }
    stored = ParquetStore(judged.data).read(SCORES)
    assert {(s.composite, s.decision, s.judge) for s in stored} == {
        (100.0, RoutingDecision.PUBLISH, "grader")
    }


def test_judged_components_come_from_the_configured_judge(judged: Project) -> None:
    set_weights(judged, {"accuracy": 0.5, "clarity": 0.5})

    counts = score(judged)

    assert (counts["revise"], counts["publish"]) == (DOCUMENTS, 0)
    stored = ParquetStore(judged.data).read(SCORES)
    assert {s.composite for s in stored} == {75.0}  # a 4 on the 1-5 scale
    assert {tuple(c.name for c in s.components) for s in stored} == {("accuracy", "clarity")}


def test_critical_failures_reject_whatever_the_composite(
    project: Project, use_llm: Callable[..., None]
) -> None:
    use_llm(lambda request: grade(request, document=CLEAN + " See `made_up`."))
    prepare(project)
    set_weights(project, {"referential_integrity": 0.5, "accuracy": 0.5})

    counts = score(project)

    assert (counts["reject"], counts["scored"]) == (DOCUMENTS, DOCUMENTS)
    stored = ParquetStore(project.data).read(SCORES)
    assert {s.critical_reasons for s in stored} == {("referential_integrity",)}
    assert {s.composite for s in stored} == {62.5}  # integrity 1/2, accuracy 3/4


def test_low_grades_escalate(project: Project, use_llm: Callable[..., None]) -> None:
    use_llm(lambda request: grade(request, score=1))
    prepare(project)
    set_weights(project, {"accuracy": 1.0})

    assert score(project)["escalate"] == DOCUMENTS


def test_variance_study_and_old_runs_never_mix_in(judged: Project) -> None:
    set_weights(judged, {"referential_integrity": 0.5, "accuracy": 0.5})
    store = ParquetStore(judged.data)
    judgments = store.read(JUDGMENTS)
    other_settings = [
        j.model_copy(update={"judgment_id": f"{i:064x}", "score": 1, **change})
        for i, (j, change) in enumerate(
            (j, change)
            for j in judgments
            for change in (
                {"temperature": 0.7},
                {"max_output_tokens": 768},
                {"rubric_hash": "0" * 64},
                {"reply_schema_hash": "0" * 64},
            )
        )
    ]
    store.write(JUDGMENTS, [*judgments, *other_settings], STORED)
    metrics = store.read(METRICS_TABLE)
    old_metrics = [m.model_copy(update={"metric_version": "old", "value": 0.0}) for m in metrics]
    store.write(METRICS_TABLE, [*metrics, *old_metrics], STORED)

    counts = score(judged)

    assert counts["publish"] == DOCUMENTS
    assert {s.composite for s in store.read(SCORES)} == {87.5}


def test_documents_without_weighted_components_are_unscored(
    project: Project, use_llm: Callable[..., None]
) -> None:
    prepare(project, judge=False)
    set_weights(project, {"accuracy": 1.0})

    counts = score(project)

    assert (counts["scored"], counts["unscored"], counts["rows"]) == (0, DOCUMENTS, 0)


def test_rescoring_replaces_the_current_version_and_keeps_others(judged: Project) -> None:
    score(judged)
    set_weights(judged, {"accuracy": 1.0})
    score(judged)

    counts = score(judged)

    stored = ParquetStore(judged.data).read(SCORES)
    assert counts["rows"] == len(stored) == 2 * DOCUMENTS
    assert len({s.scorer_version for s in stored}) == 2
    assert {s.decision for s in stored} == {RoutingDecision.PUBLISH, RoutingDecision.REVISE}


def test_scoring_needs_documents_first(project: Project) -> None:
    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded

    run = pipeline.materialize_scores(PARTITION, log_level="CRITICAL")

    assert not run.succeeded
    assert run.error == f"no documents stored for {PARTITION}; run generate first"


def test_the_score_command(judged: Project) -> None:
    result = CliRunner().invoke(app, ["score", "--config", str(judged.config), "--epoch", "0"])

    assert result.exit_code == 0, result.output
    assert "3 scored (publish 3, revise 0, escalate 0, reject 0), 0 unscored" in result.output


def test_the_score_command_reports_missing_documents(project: Project) -> None:
    build_pipeline(project.config).materialize_corpus(PARTITION, log_level="ERROR")

    result = CliRunner().invoke(app, ["score", "--config", str(project.config), "--epoch", "0"])

    assert result.exit_code == 1
    assert f"FAILED {PARTITION:<40} no documents stored" in result.output
