import json
from collections.abc import Callable

import pytest
from typer.testing import CliRunner

from assay.cli import app
from assay.judge import JUDGMENTS
from assay.llm import ChatRequest
from assay.orchestrate import build_pipeline
from assay.storage import ParquetStore

from ..fake_llm import FakeLLM
from .project import Project

PARTITION = "owner__greeter@00"
DOCUMENTS, CRITERIA, RUNS = 3, 2, 2  # see the project's config


def answer(request: ChatRequest) -> str:
    """A document for generation requests, a valid grade for judge requests."""

    if request.json_schema is None:
        return "## `greet`\n### Summary\nSays hello."
    return json.dumps({"issues": ["terse"], "rationale": "level 4", "score": 4})


@pytest.fixture
def use_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def install(reply: Callable[[ChatRequest], str] = answer, **fake: object) -> None:
        monkeypatch.setattr(
            "assay.orchestrate.resources.create_client",
            lambda provider, model_id: FakeLLM(model_id=model_id, reply=reply, **fake),  # type: ignore[arg-type]
        )

    install()
    return install


@pytest.fixture
def generated(project: Project, use_llm: Callable[..., None]) -> Project:
    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded
    assert pipeline.materialize_documents(PARTITION, log_level="ERROR").succeeded
    return project


def test_grades_every_document_on_every_criterion_and_run(generated: Project) -> None:
    run = build_pipeline(generated.config).materialize_judgments(PARTITION, log_level="ERROR")

    counts = run.metadata["judgments"]
    expected = DOCUMENTS * CRITERIA * RUNS
    assert run.succeeded
    assert run.warnings == ()
    assert (counts["new"], counts["rows"], counts["failed"]) == (expected, expected, 0)
    stored = ParquetStore(generated.data).read(JUDGMENTS)
    assert {(j.criterion, j.score) for j in stored} == {("accuracy", 4), ("clarity", 4)}


def test_a_second_run_grades_nothing_new(generated: Project) -> None:
    pipeline = build_pipeline(generated.config)
    pipeline.materialize_judgments(PARTITION, log_level="ERROR")

    counts = pipeline.materialize_judgments(PARTITION, log_level="ERROR").metadata["judgments"]

    assert (counts["new"], counts["skipped"]) == (0, DOCUMENTS * CRITERIA * RUNS)


def test_invalid_replies_warn_and_are_retried(
    generated: Project, use_llm: Callable[..., None]
) -> None:
    use_llm(reply=lambda request: "I think it deserves a 4.")
    pipeline = build_pipeline(generated.config)

    first = pipeline.materialize_judgments(PARTITION, log_level="CRITICAL")

    assert first.succeeded
    assert first.warnings == ("judgments.complete",)
    assert first.metadata["judgments"]["failed"] == DOCUMENTS * CRITERIA * RUNS

    use_llm()
    second = pipeline.materialize_judgments(PARTITION, log_level="ERROR")

    assert second.warnings == ()
    assert second.metadata["judgments"]["new"] == DOCUMENTS * CRITERIA * RUNS


def test_documents_too_large_for_the_judge_are_left_out(
    generated: Project, use_llm: Callable[..., None]
) -> None:
    use_llm(context_tokens=1000)

    run = build_pipeline(generated.config).materialize_judgments(PARTITION, log_level="ERROR")

    assert run.metadata["judgments"]["too_large"] == DOCUMENTS


def test_judging_needs_documents_first(project: Project) -> None:
    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded

    run = pipeline.materialize_judgments(PARTITION, log_level="CRITICAL")

    assert run.error == f"no documents stored for {PARTITION}; run generate first"


def test_the_judge_command(generated: Project) -> None:
    result = CliRunner().invoke(app, ["judge", "--config", str(generated.config), "--epoch", "0"])

    assert result.exit_code == 0, result.output
    assert "12 new, 0 existing, 0 failed, 12 stored; 0 documents too large" in result.output
