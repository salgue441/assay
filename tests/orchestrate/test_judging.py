import json
from collections.abc import Callable

import pytest
import yaml
from typer.testing import CliRunner

from assay.cli import app
from assay.judge import JUDGMENTS
from assay.llm import ChatRequest
from assay.orchestrate import build_pipeline
from assay.orchestrate.assets import variance_subset
from assay.storage import ParquetStore

from ..fake_llm import FakeLLM
from ..scoring.fixtures import document
from .project import Project

PARTITION = "owner__greeter@00"
DOCUMENTS, CRITERIA, RUNS = 3, 2, 2  # see the project's config


GRADE_ONE = json.dumps({"issues": ["wrong"], "rationale": "level 1", "score": 1})


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


def study_variance(project: Project, *, temperatures: list[float], subset: int) -> None:
    config = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    config["judge"]["variance_study"] = {"temperatures": temperatures, "subset_size": subset}
    project.config.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_the_variance_study_grades_a_subset_again_at_its_temperatures(
    generated: Project, use_llm: Callable[..., None]
) -> None:
    use_llm(lambda r: answer(r) if r.temperature == 0.0 or r.json_schema is None else GRADE_ONE)
    study_variance(generated, temperatures=[0.0, 0.7], subset=2)
    pipeline = build_pipeline(generated.config)

    counts = pipeline.materialize_judgments(PARTITION, log_level="ERROR").metadata["judgments"]

    main, repeated = DOCUMENTS * CRITERIA * RUNS, 2 * CRITERIA * RUNS
    assert (counts["new"], counts["variance_new"], counts["rows"]) == (
        main,
        repeated,
        main + repeated,
    )
    stored = ParquetStore(generated.data).read(JUDGMENTS)
    warm = [j for j in stored if j.temperature == 0.7]
    assert len({j.document_id for j in warm}) == 2
    assert {j.score for j in warm} == {1}

    again = pipeline.materialize_judgments(PARTITION, log_level="ERROR").metadata["judgments"]

    assert (again["new"], again["variance_new"]) == (0, 0)


def test_the_variance_subset_is_stable_and_sized() -> None:
    documents = [document(name) for name in "abcdefgh"]

    subset = variance_subset(documents, size=3, seed=7)

    assert len(subset) == 3
    assert subset == variance_subset(list(reversed(documents)), size=3, seed=7)
    assert subset != variance_subset(documents, size=3, seed=8)
    assert variance_subset(documents, size=20, seed=7) == {d.document_id for d in documents}
