from collections.abc import Callable

import pytest
from typer.testing import CliRunner

from assay.cli import app
from assay.generate import DOCUMENTS
from assay.llm import ChatRequest
from assay.orchestrate import build_pipeline
from assay.storage import ParquetStore

from ..fake_llm import FakeLLM
from .project import Project

PARTITION = "owner__greeter@00"


@pytest.fixture
def use_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Replace the model clients the pipeline creates with scripted fakes."""

    def install(**fake: object) -> None:
        monkeypatch.setattr(
            "assay.orchestrate.resources.create_client",
            lambda provider, model_id: FakeLLM(model_id=model_id, **fake),  # type: ignore[arg-type]
        )

    install()
    return install


@pytest.fixture
def built(project: Project) -> Project:
    assert build_pipeline(project.config).materialize_corpus(PARTITION, log_level="ERROR").succeeded
    return project


def test_documents_every_sampled_unit(built: Project, use_llm: Callable[..., None]) -> None:
    run = build_pipeline(built.config).materialize_documents(PARTITION, log_level="ERROR")

    assert run.succeeded
    assert run.warnings == ()
    counts = run.metadata["documents"]
    assert (counts["new"], counts["skipped"], counts["failed"], counts["rows"]) == (3, 0, 0, 3)
    stored = ParquetStore(built.data).read(DOCUMENTS)
    assert {d.subject for d in stored} == {"writer"}
    assert {d.model_id for d in stored} == {"fake:1b"}


def test_a_second_run_generates_nothing_new(built: Project, use_llm: Callable[..., None]) -> None:
    pipeline = build_pipeline(built.config)
    pipeline.materialize_documents(PARTITION, log_level="ERROR")

    counts = pipeline.materialize_documents(PARTITION, log_level="ERROR").metadata["documents"]

    assert (counts["new"], counts["skipped"], counts["rows"]) == (0, 3, 3)


def test_failures_warn_and_are_retried_on_the_next_run(
    built: Project, use_llm: Callable[..., None]
) -> None:
    def fails_on_greet(request: ChatRequest) -> bool:
        return "`greeter.core.greet`" in request.user

    use_llm(fail_when=fails_on_greet)
    pipeline = build_pipeline(built.config)

    first = pipeline.materialize_documents(PARTITION, log_level="CRITICAL")

    assert first.succeeded
    assert first.warnings == ("documents.complete",)
    assert (first.metadata["documents"]["failed"], first.metadata["documents"]["rows"]) == (1, 2)

    use_llm()
    second = pipeline.materialize_documents(PARTITION, log_level="ERROR")

    assert second.warnings == ()
    assert (second.metadata["documents"]["new"], second.metadata["documents"]["rows"]) == (1, 3)


def test_units_too_large_for_the_context_are_left_out(
    built: Project, use_llm: Callable[..., None]
) -> None:
    use_llm(context_tokens=400)

    run = build_pipeline(built.config).materialize_documents(PARTITION, log_level="ERROR")

    counts = run.metadata["documents"]
    assert run.succeeded
    assert run.warnings == ()
    assert counts["too_large"] > 0
    assert counts["new"] == 3 - counts["too_large"]


def test_generation_needs_the_corpus_first(project: Project, use_llm: Callable[..., None]) -> None:
    run = build_pipeline(project.config).materialize_documents(PARTITION, log_level="CRITICAL")

    assert not run.succeeded


def test_the_generate_command(built: Project, use_llm: Callable[..., None]) -> None:
    result = CliRunner().invoke(app, ["generate", "--config", str(built.config), "--epoch", "0"])

    assert result.exit_code == 0, result.output
    assert "3 new, 0 existing, 0 failed, 3 stored; 0 units too large to prompt" in result.output
    assert "1/1 partitions done" in result.output
