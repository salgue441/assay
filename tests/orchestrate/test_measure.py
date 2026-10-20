import pytest
from typer.testing import CliRunner

from assay.cli import app
from assay.llm import ChatRequest
from assay.metrics import METRICS_TABLE
from assay.orchestrate import build_pipeline
from assay.storage import ParquetStore

from ..fake_llm import FakeLLM
from .project import Project

PARTITION = "owner__greeter@00"
STATIC_METRICS = 4


def reply(request: ChatRequest) -> str:
    """A plausible document citing two real names and one invented one."""

    return "## `greet`\n### Summary\nSays hello to `name` with `made_up_helper`. It is short."


@pytest.fixture
def generated(project: Project, monkeypatch: pytest.MonkeyPatch) -> Project:
    monkeypatch.setattr(
        "assay.orchestrate.resources.create_client",
        lambda provider, model_id: FakeLLM(model_id=model_id, reply=reply),
    )
    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded
    assert pipeline.materialize_documents(PARTITION, log_level="ERROR").succeeded
    return project


def test_measures_every_document_with_every_static_metric(generated: Project) -> None:
    run = build_pipeline(generated.config).materialize_metrics(PARTITION, log_level="ERROR")

    counts = run.metadata["metrics"]
    assert run.succeeded
    assert (counts["documents"], counts["rows"]) == (3, 3 * STATIC_METRICS)
    rows = ParquetStore(generated.data).read(METRICS_TABLE)
    integrity = [r for r in rows if r.metric == "referential_integrity"]
    assert all(r.value == pytest.approx(2 / 3) for r in integrity)  # `greet`, `name` resolve
    assert counts["critical"] == len(integrity) == 3
    assert all("made_up_helper" in r.explanation for r in integrity)


def test_rerunning_replaces_the_partition(generated: Project) -> None:
    pipeline = build_pipeline(generated.config)
    pipeline.materialize_metrics(PARTITION, log_level="ERROR")

    run = pipeline.materialize_metrics(PARTITION, log_level="ERROR")

    assert run.metadata["metrics"]["rows"] == 3 * STATIC_METRICS


def test_measuring_needs_documents_first(project: Project) -> None:
    pipeline = build_pipeline(project.config)
    assert pipeline.materialize_corpus(PARTITION, log_level="ERROR").succeeded

    assert not pipeline.materialize_metrics(PARTITION, log_level="CRITICAL").succeeded


def test_the_measure_command(generated: Project) -> None:
    result = CliRunner().invoke(app, ["measure", "--config", str(generated.config), "--epoch", "0"])

    assert result.exit_code == 0, result.output
    assert "12 metric values for 3 documents, 3 critical" in result.output
