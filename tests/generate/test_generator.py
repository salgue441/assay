from datetime import UTC, datetime
from pathlib import Path

import pytest

from assay.generate import (
    DOCUMENTS,
    Budget,
    DocumentGenerator,
    PromptTemplate,
    Subject,
)
from assay.llm import ChatRequest, ChatResponse, LLMError, Provider
from assay.storage import ParquetStore, Partition

from ..fake_llm import FakeLLM
from .units import units

PROMPT = PromptTemplate(version="v1", system="Be precise.", user="Document $kind `$name`.\n$source")
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
UNITS = units()


def generator(*subjects: Subject, budget: Budget | None = None) -> DocumentGenerator:
    return DocumentGenerator(
        subjects or (Subject("writer", Provider.OLLAMA, FakeLLM()),),
        PROMPT,
        seed=11,
        max_output_tokens=256,
        budget=budget,
        clock=lambda: NOW,
    )


def test_documents_every_unit_with_every_subject() -> None:
    a, b = FakeLLM(model_id="a:1b"), FakeLLM(model_id="b:1b")

    report = generator(
        Subject("a", Provider.OLLAMA, a), Subject("b", Provider.OLLAMA, b, temperature=0.5)
    ).generate(UNITS, language="python")

    assert len(report.documents) == 2 * len(UNITS)
    assert {d.subject for d in report.documents} == {"a", "b"}
    assert len({d.document_id for d in report.documents}) == len(report.documents)
    assert (report.skipped, report.failures, report.stopped_by_budget) == (0, [], False)


def test_requests_carry_the_prompt_and_sampling_settings() -> None:
    llm = FakeLLM()
    generator(Subject("w", Provider.OLLAMA, llm, temperature=0.3)).generate(
        UNITS[:1], language="python"
    )

    (request,) = llm.requests
    assert request == ChatRequest(
        user=f"Document module `shapes.core`.\n{UNITS[0].source.rstrip()}",
        system="Be precise.",
        temperature=0.3,
        seed=11,
        max_output_tokens=256,
    )


def test_documents_record_their_provenance() -> None:
    llm = FakeLLM(reply=lambda request: "## `shapes.core.scale`\nDoubles it.")

    (document,) = (
        generator(Subject("w", Provider.OLLAMA, llm))
        .generate(UNITS[-1:], language="python")
        .documents
    )

    assert document.text == "## `shapes.core.scale`\nDoubles it."
    assert (document.unit_id, document.unit_hash, document.commit) == (
        UNITS[-1].unit_id,
        UNITS[-1].unit_hash,
        UNITS[-1].commit,
    )
    assert (document.model_id, document.model_revision, document.prompt_hash) == (
        "fake-model:1b",
        "rev-1",
        PROMPT.hash,
    )
    assert (document.seed, document.max_output_tokens, document.generated_at) == (11, 256, NOW)
    DOCUMENTS.validate(DOCUMENTS.to_arrow([document]))


def test_existing_documents_are_skipped_without_a_call() -> None:
    first = generator().generate(UNITS, language="python")
    llm = FakeLLM()

    again = generator(Subject("writer", Provider.OLLAMA, llm)).generate(
        UNITS, language="python", existing={d.document_id for d in first.documents}
    )

    assert (again.documents, again.skipped, llm.requests) == ([], len(UNITS), [])


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "rev-2"},
        {"model_id": "other:1b"},
    ],
)
def test_a_different_model_gives_new_document_ids(change: dict[str, str]) -> None:
    before = generator().generate(UNITS, language="python").documents
    after = (
        generator(Subject("writer", Provider.OLLAMA, FakeLLM(**change)))
        .generate(  # type: ignore[arg-type]
            UNITS, language="python"
        )
        .documents
    )

    assert not {d.document_id for d in before} & {d.document_id for d in after}


def test_failed_units_are_recorded_and_the_rest_continue() -> None:
    llm = FakeLLM(fail_when=lambda request: "`shapes.core.Square`" in request.user)

    report = generator(Subject("w", Provider.OLLAMA, llm)).generate(UNITS, language="python")

    assert len(report.documents) == len(UNITS) - 1
    (failure,) = report.failures
    assert (failure.unit_id, failure.subject) == (
        "owner/shapes:shapes/core.py#shapes.core.Square",
        "w",
    )
    assert "scripted failure" in failure.error


class MissingModel:
    """A client whose model is not installed."""

    model_id = "missing:1b"

    @property
    def model_revision(self) -> str:
        raise LLMError("not pulled", transient=False)

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("must not be called")


def test_a_model_that_cannot_be_found_stops_generation() -> None:
    with pytest.raises(LLMError, match="not pulled"):
        generator(Subject("w", Provider.OLLAMA, MissingModel())).generate(UNITS, language="python")


def test_generation_stops_once_the_budget_is_spent() -> None:
    llm = FakeLLM(cost_usd=0.4)

    report = generator(Subject("w", Provider.OLLAMA, llm), budget=Budget(1.0)).generate(
        UNITS, language="python"
    )

    assert len(report.documents) == 3  # 0.4 + 0.4 < 1.0, the third call reaches 1.2
    assert report.stopped_by_budget


def test_fits_compares_the_estimated_prompt_with_the_smallest_context() -> None:
    small_unit, large_unit = UNITS[-1], UNITS[0]
    system, user = PROMPT.render(small_unit, language="python")
    small_need = ChatRequest(user=user, system=system, max_output_tokens=256).estimated_tokens()
    budget = DocumentGenerator(
        (
            Subject("roomy", Provider.OLLAMA, FakeLLM(context_tokens=100_000)),
            Subject("tight", Provider.OLLAMA, FakeLLM(context_tokens=small_need)),
        ),
        PROMPT,
        seed=1,
        max_output_tokens=256,
    )

    assert budget.fits(small_unit, language="python")
    assert not budget.fits(large_unit, language="python")


def test_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="at least one subject"):
        DocumentGenerator((), PROMPT, seed=1, max_output_tokens=1)


def test_load_writes_documents_through_the_contract(tmp_path: Path) -> None:
    documents = generator().generate(UNITS, language="python").documents
    store = ParquetStore(tmp_path)

    store.write(DOCUMENTS, documents, Partition("owner/shapes", 0))

    assert store.read(DOCUMENTS) == documents
