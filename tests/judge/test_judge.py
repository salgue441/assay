import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assay.generate import Budget, Document, DocumentGenerator, PromptTemplate, Subject
from assay.judge import JUDGMENTS, DocumentJudge, JudgeModel, Rubric
from assay.llm import ChatRequest, ChatResponse, LLMError, Provider
from assay.storage import ParquetStore, Partition

from ..fake_llm import FakeLLM
from ..generate.units import units

PROJECT_ROOT = Path(__file__).parents[2]
RUBRIC = Rubric.load(PROJECT_ROOT / "config" / "rubrics", "v1")
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
UNITS = units()
UNITS_BY_KEY = {(u.unit_id, u.commit): u for u in UNITS}


def reply_json(score: int = 4, issues: tuple[str, ...] = ("one vague line",)) -> str:
    return json.dumps({"issues": list(issues), "rationale": "fits level", "score": score})


def documents() -> list[Document]:
    writer = Subject("writer", Provider.OLLAMA, FakeLLM(reply=lambda request: "## doc\nText."))
    prompt = PromptTemplate(version="v1", system="s", user="$name\n$source")
    return (
        DocumentGenerator((writer,), prompt, seed=1, max_output_tokens=64)
        .generate(UNITS[:2], language="python")
        .documents
    )


DOCUMENTS = documents()


def judge(*judges: JudgeModel, runs: int = 2, budget: Budget | None = None) -> DocumentJudge:
    return DocumentJudge(
        judges or (JudgeModel("grader", Provider.OLLAMA, FakeLLM(reply=lambda r: reply_json())),),
        RUBRIC,
        criteria=("accuracy", "clarity"),
        runs=runs,
        seed=100,
        max_output_tokens=256,
        budget=budget,
        clock=lambda: NOW,
    )


def test_grades_every_document_criterion_run_and_judge() -> None:
    a = JudgeModel("a", Provider.OLLAMA, FakeLLM(model_id="a:1b", reply=lambda r: reply_json()))
    b = JudgeModel(
        "b", Provider.OLLAMA, FakeLLM(model_id="b:1b", reply=lambda r: reply_json(5, ()))
    )

    report = judge(a, b).judge(DOCUMENTS, UNITS_BY_KEY, language="python")

    assert len(report.judgments) == 2 * len(DOCUMENTS) * 2 * 2
    assert len({j.judgment_id for j in report.judgments}) == len(report.judgments)
    assert {(j.judge, j.score) for j in report.judgments} == {("a", 4), ("b", 5)}
    assert (report.skipped, report.failures, report.stopped_by_budget) == (0, [], False)


def test_each_criterion_is_a_separate_constrained_call_with_a_seed_per_run() -> None:
    llm = FakeLLM(reply=lambda r: reply_json())

    judge(JudgeModel("g", Provider.OLLAMA, llm, temperature=0.3)).judge(
        DOCUMENTS[:1], UNITS_BY_KEY, language="python"
    )

    assert [
        ("criterion: accuracy" in r.user, "criterion: clarity" in r.user) for r in llm.requests
    ] == [
        (True, False),
        (True, False),
        (False, True),
        (False, True),
    ]
    assert [r.seed for r in llm.requests] == [100, 101, 100, 101]
    assert all(r.json_schema is not None and r.temperature == 0.3 for r in llm.requests)
    assert all(DOCUMENTS[0].text.strip() in r.user for r in llm.requests)


def test_judgments_record_the_grade_and_its_provenance() -> None:
    (judgment, *_) = judge().judge(DOCUMENTS[:1], UNITS_BY_KEY, language="python").judgments

    assert (judgment.document_id, judgment.subject, judgment.judge) == (
        DOCUMENTS[0].document_id,
        "writer",
        "grader",
    )
    assert (judgment.criterion, judgment.run_index, judgment.seed) == ("accuracy", 0, 100)
    assert (judgment.score, judgment.issues, judgment.rationale) == (
        4,
        ("one vague line",),
        "fits level",
    )
    assert (judgment.rubric_version, judgment.rubric_hash) == ("v1", RUBRIC.hash)
    assert (judgment.model_revision, judgment.judged_at) == ("rev-1", NOW)
    assert not judgment.contradicts_rubric


def test_a_top_score_with_listed_issues_is_flagged() -> None:
    llm = FakeLLM(reply=lambda r: reply_json(5, ("but this is wrong",)))

    report = judge(JudgeModel("g", Provider.OLLAMA, llm)).judge(
        DOCUMENTS[:1], UNITS_BY_KEY, language="python"
    )

    assert all(j.contradicts_rubric for j in report.judgments)


def test_existing_judgments_are_skipped_without_a_call() -> None:
    first = judge().judge(DOCUMENTS, UNITS_BY_KEY, language="python")
    llm = FakeLLM(reply=lambda r: reply_json())

    again = judge(JudgeModel("grader", Provider.OLLAMA, llm)).judge(
        DOCUMENTS,
        UNITS_BY_KEY,
        language="python",
        existing={j.judgment_id for j in first.judgments},
    )

    assert (again.judgments, again.skipped, llm.requests) == ([], len(first.judgments), [])


@pytest.mark.parametrize(
    "change", [{"model_revision": "rev-2"}, {"model_id": "other:1b"}], ids=["revision", "model"]
)
def test_a_different_judge_model_gives_new_ids(change: dict[str, str]) -> None:
    before = judge().judge(DOCUMENTS, UNITS_BY_KEY, language="python").judgments
    other = JudgeModel("grader", Provider.OLLAMA, FakeLLM(reply=lambda r: reply_json(), **change))  # type: ignore[arg-type]

    after = judge(other).judge(DOCUMENTS, UNITS_BY_KEY, language="python").judgments

    assert not {j.judgment_id for j in before} & {j.judgment_id for j in after}


def test_invalid_replies_and_failed_calls_are_recorded_and_the_rest_continue() -> None:
    def reply(request: ChatRequest) -> str:
        return "Score: 5" if request.seed == 101 else reply_json()

    def fail(request: ChatRequest) -> bool:
        return "criterion: clarity" in request.user and request.seed == 100

    llm = FakeLLM(reply=reply, fail_when=fail)

    report = judge(JudgeModel("g", Provider.OLLAMA, llm)).judge(
        DOCUMENTS[:1], UNITS_BY_KEY, language="python"
    )

    assert len(report.judgments) == 1
    assert sorted((f.criterion, f.run_index) for f in report.failures) == [
        ("accuracy", 1),
        ("clarity", 0),
        ("clarity", 1),
    ]
    assert any("invalid judge reply" in f.error for f in report.failures)
    assert any("scripted failure" in f.error for f in report.failures)


class MissingModel:
    """A client whose model is not installed."""

    model_id = "missing:1b"
    context_tokens = 8192

    @property
    def model_revision(self) -> str:
        raise LLMError("not pulled", transient=False)

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("must not be called")


def test_a_judge_that_cannot_be_found_stops_judging() -> None:
    missing = JudgeModel("m", Provider.OLLAMA, MissingModel())

    with pytest.raises(LLMError, match="not pulled"):
        judge(missing).judge(DOCUMENTS, UNITS_BY_KEY, language="python")


def test_judging_stops_once_the_budget_is_spent() -> None:
    llm = FakeLLM(reply=lambda r: reply_json(), cost_usd=0.5)

    report = judge(JudgeModel("g", Provider.OLLAMA, llm), budget=Budget(1.0)).judge(
        DOCUMENTS, UNITS_BY_KEY, language="python"
    )

    assert len(report.judgments) == 2
    assert report.stopped_by_budget


def test_fits_checks_every_criterion_against_the_smallest_context() -> None:
    roomy = JudgeModel("roomy", Provider.OLLAMA, FakeLLM(context_tokens=100_000))
    tight = JudgeModel("tight", Provider.OLLAMA, FakeLLM(context_tokens=600))
    unit = UNITS_BY_KEY[(DOCUMENTS[0].unit_id, DOCUMENTS[0].commit)]

    assert judge(roomy).fits(DOCUMENTS[0], unit, language="python")
    assert not judge(roomy, tight).fits(DOCUMENTS[0], unit, language="python")


@pytest.mark.parametrize(
    ("kwargs", "problem"),
    [
        ({"runs": 0}, "runs must be at least 1"),
        ({"criteria": ("accuracy", "charm")}, "unknown: \\['charm'\\]"),
        ({"criteria": ()}, "must be non-empty"),
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, object], problem: str) -> None:
    options: dict[str, object] = {
        "criteria": ("accuracy",),
        "runs": 1,
        "seed": 1,
        "max_output_tokens": 1,
    } | kwargs
    with pytest.raises(ValueError, match=problem):
        DocumentJudge((JudgeModel("g", Provider.OLLAMA, FakeLLM()),), RUBRIC, **options)  # type: ignore[arg-type]


def test_requires_a_judge() -> None:
    with pytest.raises(ValueError, match="at least one judge"):
        DocumentJudge((), RUBRIC, criteria=("accuracy",), runs=1, seed=1, max_output_tokens=1)


def test_judgments_round_trip_through_storage(tmp_path: Path) -> None:
    judgments = judge().judge(DOCUMENTS, UNITS_BY_KEY, language="python").judgments
    store = ParquetStore(tmp_path)

    store.write(JUDGMENTS, judgments, Partition("owner/shapes", 0))

    assert store.read(JUDGMENTS) == judgments
