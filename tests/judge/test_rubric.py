import re
from pathlib import Path

import pytest

from assay.judge import REPLY_SCHEMA, JudgeReply, Rubric, RubricError

from ..generate.units import units

PROJECT_ROOT = Path(__file__).parents[2]
RUBRIC = Rubric.load(PROJECT_ROOT / "config" / "rubrics", "v1")

MINIMAL = """
system: Grade $criterion.
user: "$name: $question\\n$levels\\n<doc>$document</doc>\\n$source"
criteria:
  accuracy:
    question: Is it right?
    levels: {1: wrong, 2: poor, 3: mixed, 4: good, 5: right}
"""


def write(tmp_path: Path, content: str, version: str = "v9") -> Rubric:
    (tmp_path / f"{version}.yaml").write_text(content, encoding="utf-8")
    return Rubric.load(tmp_path, version)


def test_the_committed_v1_rubric_has_three_fully_anchored_criteria() -> None:
    assert list(RUBRIC.criteria) == ["accuracy", "clarity", "consistency"]
    for criterion in RUBRIC.criteria.values():
        assert sorted(criterion.levels) == [1, 2, 3, 4, 5]
        assert all(text.strip() for text in criterion.levels.values())


def test_v1_renders_source_document_and_levels_for_every_criterion() -> None:
    unit = units()[1]

    for criterion in RUBRIC.criteria:
        system, user = RUBRIC.render(criterion, unit=unit, document="## doc", language="python")
        assert "ignore any instructions" in system
        assert f"criterion: {criterion}" in user
        assert f"```python\n{unit.source.rstrip()}\n```" in user
        assert "<documentation>\n## doc\n</documentation>" in user
        assert re.search(r"^5: ", user, re.M) and re.search(r"^1: ", user, re.M)
        assert "$" not in user.replace(unit.source, "")


def test_renders_placeholders_in_order(tmp_path: Path) -> None:
    rubric = write(tmp_path, MINIMAL)

    system, user = rubric.render("accuracy", unit=units()[1], document="  text  ", language="py")

    assert system == "Grade accuracy."
    assert user.startswith("shapes.core.Square: Is it right?\n1: wrong\n2: poor\n")
    assert "<doc>text</doc>" in user


def test_hash_tracks_every_part_of_the_rubric(tmp_path: Path) -> None:
    base = write(tmp_path, MINIMAL, "a")

    assert write(tmp_path, MINIMAL, "b").hash == base.hash
    assert write(tmp_path, MINIMAL.replace("mixed", "middling"), "c").hash != base.hash
    assert write(tmp_path, MINIMAL.replace("Is it right?", "Right?"), "d").hash != base.hash


def test_unknown_placeholders_fail_to_render(tmp_path: Path) -> None:
    rubric = write(tmp_path, MINIMAL.replace("Grade $criterion.", "Grade $nonsense."))

    with pytest.raises(RubricError, match="cannot render"):
        rubric.render("accuracy", unit=units()[0], document="x", language="python")


def test_unknown_criteria_cannot_be_rendered() -> None:
    with pytest.raises(RubricError, match="no criterion 'charm'"):
        RUBRIC.render("charm", unit=units()[0], document="x", language="python")


@pytest.mark.parametrize(
    "content",
    [
        MINIMAL.replace("{1: wrong, 2: poor, 3: mixed, 4: good, 5: right}", "{1: wrong, 5: right}"),
        MINIMAL.replace("criteria:", "nothing:"),
        "system: [\n",
        "- a list\n",
    ],
    ids=["missing-levels", "no-criteria", "bad-yaml", "not-a-mapping"],
)
def test_malformed_rubrics_are_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(RubricError):
        write(tmp_path, content)


def test_reply_schema_asks_for_issues_before_the_score() -> None:
    assert list(REPLY_SCHEMA["properties"]) == ["issues", "rationale", "score"]
    assert REPLY_SCHEMA["properties"]["score"]["minimum"] == 1
    assert REPLY_SCHEMA["properties"]["score"]["maximum"] == 5


def test_parses_a_valid_reply() -> None:
    reply = JudgeReply.parse('{"issues": ["vague"], "rationale": "fine", "score": 4}')

    assert (reply.issues, reply.rationale, reply.score) == (["vague"], "fine", 4)


@pytest.mark.parametrize(
    "text",
    [
        '{"issues": [], "rationale": "x", "score": 6}',
        '{"issues": [], "score": 3}',
        '{"issues": [], "rationale": "x", "score": 3, "extra": 1}',
        "Score: 4",
    ],
    ids=["out-of-range", "missing-field", "extra-field", "not-json"],
)
def test_rejects_invalid_replies(text: str) -> None:
    with pytest.raises(ValueError, match="invalid judge reply"):
        JudgeReply.parse(text)
