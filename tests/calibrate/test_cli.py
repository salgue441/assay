import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from assay.cli import app
from assay.rating import RATINGS
from assay.scoring import SCORES, ScorerConfig
from assay.settings import load_settings
from assay.storage import ParquetStore, Partition

from .test_study import rating, score

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """The default config over a store holding two repositories' scores and
    ratings by a human rater and an LLM rater."""

    data = yaml.safe_load((PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    data["storage"] = {"root": str(tmp_path / "data"), "duckdb_path": str(tmp_path / "a.duckdb")}
    for key in ("manifest", "cache_dir"):
        data["corpus"][key] = str(PROJECT_ROOT / "config" / Path(data["corpus"][key]).name)
    for section, key in (("generation", "prompt_dir"), ("judge", "rubric_dir")):
        data[section][key] = str(PROJECT_ROOT / data[section][key])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    settings = load_settings(path)
    scoring = settings.scoring
    version = ScorerConfig(
        weights=scoring.weights,
        judge=scoring.judge,
        publish_at_or_above=scoring.routing.publish_at_or_above,
        escalate_below=scoring.routing.escalate_below,
    ).version
    rubric_hash = settings.judge.rubric().hash
    store = ParquetStore(settings.storage.root)
    for repo in ("o/x", "o/y"):
        names = [f"{repo}-{i}" for i in range(6)]
        store.write(
            SCORES,
            [
                score(n, repo=repo, composite=15.0 * i, scorer_version=version)
                for i, n in enumerate(names)
            ],
            Partition(repo, 0),
        )
        store.write(
            RATINGS,
            [
                rating(n, rater=rater, overall=i % 5 + 1, rubric_hash=rubric_hash).model_copy(
                    update={"repo": repo}
                )
                for i, n in enumerate(names)
                for rater in ("r1", "llm-claude")
            ],
            Partition(repo, 0),
        )
    return path


def invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["calibrate", "--resamples", "50", *args])
    return result.exit_code, result.output


def test_calibrates_against_the_configured_human_raters(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    code, text = invoke("--config", str(config), "--output", str(output))

    assert code == 0, text
    assert "12 rated documents from raters r1" in text
    assert "held out" in text
    assert "without judge" in text
    assert "have no stored text" in text  # the fixture stores no documents
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["raters"] == ["r1"]
    assert set(report["correlation"]) == {"dev"}


def test_the_test_split_is_evaluated_only_with_the_flag(config: Path, tmp_path: Path) -> None:
    code, text = invoke("--config", str(config), "--test", "--output", str(tmp_path / "r.json"))

    assert code == 0, text
    assert "Test split" in text
    assert "target 0.7" in text


def test_another_rater_is_marked_as_not_human_agreement(config: Path, tmp_path: Path) -> None:
    code, text = invoke(
        "--config", str(config), "--rater", "llm-claude", "--output", str(tmp_path / "r.json")
    )

    assert code == 0, text
    assert "llm-claude not in evaluation.raters" in text
    assert "raters llm-claude" in text


def test_ratings_from_one_repository_cannot_be_split(config: Path, tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "data")
    (only,) = [
        r
        for r in store.read(RATINGS, repo="o/x", epoch=0)
        if r.unit_id == "o/x-0" and r.rater_id == "r1"
    ]
    store.write(RATINGS, [only.model_copy(update={"rater_id": "r2"})], Partition("o/x", 1))

    code, text = invoke("--config", str(config), "--rater", "r2")

    assert code == 1
    assert "at least two repositories" in text
    assert "got 1 documents from 1" in text


def test_no_matching_ratings_is_an_error(config: Path) -> None:
    code, text = invoke("--config", str(config), "--rater", "r9")

    assert code == 1
    assert "rated by r9" in text
