from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from assay.calibrate.golden import (
    freeze_golden,
    golden_version,
    read_active_golden,
    select_golden,
    write_golden,
)
from assay.calibrate.study import RatedDocument

from .test_study import join, rating, score

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def documents(python: int = 30, go: int = 10) -> list[RatedDocument]:
    names = [(f"py{i}", "python") for i in range(python)] + [(f"go{i}", "go") for i in range(go)]
    rated = join([score(n) for n, _ in names], [rating(n, overall=3) for n, _ in names])
    language = dict(names)
    return [replace(d, strata={"language": language[d.unit_id]}) for d in rated]


def test_strata_are_sampled_in_proportion() -> None:
    chosen = select_golden(documents(), size=8, seed=1, stratify_by=("language",))

    languages = [d.strata["language"] for d in chosen]
    assert (languages.count("python"), languages.count("go")) == (6, 2)


def test_selection_is_reproducible_and_seeded() -> None:
    pool = documents()

    first = select_golden(pool, size=8, seed=1, stratify_by=("language",))

    assert first == select_golden(list(reversed(pool)), size=8, seed=1, stratify_by=("language",))
    assert first != select_golden(pool, size=8, seed=2, stratify_by=("language",))


def test_a_small_pool_is_taken_whole() -> None:
    pool = documents(python=3, go=1)

    assert len(select_golden(pool, size=100, seed=1)) == 4


def test_largest_remainders_fill_the_size_exactly() -> None:
    chosen = select_golden(documents(python=5, go=5), size=3, seed=1, stratify_by=("language",))

    assert len(chosen) == 3


def test_the_version_changes_with_membership_or_ratings() -> None:
    golden = freeze_golden(documents()[:5], frozen_at=NOW, scorer_version="s", rubric_hash="h")
    fewer = freeze_golden(documents()[:4], frozen_at=NOW, scorer_version="s", rubric_hash="h")
    rerated = [replace(d, overall=4.0) for d in documents()[:5]]

    assert golden.version != fewer.version
    assert golden.version != golden_version(
        freeze_golden(rerated, frozen_at=NOW, scorer_version="s", rubric_hash="h").documents
    )
    assert (
        golden.version
        == freeze_golden(
            documents()[:5],
            frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
            scorer_version="t",
            rubric_hash="h",
        ).version
    )


def test_a_frozen_set_is_written_and_read_back_as_active(tmp_path: Path) -> None:
    golden = freeze_golden(documents()[:5], frozen_at=NOW, scorer_version="s", rubric_hash="h")

    path = write_golden(tmp_path, golden)

    assert path.name == f"{golden.version}.json"
    assert read_active_golden(tmp_path) == golden
    assert read_active_golden(tmp_path / "missing") is None
