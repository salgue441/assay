from datetime import UTC, datetime
from pathlib import Path

from assay.calibrate.reveals import Reveal, conflicting, read_reveals, record_reveal


def reveal(scorer: str = "a" * 16, *, day: int = 1) -> Reveal:
    return Reveal(
        revealed_at=datetime(2026, 10, day, 12, 0, tzinfo=UTC),
        scorer_version=scorer,
        rubric_hash="b" * 64,
        raters=("r1",),
        test_repos=("o/y",),
    )


def test_reveals_are_recorded_and_read_back_in_order(tmp_path: Path) -> None:
    ledger = tmp_path / "reports" / "test_reveals.jsonl"

    record_reveal(ledger, reveal(day=1))
    record_reveal(ledger, reveal("c" * 16, day=2))

    assert read_reveals(ledger) == [reveal(day=1), reveal("c" * 16, day=2)]


def test_a_missing_ledger_has_no_reveals(tmp_path: Path) -> None:
    assert read_reveals(tmp_path / "none.jsonl") == []


def test_only_a_changed_configuration_conflicts() -> None:
    earlier = [reveal(day=1), reveal("c" * 16, day=2)]

    assert conflicting(earlier, reveal(day=3)) == [reveal("c" * 16, day=2)]
    assert conflicting([reveal(day=1)], reveal(day=3)) == []


def test_a_reveal_describes_itself() -> None:
    assert reveal().describe() == "2026-10-01 12:00 scorer aaaaaaaaaaaaaaaa, raters r1, test o/y"
