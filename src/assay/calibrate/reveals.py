"""
A ledger of every time the held-out test split was evaluated, so "the test
split is used once" can be checked rather than assumed.

Evaluating it again under the same configuration repeats a deterministic
result and leaks nothing. Evaluating it under a different configuration,
for example after changing weights because of what the test split showed,
is the leak the ledger exists to catch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Reveal:
    revealed_at: datetime
    scorer_version: str
    rubric_hash: str
    raters: tuple[str, ...]
    test_repos: tuple[str, ...]

    def configuration(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
        """What determines the test result; reveals that share it are repeats."""

        return (self.scorer_version, self.rubric_hash, self.raters, self.test_repos)

    def describe(self) -> str:
        return (
            f"{self.revealed_at:%Y-%m-%d %H:%M} scorer {self.scorer_version}, "
            f"raters {', '.join(self.raters)}, test {', '.join(self.test_repos)}"
        )


def read_reveals(path: Path) -> list[Reveal]:
    """The reveals recorded at `path`, oldest first; none if it is missing."""

    if not path.exists():
        return []
    reveals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reveals.append(
            Reveal(
                revealed_at=datetime.fromisoformat(row["revealed_at"]),
                scorer_version=row["scorer_version"],
                rubric_hash=row["rubric_hash"],
                raters=tuple(row["raters"]),
                test_repos=tuple(row["test_repos"]),
            )
        )
    return reveals


def record_reveal(path: Path, reveal: Reveal) -> None:
    """Append `reveal` to the ledger at `path`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(reveal) | {"revealed_at": reveal.revealed_at.isoformat()}
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(row, sort_keys=True) + "\n")


def conflicting(previous: Sequence[Reveal], current: Reveal) -> list[Reveal]:
    """Earlier reveals under a different configuration than `current`."""

    return [r for r in previous if r.configuration() != current.configuration()]
