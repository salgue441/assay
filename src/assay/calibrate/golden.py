"""
The golden set: development documents with their human consensus ratings,
frozen under a version hash, that drift monitoring re-scores to watch the
judge. The test split never enters it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

from assay.calibrate.agreement import weighted_kappa
from assay.calibrate.study import RatedDocument
from assay.judge import Judgment

ACTIVE = "active"
"""File naming the version in use; one golden set is active at a time."""
_VERSION_CHARS = 16


@dataclass(frozen=True)
class GoldenDocument:
    document_id: str
    unit_id: str
    repo: str
    subject: str
    overall: float
    criteria: Mapping[str, int]
    decision: str | None
    """The raters' shared decision, or None when they disagreed."""
    strata: Mapping[str, str]


@dataclass(frozen=True)
class GoldenSet:
    version: str
    """Hash of the documents and their consensus ratings."""
    frozen_at: datetime
    scorer_version: str
    rubric_hash: str
    raters: tuple[str, ...]
    documents: tuple[GoldenDocument, ...]


def select_golden(
    documents: Sequence[RatedDocument], *, size: int, seed: int, stratify_by: Sequence[str] = ()
) -> list[RatedDocument]:
    """
    Up to `size` of `documents`, allocated to each stratum (the combination
    of its `stratify_by` values) in proportion to its share, largest
    remainders first, and drawn within a stratum in an order fixed by
    `seed`. All documents when there are no more than `size`. Sorted by id.
    """

    if len(documents) <= size:
        return sorted(documents, key=lambda d: d.document_id)

    strata: dict[tuple[str, ...], list[RatedDocument]] = defaultdict(list)
    for document in documents:
        strata[tuple(document.strata.get(v, "") for v in stratify_by)].append(document)

    exact = {key: size * len(members) / len(documents) for key, members in strata.items()}
    quota = {key: math.floor(share) for key, share in exact.items()}
    by_remainder = sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))
    for key in by_remainder[: size - sum(quota.values())]:
        quota[key] += 1

    chosen = [
        document
        for key, members in sorted(strata.items())
        for document in sorted(members, key=lambda d: _rank(seed, d.document_id))[: quota[key]]
    ]
    return sorted(chosen, key=lambda d: d.document_id)


def freeze_golden(
    documents: Sequence[RatedDocument],
    *,
    frozen_at: datetime,
    scorer_version: str,
    rubric_hash: str,
) -> GoldenSet:
    golden = tuple(
        GoldenDocument(
            document_id=d.document_id,
            unit_id=d.unit_id,
            repo=d.repo,
            subject=d.subject,
            overall=d.overall,
            criteria=dict(d.criteria),
            decision=d.human_decision.value if d.human_decision else None,
            strata=dict(d.strata),
        )
        for d in sorted(documents, key=lambda d: d.document_id)
    )
    return GoldenSet(
        version=golden_version(golden),
        frozen_at=frozen_at,
        scorer_version=scorer_version,
        rubric_hash=rubric_hash,
        raters=tuple(sorted({r for d in documents for r in d.ratings})),
        documents=golden,
    )


def golden_version(documents: Sequence[GoldenDocument]) -> str:
    """A short hash of the documents and their consensus, so any change to
    membership or ratings gives a new version."""

    canonical = json.dumps(
        [
            [d.document_id, d.overall, sorted(d.criteria.items()), d.decision]
            for d in sorted(documents, key=lambda d: d.document_id)
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_VERSION_CHARS]


def write_golden(directory: Path, golden: GoldenSet) -> Path:
    """Store `golden` as `<version>.json` and make it the active set."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{golden.version}.json"
    data = asdict(golden) | {"frozen_at": golden.frozen_at.isoformat()}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / ACTIVE).write_text(golden.version + "\n", encoding="utf-8")
    return path


def read_active_golden(directory: Path) -> GoldenSet | None:
    """The active golden set, or None when none was frozen."""

    pointer = directory / ACTIVE
    if not pointer.exists():
        return None
    version = pointer.read_text(encoding="utf-8").strip()
    data = json.loads((directory / f"{version}.json").read_text(encoding="utf-8"))
    return GoldenSet(
        version=data["version"],
        frozen_at=datetime.fromisoformat(data["frozen_at"]),
        scorer_version=data["scorer_version"],
        rubric_hash=data["rubric_hash"],
        raters=tuple(data["raters"]),
        documents=tuple(GoldenDocument(**d) for d in data["documents"]),
    )


def _rank(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}:golden:{document_id}".encode()).hexdigest()


def golden_kappa(
    golden: GoldenSet, judgments: Iterable[Judgment], *, criteria: Sequence[str]
) -> dict[str, float]:
    """
    Quadratic-weighted kappa per criterion between the judge's median grade
    over its runs and the frozen human consensus, over the golden documents
    the judge graded. A criterion where kappa is undefined is left out.
    """

    consensus = {d.document_id: d.criteria for d in golden.documents}
    runs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for judgment in judgments:
        if judgment.document_id in consensus and judgment.criterion in criteria:
            runs[(judgment.document_id, judgment.criterion)].append(judgment.score)

    kappa = {}
    for criterion in criteria:
        pairs = [
            (math.floor(median(grades) + 0.5), consensus[document_id][criterion])
            for (document_id, name), grades in sorted(runs.items())
            if name == criterion and criterion in consensus[document_id]
        ]
        if not pairs:
            continue
        try:
            kappa[criterion] = weighted_kappa([j for j, _ in pairs], [h for _, h in pairs])
        except ValueError:
            continue
    return kappa
