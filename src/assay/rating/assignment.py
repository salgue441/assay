"""
Which documents a rater sees, and in what order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from assay.generate import Document

_HASH_SPACE = 16**64


def _fraction(*parts: object) -> float:
    """A deterministic number in [0, 1) for the given parts."""

    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest, 16) / _HASH_SPACE


def in_overlap(document_id: str, *, seed: int, share: float) -> bool:
    """
    Whether a document is in the overlap subset: the share of documents that
    every additional rater rates, so their ratings can be compared.

    Membership depends only on the document and the seed, never on the
    rater, so all additional raters get the same subset.
    """

    if not 0 < share <= 1:
        msg = f"share must be in (0, 1], got {share}"
        raise ValueError(msg)
    return _fraction(seed, "overlap", document_id) < share


def assigned(
    documents: Iterable[Document], *, rater_id: str, seed: int, share: float = 1.0
) -> list[Document]:
    """
    The documents for one rater, in that rater's own random order.

    `share=1.0` assigns every document (the primary rater); a smaller share
    assigns the overlap subset. The order is shuffled per rater, so position
    effects do not line up across raters, and is stable between sessions.
    """

    chosen = [
        d for d in documents if share == 1 or in_overlap(d.document_id, seed=seed, share=share)
    ]
    return sorted(chosen, key=lambda d: _fraction(seed, "order", rater_id, d.document_id))
