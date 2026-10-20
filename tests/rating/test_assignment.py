import hashlib
from pathlib import Path

import pytest

from assay.rating import assigned, in_overlap

from .fixtures import make_store


def test_the_overlap_is_close_to_the_requested_share() -> None:
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(2000)]

    share = sum(in_overlap(i, seed=7, share=0.2) for i in ids) / len(ids)

    assert 0.17 < share < 0.23


def test_the_overlap_depends_on_the_seed() -> None:
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(200)]

    first = {i for i in ids if in_overlap(i, seed=1, share=0.5)}
    second = {i for i in ids if in_overlap(i, seed=2, share=0.5)}

    assert first != second


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_share_must_be_a_fraction(share: float) -> None:
    with pytest.raises(ValueError, match=r"share must be in \(0, 1\]"):
        in_overlap("d", seed=1, share=share)


def test_the_primary_rater_gets_every_document_in_a_personal_stable_order(tmp_path: Path) -> None:
    _, documents = make_store(tmp_path)

    first = assigned(documents, rater_id="r1", seed=7)
    again = assigned(list(reversed(documents)), rater_id="r1", seed=7)
    other = assigned(documents, rater_id="r2", seed=7)

    assert sorted(d.document_id for d in first) == sorted(d.document_id for d in documents)
    assert first == again
    assert first != other


def test_additional_raters_share_the_same_overlap_in_different_orders(tmp_path: Path) -> None:
    _, documents = make_store(tmp_path)

    r2 = assigned(documents, rater_id="r2", seed=7, share=0.5)
    r3 = assigned(documents, rater_id="r3", seed=7, share=0.5)

    assert {d.document_id for d in r2} == {d.document_id for d in r3}
    assert 0 < len(r2) < len(documents)
