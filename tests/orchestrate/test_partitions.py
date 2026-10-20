from datetime import date

import pytest

from assay.corpus import Manifest, RepoSpec
from assay.extract import SourceLanguage
from assay.orchestrate import CorpusPartitions, partition_key


def spec(name: str, epochs: int) -> RepoSpec:
    return RepoSpec(
        url=f"https://github.com/{name}",
        license="MIT",
        language=SourceLanguage.PYTHON,
        created_at=date(2025, 1, 1),
        epochs=tuple(f"{i:040x}" for i in range(1, epochs + 1)),
    )


PARTITIONS = CorpusPartitions.from_manifest(
    Manifest(repos=(spec("owner/one", 2), spec("owner/two", 3)))
)


def test_partition_key_uses_slug_and_padded_epoch() -> None:
    assert partition_key("pallets/click", 7) == "pallets__click@07"


def test_one_partition_per_repository_epoch() -> None:
    assert PARTITIONS.definition.get_partition_keys() == [
        "owner__one@00",
        "owner__one@01",
        "owner__two@00",
        "owner__two@01",
        "owner__two@02",
    ]


def test_resolve_returns_the_repository_and_epoch() -> None:
    repo, epoch = PARTITIONS.resolve("owner__two@01")

    assert (repo.name, epoch) == ("owner/two", 1)


def test_resolve_rejects_unknown_keys() -> None:
    with pytest.raises(KeyError, match="unknown corpus partition 'owner__one@09'"):
        PARTITIONS.resolve("owner__one@09")


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"repo": "owner/one"}, ["owner__one@00", "owner__one@01"]),
        ({"epoch": 2}, ["owner__two@02"]),
        ({"repo": "owner/two", "epoch": 0}, ["owner__two@00"]),
        ({"repo": "owner/missing"}, []),
    ],
)
def test_keys_can_be_filtered(filters: dict[str, object], expected: list[str]) -> None:
    assert PARTITIONS.keys(**filters) == expected  # type: ignore[arg-type]
