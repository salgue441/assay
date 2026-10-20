import pytest

from assay.calibrate import Split, grouped_split

# 60 documents from 12 repositories of uneven size.
SIZES = [2, 3, 4, 5, 6, 7, 3, 4, 8, 5, 6, 7]
ITEMS = [f"repo{g}/doc{i}" for g, size in enumerate(SIZES) for i in range(size)]


def repo(item: str) -> str:
    return item.split("/", maxsplit=1)[0]


def split(seed: int, fraction: float = 0.25) -> Split[str]:
    return grouped_split(ITEMS, group_of=repo, test_fraction=fraction, seed=seed)


@pytest.mark.parametrize("seed", range(10))
def test_no_group_is_on_both_sides(seed: int) -> None:
    result = split(seed)

    assert {repo(i) for i in result.dev}.isdisjoint({repo(i) for i in result.test})
    assert sorted(result.dev + result.test) == sorted(ITEMS)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("fraction", [0.2, 0.25, 0.5])
def test_the_test_share_is_close_to_the_fraction(seed: int, fraction: float) -> None:
    share = len(split(seed, fraction).test) / len(ITEMS)

    assert share == pytest.approx(fraction, abs=max(SIZES) / len(ITEMS) / 2)


def test_the_split_is_deterministic_for_a_seed() -> None:
    assert split(4) == split(4)
    assert len({split(seed).test for seed in range(10)}) > 1


def test_items_keep_their_input_order() -> None:
    result = split(1)

    assert list(result.dev) == [i for i in ITEMS if i in result.dev]
    assert list(result.test) == [i for i in ITEMS if i in result.test]


def test_both_sides_are_never_empty() -> None:
    items = ["a"] + ["b"] * 10

    for seed in range(10):
        for fraction in (0.01, 0.99):
            result = grouped_split(items, group_of=str, test_fraction=fraction, seed=seed)
            assert result.dev
            assert result.test


@pytest.mark.parametrize(
    ("items", "fraction", "message"),
    [
        (["a", "a"], 0.5, "two groups"),
        (["a", "b"], 0.0, r"\(0, 1\)"),
        (["a", "b"], 1.0, r"\(0, 1\)"),
    ],
)
def test_invalid_inputs_raise(items: list[str], fraction: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        grouped_split(items, group_of=str, test_fraction=fraction, seed=0)
