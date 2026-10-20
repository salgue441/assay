import numpy as np
import pytest

from assay.calibrate import choose_shrinkage, fit_weights, grouped_folds

RNG = np.random.default_rng(42)
FEATURES = RNG.uniform(0.0, 1.0, size=(40, 3))
PRIOR = np.full(3, 1 / 3)
GROUPS = [f"repo{i // 5}" for i in range(40)]
CANDIDATES = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]


@pytest.mark.parametrize("weights", [[0.5, 0.3, 0.2], [0.7, 0.3, 0.0]])
def test_noiseless_data_recovers_the_true_weights(weights: list[float]) -> None:
    target = FEATURES @ np.array(weights)

    fitted = fit_weights(FEATURES, target, prior=PRIOR, shrinkage=0.0)

    np.testing.assert_allclose(fitted, weights, atol=1e-6)


def test_very_large_shrinkage_returns_the_prior() -> None:
    target = FEATURES @ np.array([0.8, 0.1, 0.1])

    fitted = fit_weights(FEATURES, target, prior=PRIOR, shrinkage=1e9)

    np.testing.assert_allclose(fitted, PRIOR, atol=1e-6)


def test_fitted_weights_lie_on_the_simplex() -> None:
    target = RNG.uniform(1.0, 5.0, size=40)

    fitted = fit_weights(FEATURES, target, prior=PRIOR, shrinkage=0.5)

    assert np.all(fitted >= 0)
    assert fitted.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("features", "target", "prior", "shrinkage", "message"),
    [
        (FEATURES[:, 0], FEATURES[:, 0], PRIOR, 0.0, "expected features"),
        (FEATURES, FEATURES[:-1, 0], PRIOR, 0.0, "do not match"),
        (FEATURES, FEATURES[:, 0], np.array([0.5, 0.5]), 0.0, "do not match"),
        (FEATURES, FEATURES[:, 0], np.array([0.5, 0.5, 0.5]), 0.0, "sum to 1"),
        (FEATURES, FEATURES[:, 0], np.array([1.5, -0.5, 0.0]), 0.0, "non-negative"),
        (FEATURES, FEATURES[:, 0], PRIOR, -1.0, "shrinkage"),
        (FEATURES, FEATURES[:, 0], PRIOR, float("inf"), "shrinkage"),
    ],
)
def test_invalid_fit_inputs_raise(
    features: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    shrinkage: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_weights(features, target, prior=prior, shrinkage=shrinkage)


def test_folds_keep_groups_whole_and_cover_every_row() -> None:
    folds = grouped_folds(GROUPS, folds=3, seed=1)

    assert sorted(np.concatenate(folds).tolist()) == list(range(40))
    fold_of = {GROUPS[i]: k for k, rows in enumerate(folds) for i in rows}
    assert all(fold_of[GROUPS[i]] == k for k, rows in enumerate(folds) for i in rows)
    assert sorted(len(rows) for rows in folds) == [10, 15, 15]


@pytest.mark.parametrize("folds", [1, 9])
def test_fold_count_must_fit_the_groups(folds: int) -> None:
    with pytest.raises(ValueError, match="folds"):
        grouped_folds(GROUPS, folds=folds, seed=0)


def test_clean_data_selects_a_small_shrinkage() -> None:
    target = FEATURES @ np.array([0.6, 0.3, 0.1])

    chosen = choose_shrinkage(
        FEATURES, target, groups=GROUPS, prior=PRIOR, candidates=CANDIDATES, seed=0
    )

    assert chosen <= 0.01


def test_the_choice_is_deterministic() -> None:
    target = FEATURES @ np.array([0.6, 0.3, 0.1]) + RNG.normal(0.0, 0.2, size=40)

    def run() -> float:
        return choose_shrinkage(
            FEATURES, target, groups=GROUPS, prior=PRIOR, candidates=CANDIDATES, seed=3
        )

    assert run() == run()


def test_ties_go_to_the_larger_shrinkage() -> None:
    target = FEATURES @ PRIOR

    chosen = choose_shrinkage(
        FEATURES, target, groups=GROUPS, prior=PRIOR, candidates=CANDIDATES, folds=4, seed=0
    )

    assert chosen == 100.0


@pytest.mark.parametrize(
    ("groups", "candidates", "message"),
    [(GROUPS[:-1], CANDIDATES, "groups has"), (GROUPS, [], "at least one candidate")],
)
def test_invalid_choice_inputs_raise(
    groups: list[str], candidates: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        choose_shrinkage(
            FEATURES, FEATURES[:, 0], groups=groups, prior=PRIOR, candidates=candidates, seed=0
        )
