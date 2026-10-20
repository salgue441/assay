import math

import numpy as np
import pytest

from assay.calibrate import calibrate
from assay.calibrate.baseline import (
    MODELS,
    BaselineModel,
    feature_matrix,
    fit_and_predict,
    make_model,
    out_of_fold,
    text_features,
)

from .test_study import PRIOR, ROUTING, corpus, run, texts

DOC = """## `parse`

Parse a value.

### Parameters
- `value`: The text to parse.
- `strict`: Fail on unknown keys.

### Returns
The parsed value.

### Example
```python
parse("1")
```
"""


def test_text_features_count_structure_and_sections() -> None:
    features = text_features(DOC)

    assert features["log_words"] == pytest.approx(math.log1p(len(DOC.split())))
    assert features["headings"] == 4
    assert features["code_blocks"] == 1
    assert features["code_spans"] == 3
    assert features["list_items"] == 2
    assert (features["has_parameters"], features["has_returns"]) == (1.0, 1.0)
    assert (features["has_raises"], features["has_example"]) == (0.0, 1.0)


def test_an_empty_document_has_zero_features() -> None:
    assert set(text_features("").values()) == {0.0}


def test_the_feature_matrix_leaves_missing_metrics_as_nan() -> None:
    documents = corpus()[:2]

    matrix, names = feature_matrix(documents, texts(documents), ["accuracy", "clarity"])

    assert matrix.shape == (2, len(names))
    assert names[:2] == ["accuracy", "clarity"]
    assert np.isnan(matrix[:, 1]).all()
    assert not np.isnan(matrix[:, 0]).any()


def learnable(n_groups: int = 4, per_group: int = 12) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    features = rng.uniform(size=(n_groups * per_group, 3))
    target = 1 + 4 * features[:, 0] + rng.normal(scale=0.1, size=len(features))
    groups = [f"g{i // per_group}" for i in range(len(features))]
    return features, target, groups


@pytest.mark.parametrize("model", MODELS)
def test_out_of_fold_predictions_recover_a_learnable_signal(model: BaselineModel) -> None:
    features, target, groups = learnable()

    predicted = out_of_fold(features, target, groups, model=model, seed=1)

    assert predicted.shape == target.shape
    assert np.corrcoef(predicted, target)[0, 1] > 0.8


@pytest.mark.parametrize("model", MODELS)
def test_models_tolerate_missing_values(model: BaselineModel) -> None:
    features, target, groups = learnable()
    features[::5, 1] = np.nan

    predicted = fit_and_predict(features, target, groups, features[:3], model=model, seed=1)

    assert np.isfinite(predicted).all()


def test_out_of_fold_needs_two_groups() -> None:
    features, target, _ = learnable()

    with pytest.raises(ValueError, match="at least two repositories, got 1"):
        out_of_fold(features, target, ["g"] * len(target), model="ridge", seed=1)


def test_one_group_trains_without_tuning() -> None:
    assert not hasattr(make_model("ridge", groups=1, seed=1), "param_grid")
    assert hasattr(make_model("ridge", groups=3, seed=1), "param_grid")


def test_unknown_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown baseline model"):
        make_model("forest", groups=2, seed=1)  # type: ignore[arg-type]


def test_the_report_compares_the_baseline_with_the_composite_and_the_judge() -> None:
    report = run(corpus(), include_test=True)

    for side in ("dev", "test"):
        assert {"composite", "judge_alone", "ridge"} <= set(report.baseline[side])
        # A model trained on a few documents may predict a constant; then the
        # report says why instead of showing a correlation.
        for model in MODELS:
            assert model in report.baseline[side] or any(
                note.startswith(f"{side} baseline {model}") for note in report.notes
            )
    assert report.baseline["dev"]["composite"].value == pytest.approx(1.0)


def test_documents_without_text_skip_the_models_but_not_the_comparison() -> None:
    report = calibrate(
        corpus(),
        prior=PRIOR,
        routing=ROUTING,
        development_fraction=0.5,
        shrink_to_prior=False,
        resamples=20,
        seed=7,
        include_test=False,
        texts={},
    )

    assert set(report.baseline["dev"]) == {"composite", "judge_alone"}
    assert any("have no stored text" in note for note in report.notes)
