"""
The learned baseline: models that predict the human overall rating from the
static metrics and cheap text features, with no LLM call, to measure how
much of the judge's agreement with the raters they recover.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from assay.calibrate.study import RatedDocument

BaselineModel = Literal["ridge", "hist_gradient_boosting"]
MODELS: tuple[BaselineModel, ...] = ("ridge", "hist_gradient_boosting")
MAX_FOLDS = 5

_SECTIONS = {
    "has_parameters": r"param|argument|\bargs\b",
    "has_returns": r"return",
    "has_raises": r"raise|exception|error",
    "has_example": r"example|usage",
}
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.MULTILINE)
_CODE_BLOCK = re.compile(r"^\s*(```|~~~)", re.MULTILINE)
_CODE_SPAN = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)


def text_features(text: str) -> dict[str, float]:
    """Cheap features of a Markdown document."""

    headings = [h.lower() for h in _HEADING.findall(text)]
    features = {
        "log_words": math.log1p(len(text.split())),
        "headings": float(len(headings)),
        "code_blocks": len(_CODE_BLOCK.findall(text)) / 2,
        "code_spans": float(len(_CODE_SPAN.findall(text))),
        "list_items": float(len(_LIST_ITEM.findall(text))),
    }
    for name, pattern in _SECTIONS.items():
        features[name] = float(any(re.search(pattern, h) for h in headings))
    return features


def feature_matrix(
    documents: Sequence[RatedDocument], texts: Mapping[str, str], metrics: Sequence[str]
) -> tuple[NDArray[np.float64], list[str]]:
    """
    One row per document: each of `metrics` (NaN when the document has no
    value for it), then the text features of its text.
    """

    rows = []
    names: list[str] = []
    for document in documents:
        text = text_features(texts[document.document_id])
        names = [*metrics, *text]
        rows.append([document.components.get(m, math.nan) for m in metrics] + list(text.values()))
    return np.array(rows, dtype=np.float64).reshape(len(documents), -1), names


def make_model(name: BaselineModel, *, groups: int, seed: int) -> RegressorMixin:
    """
    The model, tuned by cross-validation grouped by repository when there
    are at least two `groups`; otherwise with fixed, moderate settings.
    """

    estimator, grid = _estimator(name, seed)
    if groups < 2:
        return estimator
    search: RegressorMixin = GridSearchCV(
        estimator,
        grid,
        cv=GroupKFold(n_splits=min(MAX_FOLDS, groups)),
        scoring="neg_mean_squared_error",
    )
    return search


def out_of_fold(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    groups: Sequence[str],
    *,
    model: BaselineModel,
    seed: int,
) -> NDArray[np.float64]:
    """
    Predictions for each row from a model fitted (and tuned) without that
    row's group, so in-sample fit never inflates development results.
    Raises `ValueError` with fewer than two groups.
    """

    labels = np.asarray(groups)
    names = sorted(set(groups))
    if len(names) < 2:
        msg = f"out-of-fold predictions need at least two repositories, got {len(names)}"
        raise ValueError(msg)
    predictions = np.full(len(target), np.nan)
    for held_out in names:
        test = labels == held_out
        train_groups = labels[~test]
        estimator = make_model(model, groups=len(set(train_groups)), seed=seed)
        _fit(estimator, features[~test], target[~test], train_groups)
        predictions[test] = estimator.predict(features[test])
    return predictions


def fit_and_predict(
    train_features: NDArray[np.float64],
    train_target: NDArray[np.float64],
    train_groups: Sequence[str],
    features: NDArray[np.float64],
    *,
    model: BaselineModel,
    seed: int,
) -> NDArray[np.float64]:
    """Fit (and tune) on the training rows, then predict `features`."""

    estimator = make_model(model, groups=len(set(train_groups)), seed=seed)
    _fit(estimator, train_features, train_target, np.asarray(train_groups))
    predicted: NDArray[np.float64] = estimator.predict(features)
    return predicted


def _fit(
    estimator: RegressorMixin,
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    groups: NDArray[np.str_],
) -> None:
    if isinstance(estimator, GridSearchCV):
        estimator.fit(features, target, groups=groups)
    else:
        estimator.fit(features, target)


def _estimator(name: BaselineModel, seed: int) -> tuple[object, dict[str, list[object]]]:
    builders: dict[BaselineModel, Callable[[], tuple[object, dict[str, list[object]]]]] = {
        "ridge": lambda: (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
            {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "hist_gradient_boosting": lambda: (
            HistGradientBoostingRegressor(
                max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=5, random_state=seed
            ),
            {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        ),
    }
    if name not in builders:
        msg = f"unknown baseline model: {name!r}"
        raise ValueError(msg)
    return builders[name]()
