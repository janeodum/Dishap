"""Tests for the DI-SHAP conditional generator masker."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import shap
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression


def _fit_linear_predictor(X: np.ndarray, y: np.ndarray):
    """Fit a linear model and return a callable that accepts masked inputs."""

    model = LinearRegression().fit(X, y)

    def predict(data: np.ndarray, _mask: np.ndarray | None = None) -> np.ndarray:
        return model.predict(data)

    return predict


def test_dishap_masker_output_shape_and_determinism():
    """The masker should return deterministic samples with consistent shapes."""

    X, _ = make_regression(n_samples=40, n_features=4, random_state=0)
    masker_a = shap.maskers.ConditionalGeneratorMasker(X, m_mc=32, random_state=0)
    masker_b = shap.maskers.ConditionalGeneratorMasker(X, m_mc=32, random_state=0)

    mask = np.array([1, 0, 1, 0], dtype=bool)
    x = X[0]

    samples_a, masks_a = masker_a(mask, x)
    samples_b, masks_b = masker_b(mask, x)

    assert samples_a.shape == (32, X.shape[1])
    assert masks_a.shape == (32, X.shape[1])

    assert_allclose(samples_a, samples_b)
    assert_array_equal(masks_a, masks_b)

    assert_allclose(samples_a[:, mask], x[mask])


def test_dishap_masker_matches_independent_on_uncorrelated_data():
    """Conditional sampling should agree with the independent masker on IID data."""

    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    weights = np.array([1.5, -2.0, 0.5])
    y = X @ weights

    predict = _fit_linear_predictor(X, y)

    independent_masker = shap.maskers.Independent(X, max_samples=200)
    conditional_masker = shap.maskers.ConditionalGeneratorMasker(X, m_mc=64, random_state=0)

    np.random.seed(0)
    explainer_ind = shap.Explainer(predict, independent_masker, algorithm="permutation")
    np.random.seed(0)
    explainer_cond = shap.Explainer(predict, conditional_masker, algorithm="permutation")

    np.random.seed(0)
    values_ind = explainer_ind(X[:3], max_evals=40, silent=True).values
    np.random.seed(0)
    values_cond = explainer_cond(X[:3], max_evals=40, silent=True).values

    mean_abs_diff = np.abs(values_ind - values_cond).mean()
    assert mean_abs_diff < 0.2


def test_dishap_masker_shares_attributions_for_correlated_features():
    """Conditional sampling should better share credit across correlated inputs."""

    rng = np.random.RandomState(0)
    base = rng.normal(size=200)
    X = np.column_stack(
        [
            base + 0.01 * rng.normal(size=200),
            base + 0.01 * rng.normal(size=200),
        ]
    )
    y = X.sum(axis=1)

    predict = _fit_linear_predictor(X, y)

    independent_masker = shap.maskers.Independent(X, max_samples=200)
    conditional_masker = shap.maskers.ConditionalGeneratorMasker(X, m_mc=64, random_state=0)

    np.random.seed(0)
    explainer_ind = shap.Explainer(predict, independent_masker, algorithm="permutation")
    np.random.seed(0)
    explainer_cond = shap.Explainer(predict, conditional_masker, algorithm="permutation")

    x = X[:1]
    np.random.seed(0)
    phi_ind = explainer_ind(x, max_evals=60, silent=True).values[0]
    np.random.seed(0)
    phi_cond = explainer_cond(x, max_evals=60, silent=True).values[0]

    imbalance_ind = float(np.abs(phi_ind[0] - phi_ind[1]))
    imbalance_cond = float(np.abs(phi_cond[0] - phi_cond[1]))
    assert imbalance_cond < imbalance_ind


def test_dishap_masker_predict_proba_compatibility():
    """Permutation explanations over predict_proba should work with the masker."""

    X, y = make_classification(
        n_samples=180,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )

    model = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)

    def predict_proba(data: np.ndarray, _mask: np.ndarray | None = None) -> np.ndarray:
        return model.predict_proba(data)

    masker = shap.maskers.ConditionalGeneratorMasker(X, m_mc=48, random_state=0)

    np.random.seed(0)
    explainer = shap.Explainer(predict_proba, masker, algorithm="permutation")
    np.random.seed(0)
    values = explainer(X[:4], max_evals=32, silent=True).values

    assert values.shape == (4, X.shape[1], model.classes_.size)
