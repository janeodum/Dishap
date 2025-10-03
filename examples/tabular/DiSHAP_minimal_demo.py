"""Minimal end-to-end example comparing KernelSHAP with the DiSHAP masker.

This script intentionally keeps the dataset tiny and self-contained so that
users can quickly observe how Shapley attributions behave under correlated
features.  To run it from the repository root, execute::

    python examples/tabular/DiSHAP_minimal_demo.py

Two plots and the printed console output will highlight how the conditional
masker shares credit across correlated features for both regression and
classification models.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Callable, Iterable, Tuple

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # matplotlib is only needed for visualization, but provide a friendly error.
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard.
    raise ImportError(
        "matplotlib is required for plotting. Install it via 'pip install matplotlib'."
    ) from exc


@dataclasses.dataclass(frozen=True)
class ToyDataset:
    """Container bundling the synthetic dataset and feature names."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_reg_train: np.ndarray
    y_reg_test: np.ndarray
    y_clf_train: np.ndarray
    y_clf_test: np.ndarray
    feature_names: Tuple[str, ...]


def build_correlated_dataset(n_samples: int = 400, random_state: int = 0) -> ToyDataset:
    """Create a toy dataset where features are strongly correlated."""

    rng = np.random.default_rng(random_state)

    # Draw a latent factor and build correlated copies with small independent noise.
    latent = rng.normal(size=n_samples)
    noise = rng.normal(scale=0.2, size=(n_samples, 3))
    X = np.column_stack(
        [
            latent + noise[:, 0],
            latent + 0.8 * noise[:, 1],
            0.5 * latent - noise[:, 2],
        ]
    )
    feature_names = ("x1_shared", "x2_shared", "x3_partial")

    # Regression target: linear combination of the latent drivers.
    y_reg = 2.0 * latent + rng.normal(scale=0.3, size=n_samples)

    # Classification target: logistic response of correlated features.
    logits = 1.5 * latent + 0.5 * noise[:, 1] - 0.8 * noise[:, 2]
    prob = 1.0 / (1.0 + np.exp(-logits))
    y_clf = rng.binomial(1, prob)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=random_state
    )

    return ToyDataset(
        X_train=X_train,
        X_test=X_test,
        y_reg_train=y_reg_train,
        y_reg_test=y_reg_test,
        y_clf_train=y_clf_train,
        y_clf_test=y_clf_test,
        feature_names=feature_names,
    )


def fit_models(data: ToyDataset) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Train ridge and logistic models on the correlated dataset."""

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=0))
    ridge.fit(data.X_train, data.y_reg_train)

    log_reg = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0)
    )
    log_reg.fit(data.X_train, data.y_clf_train)

    return ridge, log_reg


def explain_with_kernel_shap(
    predict: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Run classic KernelSHAP with an independent background distribution."""

    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(samples)
    return np.array(shap_values)


def explain_with_conditional_masker(
    predict: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    background: np.ndarray,
    samples: np.ndarray,
    m_mc: int = 128,
) -> np.ndarray:
    """Run DiSHAP via the conditional generator masker using permutation SHAP."""

    masker = shap.maskers.ConditionalGeneratorMasker(background, m_mc=m_mc, random_state=0)
    explainer = shap.Explainer(predict, masker, algorithm="permutation")
    values = explainer(samples, max_evals="auto", silent=True).values
    return np.array(values)


def _print_and_plot(
    feature_names: Iterable[str],
    phi_kernel: np.ndarray,
    phi_conditional: np.ndarray,
    title: str,
) -> None:
    """Helper to show textual and graphical comparisons of the attributions."""

    feature_names = tuple(feature_names)
    print(f"\n=== {title} ===")
    for name, phi_k, phi_c in zip(feature_names, phi_kernel, phi_conditional):
        print(f"{name:>12s} | KernelSHAP={phi_k: .4f} | DiSHAP={phi_c: .4f}")

    y_pos = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y_pos - width / 2, phi_kernel, height=width, label="KernelSHAP")
    ax.barh(y_pos + width / 2, phi_conditional, height=width, label="DiSHAP")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Shapley value")
    ax.set_title(title)
    ax.legend()
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()


def main(show_plots: bool = True) -> None:
    data = build_correlated_dataset()
    ridge, log_reg = fit_models(data)

    # Use a manageable subset for background and evaluation to keep runtime low.
    background = data.X_train[:80]
    samples = data.X_test[:5]

    # Regression predictions for both approaches.
    ridge_predict = ridge.predict

    ridge_conditional_predict = lambda inputs, _mask=None: ridge.predict(inputs)

    phi_kernel_ridge = explain_with_kernel_shap(ridge_predict, background, samples)
    phi_cond_ridge = explain_with_conditional_masker(
        ridge_conditional_predict, background, samples, m_mc=128
    )

    _print_and_plot(
        data.feature_names,
        phi_kernel_ridge[0],
        phi_cond_ridge[0],
        "Ridge regression on correlated features",
    )

    # Classification: explain the probability of the positive class.
    def proba_positive(inputs: np.ndarray) -> np.ndarray:
        return log_reg.predict_proba(inputs)[:, 1]

    def proba_positive_masked(inputs: np.ndarray, _mask=None) -> np.ndarray:
        return proba_positive(inputs)

    phi_kernel_clf = explain_with_kernel_shap(proba_positive, background, samples)
    phi_cond_clf = explain_with_conditional_masker(
        proba_positive_masked, background, samples, m_mc=192
    )

    # KernelExplainer returns a 2D array for scalar outputs, but the permutation
    # explainer returns (n_samples, n_features, n_classes). Select class-1 column.
    if phi_cond_clf.ndim == 3:
        phi_cond_clf = phi_cond_clf[:, :, 1]

    _print_and_plot(
        data.feature_names,
        phi_kernel_clf[0],
        phi_cond_clf[0],
        "Logistic regression P(y=1) on correlated features",
    )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare KernelSHAP with DiSHAP maskers.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run the computations without showing matplotlib windows (useful for CI).",
    )
    args = parser.parse_args()
    main(show_plots=not args.no_show)
