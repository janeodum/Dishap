"""Self-contained diffusion imputer demo for tabular regression."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - torch is optional for the main package, but mandatory here.
    import torch
except ImportError as exc:  # pragma: no cover - provide a friendly error for users.
    raise ImportError("This demo requires PyTorch. Install it via 'pip install torch'.") from exc


@dataclass(frozen=True)
class ToyRegressionData:
    """Simple container bundling training and evaluation splits."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]


def build_correlated_regression_dataset(
    n_samples: int = 256, *, random_state: int = 0
) -> ToyRegressionData:
    """Create a tiny regression dataset with correlated features."""

    rng = np.random.default_rng(random_state)

    latent = rng.normal(size=n_samples)
    noise = rng.normal(scale=0.2, size=(n_samples, 3))

    X = np.column_stack(
        [
            latent + noise[:, 0],
            0.8 * latent + noise[:, 1],
            0.5 * latent - noise[:, 2],
        ]
    )
    feature_names = ("shared_1", "shared_2", "partial")

    weights = np.array([1.5, -2.0, 0.8])
    y = X @ weights + 0.5 * latent + rng.normal(scale=0.3, size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    return ToyRegressionData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )


class MockDiffusionBackbone:
    """Tiny torch-based sampler mimicking a conditional diffusion model."""

    def __init__(self, noise_scale: float = 0.05):
        self.noise_scale = float(noise_scale)
        self._background: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "MockDiffusionBackbone":
        # ``DiffusionImputer`` converts incoming arrays to tensors on the target device.
        if data.ndim != 2:
            raise ValueError("Expected a 2D tensor for the training data.")
        self._background = data.detach().clone()
        return self

    def sample_conditional(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self._background is None:
            raise RuntimeError("Call 'fit' before sampling conditionals.")

        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor with shape (batch, n_features).")
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D tensor matching the shape of x.")

        batch_x = x[0]
        batch_mask = mask[0].to(torch.bool)
        n_features = batch_x.shape[0]

        samples = batch_x.expand(n_samples, n_features).clone()

        if (~batch_mask).any():
            # Draw random rows from the background table for the missing entries.
            idx = torch.randint(
                low=0,
                high=self._background.shape[0],
                size=(n_samples,),
                generator=generator,
                device=self._background.device,
            )
            background_draws = self._background[idx][:, ~batch_mask]
            noise = torch.randn_like(background_draws, generator=generator) * self.noise_scale
            samples[:, ~batch_mask] = background_draws + noise

        return samples

    def eval(self) -> "MockDiffusionBackbone":
        # Provided for API compatibility; no-op for the mock backbone.
        return self


def train_regression_model(data: ToyRegressionData) -> LinearRegression:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.X_train)
    model = LinearRegression()
    model.fit(X_scaled, data.y_train)

    # Wrap prediction to include scaling at inference time.
    model.predict_with_preprocessing = lambda inputs: model.predict(scaler.transform(inputs))
    return model


def explain_with_diffusion_imputer(
    model: LinearRegression,
    background: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    backbone = MockDiffusionBackbone(noise_scale=0.1)
    generator = shap.maskers.DiffusionImputer(backbone, random_state=0)
    masker = shap.maskers.ConditionalGeneratorMasker(
        background,
        generator=generator,
        m_mc=64,
        random_state=0,
    )
    def predict_with_mask(inputs: np.ndarray, _mask=None) -> np.ndarray:
        return model.predict_with_preprocessing(inputs)

    explainer = shap.Explainer(predict_with_mask, masker, algorithm="permutation")
    explanation = explainer(samples, max_evals="auto", silent=True)
    return explanation.values


def main(n_samples: int = 3) -> None:
    data = build_correlated_regression_dataset()
    model = train_regression_model(data)

    background = data.X_train[:64]
    samples = data.X_test[:n_samples]

    shap_values = explain_with_diffusion_imputer(model, background, samples)

    print("Diffusion imputer SHAP values (regression):")
    for row, sample_values in enumerate(shap_values):
        formatted = ", ".join(f"{name}={value: .4f}" for name, value in zip(data.feature_names, sample_values))
        print(f"Sample {row}: {formatted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of test samples to explain (default: 3).",
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples)
