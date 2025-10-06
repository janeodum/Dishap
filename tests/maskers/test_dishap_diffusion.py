import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import shap


def _load_dishap_module():
    root = Path(__file__).resolve().parents[2]
    shap_path = root / "shap"

    shap_pkg = types.ModuleType("shap")
    shap_pkg.__path__ = [str(shap_path)]
    shap_pkg.__file__ = str(shap_path / "__init__.py")
    shap_pkg.__package__ = "shap"

    maskers_pkg = types.ModuleType("shap.maskers")
    maskers_pkg.__path__ = [str(shap_path / "maskers")]
    maskers_pkg.__package__ = "shap.maskers"

    sys.modules.setdefault("shap", shap_pkg)
    sys.modules.setdefault("shap.maskers", maskers_pkg)

    module_name = "shap.maskers._dishap"
    module_file = shap_path / "maskers" / "_dishap.py"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


try:  # pragma: no cover - executed when compiled extensions are available
    from shap.maskers import _dishap  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback when shap cannot be imported directly
    _dishap = _load_dishap_module()


class FakeTensor:
    """Minimal tensor implementation compatible with DiffusionImputer logic."""

    def __init__(self, array):
        self.array = np.array(array)

    # ------------------------------------------------------------------
    # Array-like interface
    # ------------------------------------------------------------------
    def __array__(self, dtype=None):
        if dtype is not None:
            return np.asarray(self.array, dtype=dtype)
        return np.asarray(self.array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    # ------------------------------------------------------------------
    # Torch-style methods used by DiffusionImputer
    # ------------------------------------------------------------------
    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.array, axis=dim))

    def squeeze(self, dim=None):
        if dim is None:
            return FakeTensor(np.squeeze(self.array))
        return FakeTensor(np.squeeze(self.array, axis=dim))

    def repeat(self, *reps):
        if len(reps) == 1 and isinstance(reps[0], (tuple, list)):
            reps = tuple(reps[0])
        return FakeTensor(np.tile(self.array, reps))

    def detach(self):
        return self

    def to(self, device=None):
        return self

    def numpy(self):
        return np.array(self.array, copy=True)

    def reshape(self, *shape):
        return FakeTensor(self.array.reshape(*shape))


class FakeGenerator:
    """Simple replacement for ``torch.Generator``."""

    def __init__(self, device=None):
        self.device = device
        self._state = 0

    def manual_seed(self, seed):
        self._state = int(seed)
        return self


def build_fake_torch():
    fake_torch = types.SimpleNamespace()

    def as_tensor(data, dtype=None, device=None):
        array = np.array(data, dtype=dtype if dtype is not None else None)
        return FakeTensor(array)

    fake_torch.as_tensor = as_tensor
    fake_torch.device = lambda value=None: value
    fake_torch.float32 = np.float32
    fake_torch.float64 = np.float64
    fake_torch.bool = np.bool_
    fake_torch.Generator = FakeGenerator
    return fake_torch


@pytest.fixture(autouse=True)
def fake_torch_module(monkeypatch):
    """Patch the local torch reference used by DiffusionImputer."""

    fake_torch = build_fake_torch()
    monkeypatch.setattr(_dishap, "torch", fake_torch)
    yield fake_torch


class MockDiffusionBackbone:
    """Deterministic backbone that mimics diffusion sampling behaviour."""

    def __init__(self):
        self.fit_called_with = None

    def fit(self, data):
        if isinstance(data, FakeTensor):
            data = data.numpy()
        self.fit_called_with = np.array(data)
        return self

    def sample_conditional(self, x, mask, n_samples, generator=None):
        if isinstance(x, FakeTensor):
            x_array = x.numpy()
        else:
            x_array = np.array(x)
        if x_array.ndim == 2 and x_array.shape[0] == 1:
            x_array = x_array[0]

        if isinstance(mask, FakeTensor):
            mask_array = mask.numpy()
        else:
            mask_array = np.array(mask)
        mask_array = mask_array.reshape(-1).astype(bool)

        seed = 0 if generator is None else getattr(generator, "_state", 0)
        rng = np.random.RandomState(seed)
        baseline = np.tile(x_array, (int(n_samples), 1))
        noise = rng.standard_normal(size=baseline.shape)
        baseline[:, ~mask_array] = noise[:, ~mask_array]
        return baseline


def build_correlated_dataset(n_samples=240, random_state=0):
    rng = np.random.RandomState(random_state)
    base = rng.normal(size=n_samples)
    correlated_1 = base + 0.01 * rng.normal(size=n_samples)
    correlated_2 = base + 0.01 * rng.normal(size=n_samples)
    independent = rng.normal(size=n_samples)
    X = np.column_stack([correlated_1, correlated_2, independent])

    regression_weights = np.array([1.2, -0.8, 0.5])
    regression_targets = X @ regression_weights

    classification_weights = np.array([1.5, 1.5, -0.75])
    logits = X @ classification_weights
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    return X, regression_targets, probabilities


def compute_expected_draws(x, mask, n_samples, seed_state):
    rng = np.random.RandomState(seed_state)
    expected = np.tile(x, (n_samples, 1))
    draws = rng.standard_normal(size=expected.shape)
    expected[:, ~mask] = draws[:, ~mask]
    return expected


def derive_initial_seed(random_state_seed):
    state_rng = np.random.RandomState(random_state_seed)
    return int(state_rng.randint(np.iinfo(np.int32).max))


def test_diffusion_imputer_respects_shape_and_conditioning():
    background = np.arange(18, dtype=float).reshape(9, 2)
    x = background[0]
    mask = np.array([True, False])
    n_samples = 4

    backbone = MockDiffusionBackbone()
    imputer = _dishap.DiffusionImputer(backbone, ensure_observed=True, random_state=0)
    imputer.fit(background)

    samples = imputer.sample(x, mask, n_samples)

    assert samples.shape == (n_samples, background.shape[1])
    assert np.all(samples[:, mask] == x[mask])

    seed = derive_initial_seed(0)
    expected = compute_expected_draws(x, mask, n_samples, seed)
    assert np.allclose(samples, expected)


def test_diffusion_imputer_integrates_with_masker():
    background = np.linspace(0, 1, 12, dtype=float).reshape(6, 2)
    x = background[1]
    mask = np.array([False, True])

    backbone = MockDiffusionBackbone()
    generator = _dishap.DiffusionImputer(backbone, ensure_observed=False, random_state=7)
    masker = _dishap.ConditionalGeneratorMasker(background, generator=generator, m_mc=3, random_state=11)

    samples, masks = masker(mask, x)

    assert samples.shape == (masker.m_mc, background.shape[1])
    assert masks.shape == samples.shape
    assert np.all(masks == mask)

    seed = derive_initial_seed(11)
    expected = compute_expected_draws(x, mask, masker.m_mc, seed)
    assert np.allclose(samples, expected)


def test_diffusion_imputer_rng_reproducibility():
    background = np.random.RandomState(123).randn(8, 3)
    x = background[2]
    mask = np.array([True, False, False])

    backbone_a = MockDiffusionBackbone()
    backbone_b = MockDiffusionBackbone()

    imputer_a = _dishap.DiffusionImputer(backbone_a, random_state=42)
    imputer_b = _dishap.DiffusionImputer(backbone_b, random_state=42)

    imputer_a.fit(background)
    imputer_b.fit(background)

    samples_a = imputer_a.sample(x, mask, 5)
    samples_b = imputer_b.sample(x, mask, 5)

    assert np.allclose(samples_a, samples_b)


def test_diffusion_masker_kernel_shap_regression_behaviour():
    X, _, _ = build_correlated_dataset()
    background = X[:80]
    samples = X[80:83]

    def predict(inputs: np.ndarray, _mask: np.ndarray | None = None) -> np.ndarray:
        return inputs @ np.array([1.2, -0.8, 0.5])

    kernel_explainer = shap.KernelExplainer(predict, background, nsamples=64)
    phi_kernel = np.asarray(kernel_explainer.shap_values(samples))

    backbone = MockDiffusionBackbone()
    generator = _dishap.DiffusionImputer(backbone, ensure_observed=True, random_state=0)
    masker = _dishap.ConditionalGeneratorMasker(background, generator=generator, m_mc=64, random_state=0)

    np.random.seed(0)
    explainer = shap.Explainer(predict, masker, algorithm="permutation")
    np.random.seed(0)
    values = explainer(samples, max_evals=128, silent=True).values

    assert values.shape == phi_kernel.shape
    assert values.ndim == 2

    kernel_imbalance = float(np.abs(phi_kernel[:, 0] - phi_kernel[:, 1]).mean())
    conditional_imbalance = float(np.abs(values[:, 0] - values[:, 1]).mean())
    assert not np.isclose(conditional_imbalance, kernel_imbalance)


def test_diffusion_masker_supports_probability_models():
    X, _, _ = build_correlated_dataset(random_state=1)
    background = X[:80]
    samples = X[80:84]

    def proba_positive(inputs: np.ndarray, _mask: np.ndarray | None = None) -> np.ndarray:
        logits = inputs @ np.array([1.5, 1.5, -0.75])
        return 1.0 / (1.0 + np.exp(-logits))

    kernel_explainer = shap.KernelExplainer(proba_positive, background, nsamples=64)
    phi_kernel = np.asarray(kernel_explainer.shap_values(samples))

    backbone = MockDiffusionBackbone()
    generator = _dishap.DiffusionImputer(backbone, ensure_observed=True, random_state=7)
    masker = _dishap.ConditionalGeneratorMasker(background, generator=generator, m_mc=96, random_state=11)

    np.random.seed(0)
    explainer = shap.Explainer(proba_positive, masker, algorithm="permutation")
    np.random.seed(0)
    values = explainer(samples, max_evals=160, silent=True).values

    assert values.shape == phi_kernel.shape
    assert values.ndim == 2

    kernel_imbalance = float(np.abs(phi_kernel[:, 0] - phi_kernel[:, 1]).mean())
    conditional_imbalance = float(np.abs(values[:, 0] - values[:, 1]).mean())
    assert not np.isclose(conditional_imbalance, kernel_imbalance)

    kernel_mean = float(phi_kernel.mean())
    conditional_mean = float(values.mean())
    assert np.isfinite(conditional_mean)
    assert not np.isclose(kernel_mean, conditional_mean)
