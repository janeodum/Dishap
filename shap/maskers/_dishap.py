"""Maskers powered by conditional generators for DI-SHAP style explainers.

This module introduces a small abstraction for conditional data generators and a
masker that relies on them to generate background samples conditioned on the
observed features of the instance under explanation.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state

from ._masker import Masker


try:  # pragma: no cover - torch is an optional dependency for diffusion models
    import torch
except ImportError:  # pragma: no cover - gracefully handle environments without torch
    torch = None

if TYPE_CHECKING:  # pragma: no cover - used only for static typing
    from torch import Generator as TorchGenerator, Tensor as TorchTensor


class BaseConditionalGenerator:
    """Base class for conditional data generators used by the DI-SHAP masker.

    Sub-classes are expected to implement :meth:`_fit` and :meth:`_sample`
    which respectively fit the underlying statistical model and draw
    conditional samples from it.
    """

    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        self.set_random_state(random_state)
        self._fitted = False
        self._n_features: Optional[int] = None
        self._dtype: Optional[np.dtype] = None

    def set_random_state(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        """Set the random state used for sampling.

        Parameters
        ----------
        random_state : int, numpy.random.RandomState, numpy.random.Generator, or None
            Seed or random state controlling reproducibility. The behaviour
            mirrors :func:`sklearn.utils.check_random_state`.
        """

        if isinstance(random_state, np.random.Generator):
            # QuantileTransformer expects an int or RandomState, so convert the
            # generator into a RandomState seeded by the generator.
            random_state = np.random.RandomState(random_state.integers(np.iinfo(np.int32).max))

        self.random_state = check_random_state(random_state)
        return self

    def fit(self, data: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]]):
        """Fit the generator on a background dataset."""

        data_array = self._validate_data(data)
        self._n_features = data_array.shape[1]
        self._dtype = data_array.dtype
        self._fit(data_array)
        self._fitted = True
        return self

    def sample(self, x: Sequence[float], mask: Iterable[Union[int, bool]], n_samples: int) -> np.ndarray:
        """Sample conditional draws for ``x`` under the provided ``mask``."""

        self._check_is_fitted()

        x_arr = np.asarray(x, dtype=float)
        mask_arr = np.asarray(mask, dtype=bool)

        if x_arr.ndim != 1:
            raise ValueError("x must be a one-dimensional array-like object.")
        if mask_arr.ndim != 1:
            raise ValueError("mask must be a one-dimensional array-like object.")
        if self._n_features is None:
            raise RuntimeError("Generator has not been fitted. Call 'fit' first.")
        if x_arr.shape[0] != self._n_features or mask_arr.shape[0] != self._n_features:
            raise ValueError("x and mask must match the number of features in the fitted data.")
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        samples = self._sample(x_arr, mask_arr, int(n_samples))
        if samples.shape != (n_samples, self._n_features):
            raise ValueError("Generator returned samples with an unexpected shape.")

        # Preserve the dtype of the training data when possible.
        if self._dtype is not None:
            samples = samples.astype(self._dtype, copy=False)

        return samples

    def _fit(self, data: np.ndarray):
        raise NotImplementedError

    def _sample(self, x: np.ndarray, mask: np.ndarray, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def _validate_data(self, data: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data, dtype=float)

        if data_array.ndim != 2:
            raise ValueError("Background data must be a 2D array-like object.")
        if data_array.shape[0] < 2:
            raise ValueError("Background data must contain at least two samples to fit a copula.")

        return data_array

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("The generator is not fitted yet. Call 'fit' with appropriate data before sampling.")


class DiffusionImputer(BaseConditionalGenerator):
    """Conditional generator that delegates to a diffusion style imputer.

    Parameters
    ----------
    backbone : object
        Object implementing a :meth:`sample_conditional` method with signature
        ``(x, mask, n_samples, generator=None)`` and optionally ``fit``.
    device : str or torch.device, optional
        Device on which the underlying backbone should run. Defaults to CPU
        when not provided.
    torch_dtype : torch.dtype or str, optional
        Target dtype for tensors passed to the backbone. If ``None`` a
        floating point dtype (``torch.float32``) is used.
    ensure_observed : bool, default True
        If ``True``, enforce that observed features in sampled batches match
        the conditioning vector ``x`` exactly. This mirrors the behaviour of
        standard DI-SHAP generators.
    """

    def __init__(
        self,
        backbone: Any,
        *,
        device: Optional[Any] = None,
        torch_dtype: Optional[Any] = None,
        ensure_observed: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if torch is None:  # pragma: no cover - executed only when torch is missing
            raise ImportError(
                "DiffusionImputer requires the optional dependency 'torch'. "
                "Install PyTorch to enable diffusion-based maskers."
            )
        super().__init__(random_state=random_state)
        self.backbone = backbone
        self.device = device
        self.torch_dtype = torch_dtype
        self.ensure_observed = bool(ensure_observed)

        self._torch_device: Optional[torch.device] = None
        self._torch_dtype: Optional[torch.dtype] = None
        self._torch_generator: Optional["TorchGenerator"] = None
        self._backbone_fitted: bool = False
        self._eval_called: bool = False

    # ------------------------------------------------------------------
    # BaseConditionalGenerator hooks
    # ------------------------------------------------------------------
    def _fit(self, data: np.ndarray):
        self._torch_device = self._resolve_device()
        self._torch_dtype = self._resolve_dtype(data)
        tensor_data = self._to_model_space(data)

        if hasattr(self.backbone, "fit") and callable(self.backbone.fit):
            try:
                self.backbone.fit(tensor_data)
            except TypeError:
                try:
                    # Some backbones expect numpy data. Fall back to the raw array.
                    self.backbone.fit(data)
                except TypeError:
                    self.backbone.fit()

        self._backbone_fitted = True

    def _sample(self, x: np.ndarray, mask: np.ndarray, n_samples: int) -> np.ndarray:
        self._ensure_torch_available()
        self._check_backbone_ready()

        x_tensor = self._prepare_example_tensor(x)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self._torch_device)
        mask_tensor = mask_tensor.unsqueeze(0)

        torch_generator = self._get_torch_generator()
        sampler = getattr(self.backbone, "sample_conditional")
        try:
            samples_tensor = sampler(
                x_tensor,
                mask_tensor,
                int(n_samples),
                generator=torch_generator,
            )
        except TypeError:
            samples_tensor = sampler(
                x_tensor,
                mask_tensor,
                int(n_samples),
            )
        samples_tensor = self._ensure_tensor_output(samples_tensor, n_samples)
        samples_array = self._from_model_space(samples_tensor)

        if self.ensure_observed:
            observed_idx = np.where(mask)[0]
            if observed_idx.size:
                samples_array[:, observed_idx] = x[observed_idx]

        return samples_array

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_torch_available(self):
        if torch is None:  # pragma: no cover - safeguard for type checkers
            raise ImportError("DiffusionImputer requires PyTorch to operate.")

    def _resolve_device(self) -> "torch.device":
        self._ensure_torch_available()
        if self.device is None:
            return torch.device("cpu")
        return torch.device(self.device)

    def _resolve_dtype(self, data: np.ndarray) -> "torch.dtype":
        self._ensure_torch_available()
        if self.torch_dtype is not None:
            if isinstance(self.torch_dtype, str):
                try:
                    return getattr(torch, self.torch_dtype)
                except AttributeError as exc:  # pragma: no cover - invalid dtype provided
                    raise ValueError(f"Unknown torch dtype '{self.torch_dtype}'.") from exc
            return self.torch_dtype

        if data.dtype == np.float64:
            return torch.float64
        return torch.float32

    def _to_model_space(self, array: Union[np.ndarray, Sequence[float]]) -> "TorchTensor":
        tensor = torch.as_tensor(array, dtype=self._torch_dtype, device=self._torch_device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _from_model_space(self, tensor: "TorchTensor") -> np.ndarray:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        tensor = tensor.detach().to("cpu")
        array = tensor.numpy()
        return array.astype(self._dtype, copy=False) if self._dtype is not None else array

    def _prepare_example_tensor(self, x: np.ndarray) -> "TorchTensor":
        tensor = torch.as_tensor(x, dtype=self._torch_dtype, device=self._torch_device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _ensure_tensor_output(self, output: Any, n_samples: int) -> "TorchTensor":
        if isinstance(output, (list, tuple)):
            if not output:  # pragma: no cover - defensive
                raise ValueError("Backbone returned an empty output sequence.")
            output = output[0]

        tensor = torch.as_tensor(output, dtype=self._torch_dtype, device=self._torch_device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)

        if tensor.shape[0] != n_samples:
            # Broadcast a single sample if provided, otherwise raise.
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(n_samples, 1)
            else:
                raise ValueError(
                    "Backbone returned samples with an unexpected leading dimension. "
                    f"Expected {n_samples}, received {tensor.shape[0]}."
                )
        if tensor.shape[1] != int(self._n_features or tensor.shape[1]):
            raise ValueError(
                "Backbone returned samples with incompatible feature dimension. "
                f"Expected {self._n_features}, received {tensor.shape[1]}."
            )
        return tensor

    def _get_torch_generator(self) -> "TorchGenerator":
        self._ensure_torch_available()
        if self._torch_generator is None:
            self._torch_generator = torch.Generator(device=self._torch_device)
        seed = int(self.random_state.randint(np.iinfo(np.int32).max))
        self._torch_generator.manual_seed(seed)
        return self._torch_generator

    def _check_backbone_ready(self):
        if hasattr(self.backbone, "eval") and not self._eval_called:
            # If the model differentiates between train/eval we assume eval mode.
            try:
                self.backbone.eval()
            except Exception:  # pragma: no cover - some models may not support eval
                pass
            self._eval_called = True


class GenericTorchDenoiserAdapter:
    """Adapter that exposes a uniform ``sample_conditional`` interface."""

    def __init__(
        self,
        model: Any,
        *,
        sampler: Optional[Callable[..., Any]] = None,
        fitter: Optional[Callable[[Any, Any], Any]] = None,
    ):
        self.model = model
        self._sampler = sampler
        self._fitter = fitter

    def fit(self, data: Any):
        if self._fitter is not None:
            self._fitter(self.model, data)
            return self
        if hasattr(self.model, "fit") and callable(self.model.fit):
            self.model.fit(data)
        return self

    def sample_conditional(
        self,
        x: Any,
        mask: Any,
        n_samples: int,
        *,
        generator: Optional[Any] = None,
    ) -> Any:
        sampler = self._resolve_sampler()
        signature = inspect.signature(sampler)
        kwargs: dict[str, Any] = {}
        args: list[Any] = []

        param_names = list(signature.parameters)
        if "x" in signature.parameters:
            kwargs["x"] = x
        elif param_names:
            args.append(x)

        if "mask" in signature.parameters:
            kwargs["mask"] = mask
        elif len(args) < len(param_names):
            args.append(mask)

        if "n_samples" in signature.parameters:
            kwargs["n_samples"] = n_samples
        elif len(args) < len(param_names):
            args.append(n_samples)

        if generator is not None:
            if "generator" in signature.parameters:
                kwargs["generator"] = generator
            elif "torch_generator" in signature.parameters:
                kwargs["torch_generator"] = generator

        return sampler(*args, **kwargs)

    def _resolve_sampler(self) -> Callable[..., Any]:
        if self._sampler is not None:
            return self._sampler

        if hasattr(self.model, "sample_conditional") and callable(self.model.sample_conditional):
            return self.model.sample_conditional
        if hasattr(self.model, "sample") and callable(self.model.sample):
            return self.model.sample
        raise AttributeError("The wrapped model does not expose a sampling routine.")


class TabDDPMAdapter(GenericTorchDenoiserAdapter):
    """Adapter for TabDDPM style conditional diffusion models."""

    def sample_conditional(
        self,
        x: Any,
        mask: Any,
        n_samples: int,
        *,
        generator: Optional[Any] = None,
    ) -> Any:
        if torch is not None:
            x = torch.as_tensor(x)
            mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
        return super().sample_conditional(x, mask, n_samples, generator=generator)


class GaussianCopulaGenerator(BaseConditionalGenerator):
    """Gaussian copula based conditional generator.

    The model follows a two-step approach: first fit a Gaussian copula using a
    quantile transformation, and then use the conditional multivariate normal
    distribution to draw samples in copula space before mapping them back to the
    original feature space.
    """

    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None, jitter: float = 1e-6):
        super().__init__(random_state=random_state)
        self.jitter = float(jitter)
        self._transformer: Optional[QuantileTransformer] = None
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None

    def _fit(self, data: np.ndarray):
        n_quantiles = min(1000, max(10, data.shape[0]))
        self._transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=int(data.shape[0]),
            random_state=self.random_state,
            copy=True,
        )
        gaussian_data = self._transformer.fit_transform(data)
        self._mean = gaussian_data.mean(axis=0)
        cov = np.cov(gaussian_data, rowvar=False)
        cov = self._ensure_positive_definite(cov)
        self._cov = cov

    def _sample(self, x: np.ndarray, mask: np.ndarray, n_samples: int) -> np.ndarray:
        assert self._transformer is not None
        assert self._cov is not None
        assert self._mean is not None

        z_x = self._transformer.transform(x.reshape(1, -1))[0]

        observed_idx = np.where(mask)[0]
        missing_idx = np.where(~mask)[0]

        z_samples = np.tile(z_x, (n_samples, 1))
        if missing_idx.size > 0:
            cond_mean, cond_cov = self._conditional_gaussian(z_x, observed_idx, missing_idx)
            gaussian_draws = self.random_state.multivariate_normal(cond_mean, cond_cov, size=n_samples)
            if gaussian_draws.ndim == 1:
                gaussian_draws = gaussian_draws.reshape(n_samples, 1)
            z_samples[:, missing_idx] = gaussian_draws

        samples = self._transformer.inverse_transform(z_samples)
        if observed_idx.size > 0:
            samples[:, observed_idx] = x[observed_idx]
        return samples

    def _conditional_gaussian(
        self, z_x: np.ndarray, observed_idx: np.ndarray, missing_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mu = self._mean
        cov = self._cov

        mu_missing = mu[missing_idx]
        cov_mm = cov[np.ix_(missing_idx, missing_idx)]

        if observed_idx.size == 0:
            return mu_missing, cov_mm

        mu_observed = mu[observed_idx]
        cov_mo = cov[np.ix_(missing_idx, observed_idx)]
        cov_oo = cov[np.ix_(observed_idx, observed_idx)]

        diff = z_x[observed_idx] - mu_observed
        solve_cov_oo = self._solve_psd(cov_oo, cov_mo.T)
        cond_mean = mu_missing + solve_cov_oo.T @ diff
        cond_cov = cov_mm - cov_mo @ solve_cov_oo
        cond_cov = self._ensure_positive_definite(cond_cov)
        return cond_mean, cond_cov

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        # Symmetrize and add jitter until the matrix is positive semi-definite.
        matrix = (matrix + matrix.T) / 2.0
        jitter = self.jitter
        identity = np.eye(matrix.shape[0])
        for _ in range(5):
            try:
                # Attempt a Cholesky decomposition to ensure positive definiteness.
                np.linalg.cholesky(matrix)
                return matrix
            except np.linalg.LinAlgError:
                matrix = matrix + jitter * identity
                jitter *= 10
        # Final fallback using eigenvalue clipping.
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = self.jitter
        return (eigvecs * eigvals) @ eigvecs.T

    def _solve_psd(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            a = a + self.jitter * np.eye(a.shape[0])
            return np.linalg.solve(a, b)


class ConditionalGeneratorMasker(Masker):
    """Masker that samples from a conditional generator.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Background dataset used to fit the conditional generator.
    generator : str, BaseConditionalGenerator, or type
        Generator specification. Passing ``"gaussian_copula"`` builds a
        :class:`GaussianCopulaGenerator`. Alternatively, a generator instance or
        class can be provided.
    m_mc : int, default=50
        Number of Monte-Carlo samples drawn for each mask evaluation.
    random_state : int, numpy.random.RandomState, numpy.random.Generator, or None
        Controls the randomness of the conditional generator.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]],
        generator: Union[str, BaseConditionalGenerator, type] = "gaussian_copula",
        m_mc: int = 50,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
    ):
        super().__init__()

        if m_mc <= 0:
            raise ValueError("m_mc must be a positive integer.")

        self.m_mc = int(m_mc)
        self.random_state = self._init_random_state(random_state)

        background, feature_names = self._as_array(data)
        self.data = background
        if feature_names is not None:
            self.feature_names = feature_names

        self.generator = self._init_generator(generator)
        self.generator.set_random_state(self.random_state)
        self.generator.fit(self.data)

        self.shape = (self.m_mc, self.data.shape[1])

    def _init_random_state(
        self, random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]
    ) -> np.random.RandomState:
        if isinstance(random_state, np.random.Generator):
            random_state = np.random.RandomState(random_state.integers(np.iinfo(np.int32).max))
        return check_random_state(random_state)

    def _init_generator(self, generator: Union[str, BaseConditionalGenerator, type]) -> BaseConditionalGenerator:
        if isinstance(generator, BaseConditionalGenerator):
            return generator
        if isinstance(generator, str):
            key = generator.lower()
            if key == "gaussian_copula":
                return GaussianCopulaGenerator(random_state=self.random_state)
            raise ValueError(f"Unknown generator '{generator}'.")
        if inspect.isclass(generator) and issubclass(generator, BaseConditionalGenerator):
            return generator(random_state=self.random_state)
        raise TypeError("generator must be a string identifier, generator instance, or generator class.")

    def _as_array(
        self, data: Union[np.ndarray, pd.DataFrame, Sequence[Sequence[float]]]
    ) -> tuple[np.ndarray, Optional[Sequence[str]]]:
        feature_names = None
        if isinstance(data, pd.DataFrame):
            feature_names = list(data.columns)
            array = data.values
        else:
            array = np.asarray(data, dtype=float)

        if array.ndim != 2:
            raise ValueError("data must be a 2D array-like structure.")

        return array, feature_names

    def __call__(self, mask: Union[np.ndarray, Sequence[Union[int, bool]]], x: Sequence[float]):
        mask_array = self._standardize_mask(mask, x)
        mask_array = np.asarray(mask_array, dtype=bool)

        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if mask_array.shape[0] != x_array.shape[0]:
            raise ValueError("mask and x must have the same dimensionality.")

        samples = self.generator.sample(x_array, mask_array, self.m_mc)
        masks = np.repeat(mask_array.reshape(1, -1), samples.shape[0], axis=0)
        return samples, masks
