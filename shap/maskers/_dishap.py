"""Maskers powered by conditional generators for DI-SHAP style explainers.

This module introduces a small abstraction for conditional data generators and a
masker that relies on them to generate background samples conditioned on the
observed features of the instance under explanation.
"""

from __future__ import annotations

import inspect
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state

from ._masker import Masker


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
