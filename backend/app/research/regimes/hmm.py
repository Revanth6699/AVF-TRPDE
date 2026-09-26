from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMError(ValueError):
    """Raised when HMM regime detection fails."""


@dataclass(frozen=True)
class HMMConfig:
    """Configuration for the Gaussian HMM regime detector."""

    n_components: int
    covariance_type: str = "full"
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42


@dataclass(frozen=True)
class HMMResult:
    """Result from HMM regime detection."""

    probabilities: pd.DataFrame
    regimes: pd.Series
    log_likelihood: float
    n_components: int


class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.

    The HMM is a regime detector, not a volatility forecasting model.

    Filtering probabilities are calculated as:

        P(S_t = k | O_1:t)

    No backward smoothing is used, so future observations do not
    influence the regime probability available at time t.

    HMM observations are standardized using statistics estimated only
    from the observations supplied to fit(). The same transformation
    is then applied to all subsequent observations.
    """

    def __init__(
        self,
        config: HMMConfig,
    ) -> None:
        self._validate_config(config)

        self.config = config

        self._model = GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state,
        )

        self._fitted = False
        self._last_filter_probability: np.ndarray | None = None

        self._observation_mean: np.ndarray | None = None
        self._observation_scale: np.ndarray | None = None

    @property
    def model(self) -> GaussianHMM:
        """Return the fitted HMM object."""

        self._require_fitted()

        return self._model

    @property
    def is_fitted(self) -> bool:
        """Return whether the HMM has been fitted."""

        return self._fitted

    def fit(
        self,
        observations: pd.DataFrame | np.ndarray,
    ) -> "HMMRegimeDetector":
        """
        Fit the HMM using the supplied training observations.

        The observation transformation is fitted exclusively on this
        training data. In walk-forward validation this therefore remains
        fold-local.
        """

        values = self._validate_observations(
            observations
        )

        self._fit_observation_scaler(
            values
        )

        scaled_values = self._transform_observations(
            values
        )

        try:
            self._model.fit(
                scaled_values
            )
        except Exception as exc:
            raise HMMError(
                f"HMM fitting failed: {exc}"
            ) from exc

        if not self._model.monitor_.converged:
            raise HMMError(
                "HMM optimizer did not converge."
            )

        self._validate_fitted_covariances()

        training_probabilities = self._filter_probabilities(
            scaled_values,
            initial_probability=self._model.startprob_,
        )

        self._last_filter_probability = (
            training_probabilities[-1].copy()
        )

        self._fitted = True

        return self

    def filter_probabilities(
        self,
        observations: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Calculate filtering probabilities for observations.

        Calculates:

            P(S_t | O_1:t)

        without using future observations.
        """

        self._require_fitted()

        values = self._validate_observations(
            observations
        )

        scaled_values = self._transform_observations(
            values
        )

        return self._filter_probabilities(
            scaled_values,
            initial_probability=self._model.startprob_,
        )

    def filter_after_training(
        self,
        observations: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Calculate filtering probabilities for observations immediately
        following the fitted training window.

        The filtering process begins from the final filtered state
        probability of the training window.
        """

        self._require_fitted()

        if self._last_filter_probability is None:
            raise HMMError(
                "Training filter state is unavailable."
            )

        values = self._validate_observations(
            observations,
            min_observations=1,
        )

        scaled_values = self._transform_observations(
            values
        )

        return self._filter_probabilities(
            scaled_values,
            initial_probability=self._last_filter_probability,
            propagate_initial=True,
        )

    def predict_regimes(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Return the highest-probability regime for each observation."""

        probabilities = self._validate_probabilities(
            probabilities
        )

        return np.argmax(
            probabilities,
            axis=1,
        )

    def fit_result(
        self,
        observations: pd.DataFrame | np.ndarray,
    ) -> HMMResult:
        """Fit the HMM and return its training results."""

        self.fit(
            observations
        )

        probabilities = self.filter_probabilities(
            observations
        )

        probability_columns = [
            f"regime_probability_{index}"
            for index in range(
                self.config.n_components
            )
        ]

        probability_frame = pd.DataFrame(
            probabilities,
            columns=probability_columns,
        )

        regimes = pd.Series(
            self.predict_regimes(
                probabilities
            ),
            name="regime",
        )

        scaled_values = self._transform_observations(
            self._validate_observations(
                observations
            )
        )

        return HMMResult(
            probabilities=probability_frame,
            regimes=regimes,
            log_likelihood=float(
                self._model.score(
                    scaled_values
                )
            ),
            n_components=self.config.n_components,
        )

    def _fit_observation_scaler(
        self,
        observations: np.ndarray,
    ) -> None:
        """Fit the observation standardization on training data only."""

        mean = np.mean(
            observations,
            axis=0,
        )

        scale = np.std(
            observations,
            axis=0,
            ddof=0,
        )

        if not np.isfinite(mean).all():
            raise HMMError(
                "HMM observation means are non-finite."
            )

        if not np.isfinite(scale).all():
            raise HMMError(
                "HMM observation scales are non-finite."
            )

        # Constant observation variables cannot be standardized by
        # their empirical standard deviation. A unit scale preserves
        # the centered variable without introducing division by zero.
        scale = np.where(
            scale > np.finfo(float).eps,
            scale,
            1.0,
        )

        self._observation_mean = mean
        self._observation_scale = scale

    def _transform_observations(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Apply the training-fold observation transformation."""

        if (
            self._observation_mean is None
            or self._observation_scale is None
        ):
            raise HMMError(
                "HMM observation scaler is not fitted."
            )

        if observations.shape[1] != (
            self._observation_mean.shape[0]
        ):
            raise HMMError(
                "Observation feature count does not match "
                "the fitted HMM."
            )

        transformed = (
            observations
            - self._observation_mean
        ) / self._observation_scale

        if not np.isfinite(
            transformed
        ).all():
            raise HMMError(
                "Scaled HMM observations contain "
                "non-finite values."
            )

        return transformed

    def _validate_fitted_covariances(
        self,
    ) -> None:
        """Validate fitted HMM covariance matrices."""

        covariances = np.asarray(
            self._model.covars_,
            dtype=float,
        )

        if not np.isfinite(
            covariances
        ).all():
            raise HMMError(
                "HMM fitted covariances contain "
                "non-finite values."
            )

        if self.config.covariance_type == "full":
            for state, covariance in enumerate(
                covariances
            ):
                if not np.allclose(
                    covariance,
                    covariance.T,
                    atol=1e-10,
                    rtol=1e-8,
                ):
                    raise HMMError(
                        "HMM covariance matrix for state "
                        f"{state} is not symmetric."
                    )

                eigenvalues = np.linalg.eigvalsh(
                    covariance
                )

                if not np.isfinite(
                    eigenvalues
                ).all():
                    raise HMMError(
                        "HMM covariance eigenvalues are "
                        "non-finite."
                    )

                if eigenvalues.min() <= 0:
                    raise HMMError(
                        "HMM covariance matrix for state "
                        f"{state} is not positive-definite."
                    )

        elif self.config.covariance_type == "diag":
            if (
                covariances <= 0
            ).any():
                raise HMMError(
                    "HMM diagonal covariance contains "
                    "non-positive values."
                )

    def _filter_probabilities(
        self,
        observations: np.ndarray,
        *,
        initial_probability: np.ndarray,
        propagate_initial: bool = False,
    ) -> np.ndarray:
        """
        Run the forward filtering recursion.

        No backward pass or smoothing is performed.
        """

        n_observations = observations.shape[0]
        n_states = self.config.n_components

        initial_probability = np.asarray(
            initial_probability,
            dtype=float,
        )

        if initial_probability.shape != (
            n_states,
        ):
            raise HMMError(
                "Initial HMM state probability has an invalid shape."
            )

        probabilities = np.empty(
            (n_observations, n_states),
            dtype=float,
        )

        previous = initial_probability.copy()

        if propagate_initial:
            previous = self._propagate(
                previous
            )

        for index, observation in enumerate(
            observations
        ):
            emission = self._emission_probabilities(
                observation
            )

            current = previous * emission

            total = float(
                current.sum()
            )

            if not np.isfinite(total) or total <= 0:
                raise HMMError(
                    "HMM filtering produced an invalid "
                    "probability normalization."
                )

            current /= total

            probabilities[index] = current

            previous = self._propagate(
                current
            )

        return probabilities

    def _propagate(
        self,
        probability: np.ndarray,
    ) -> np.ndarray:
        """Propagate filtered probabilities through the transition matrix."""

        propagated = probability @ np.asarray(
            self._model.transmat_,
            dtype=float,
        )

        total = float(
            propagated.sum()
        )

        if not np.isfinite(total) or total <= 0:
            raise HMMError(
                "HMM transition probabilities became invalid."
            )

        return propagated / total

    def _emission_probabilities(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Calculate Gaussian emission probabilities for one observation."""

        means = np.asarray(
            self._model.means_,
            dtype=float,
        )

        covariance = np.asarray(
            self._model.covars_,
            dtype=float,
        )

        n_states = self.config.n_components

        probabilities = np.empty(
            n_states,
            dtype=float,
        )

        for state in range(
            n_states
        ):
            mean = means[state]

            if self.config.covariance_type == "full":
                covariance_matrix = covariance[state]

            elif self.config.covariance_type == "diag":
                covariance_matrix = np.diag(
                    covariance[state]
                )

            else:
                raise HMMError(
                    "Unsupported covariance type."
                )

            covariance_matrix = (
                covariance_matrix
                + np.eye(
                    covariance_matrix.shape[0]
                ) * 1e-12
            )

            sign, log_determinant = np.linalg.slogdet(
                covariance_matrix
            )

            if sign <= 0:
                raise HMMError(
                    "HMM covariance matrix is not positive definite."
                )

            difference = observation - mean

            try:
                solved = np.linalg.solve(
                    covariance_matrix,
                    difference,
                )
            except np.linalg.LinAlgError as exc:
                raise HMMError(
                    "HMM covariance matrix could not be solved."
                ) from exc

            quadratic = float(
                difference @ solved
            )

            log_probability = -0.5 * (
                len(observation) * np.log(2.0 * np.pi)
                + log_determinant
                + quadratic
            )

            probabilities[state] = np.exp(
                np.clip(
                    log_probability,
                    -745.0,
                    700.0,
                )
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise HMMError(
                "HMM emission probabilities are non-finite."
            )

        return probabilities

    def _require_fitted(
        self,
    ) -> None:
        """Ensure the HMM has been fitted before inference."""

        if not self._fitted:
            raise HMMError(
                "HMM model has not been fitted."
            )

    @staticmethod
    def _validate_config(
        config: HMMConfig,
    ) -> None:
        if not isinstance(
            config.n_components,
            int,
        ):
            raise TypeError(
                "n_components must be an integer."
            )

        if isinstance(
            config.n_components,
            bool,
        ):
            raise TypeError(
                "n_components must be an integer."
            )

        if config.n_components < 2:
            raise HMMError(
                "n_components must be at least 2."
            )

        if config.covariance_type not in {
            "full",
            "diag",
        }:
            raise HMMError(
                "covariance_type must be 'full' or 'diag'."
            )

        if config.n_iter <= 0:
            raise HMMError(
                "n_iter must be greater than zero."
            )

        if config.tol <= 0:
            raise HMMError(
                "tol must be greater than zero."
            )

    @staticmethod
    def _validate_observations(
        observations: pd.DataFrame | np.ndarray,
        *,
        min_observations: int = 2,
    ) -> np.ndarray:
        if isinstance(
            observations,
            pd.DataFrame,
        ):
            values = observations.to_numpy(
                dtype=float
            )

        elif isinstance(
            observations,
            np.ndarray,
        ):
            values = np.asarray(
                observations,
                dtype=float,
            )

        else:
            raise TypeError(
                "observations must be a pandas.DataFrame "
                "or numpy.ndarray."
            )

        if values.ndim != 2:
            raise HMMError(
                "observations must be a two-dimensional matrix."
            )

        if values.shape[0] < min_observations:
            raise HMMError(
                f"At least {min_observations} observation"
                f"{'s' if min_observations != 1 else ''} "
                "are required."
            )

        if values.shape[1] < 1:
            raise HMMError(
                "At least one observation variable is required."
            )

        if not np.isfinite(values).all():
            raise HMMError(
                "observations must contain only finite values."
            )

        return values
        

    @staticmethod
    def _validate_probabilities(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            probabilities,
            dtype=float,
        )

        if values.ndim != 2:
            raise HMMError(
                "probabilities must be a two-dimensional matrix."
            )

        if not np.isfinite(
            values
        ).all():
            raise HMMError(
                "probabilities contain non-finite values."
            )

        if (values < 0).any():
            raise HMMError(
                "probabilities must not be negative."
            )

        row_sums = values.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            raise HMMError(
                "Each probability row must sum to one."
            )

        return values


def fit_hmm(
    observations: pd.DataFrame | np.ndarray,
    *,
    n_components: int,
    covariance_type: str = "full",
    n_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
) -> HMMRegimeDetector:
    """Create and fit the HMM regime detector."""

    config = HMMConfig(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )

    detector = HMMRegimeDetector(
        config=config,
    )

    detector.fit(
        observations
    )

    return detector