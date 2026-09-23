from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class BootstrapError(ValueError):
    """Raised when bootstrap estimation cannot be performed."""


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap estimate and confidence interval."""

    estimate: float
    standard_error: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    bootstrap_samples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the result as a dictionary."""
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "bootstrap_samples": self.bootstrap_samples,
        }


def bootstrap_confidence_interval(
    values: np.ndarray | list[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2_000,
    random_state: int | None = 42,
) -> BootstrapResult:
    """
    Estimate a statistic and its bootstrap confidence interval.

    The percentile bootstrap is used.

    Args:
        values:
            One-dimensional observations.
        statistic:
            Statistic calculated on the observations.
        confidence_level:
            Confidence level between 0 and 1.
        n_bootstrap:
            Number of bootstrap resamples.
        random_state:
            Seed for deterministic results. None disables fixed seeding.

    Returns:
        BootstrapResult containing the point estimate and
        percentile confidence interval.
    """
    array = _to_array(values)

    _validate_parameters(
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )

    try:
        estimate = float(statistic(array))
    except Exception as exc:
        raise BootstrapError(
            "Unable to calculate the bootstrap statistic."
        ) from exc

    if not np.isfinite(estimate):
        raise BootstrapError(
            "Bootstrap statistic returned a non-finite value."
        )

    rng = np.random.default_rng(random_state)

    bootstrap_statistics = np.empty(
        n_bootstrap,
        dtype=float,
    )

    observations = len(array)

    for index in range(n_bootstrap):
        sample = rng.choice(
            array,
            size=observations,
            replace=True,
        )

        try:
            value = float(statistic(sample))
        except Exception as exc:
            raise BootstrapError(
                "Bootstrap statistic failed during resampling."
            ) from exc

        if not np.isfinite(value):
            raise BootstrapError(
                "Bootstrap statistic returned a non-finite "
                "value during resampling."
            )

        bootstrap_statistics[index] = value

    alpha = 1.0 - confidence_level

    lower_bound = float(
        np.quantile(
            bootstrap_statistics,
            alpha / 2.0,
        )
    )

    upper_bound = float(
        np.quantile(
            bootstrap_statistics,
            1.0 - alpha / 2.0,
        )
    )

    standard_error = float(
        np.std(
            bootstrap_statistics,
            ddof=1,
        )
    )

    return BootstrapResult(
        estimate=estimate,
        standard_error=standard_error,
        confidence_level=confidence_level,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bootstrap_samples=n_bootstrap,
    )


def bootstrap_metric(
    actual: np.ndarray | list[float],
    forecast: np.ndarray | list[float],
    *,
    metric: str = "mae",
    confidence_level: float = 0.95,
    n_bootstrap: int = 2_000,
    random_state: int | None = 42,
) -> BootstrapResult:
    """
    Bootstrap a forecast metric.

    Supported metrics:
        - mae
        - rmse
        - qlike
    """
    actual_array = _to_array(actual)
    forecast_array = _to_array(forecast)

    _validate_matching_lengths(
        actual_array,
        forecast_array,
    )

    metric_name = metric.strip().lower()

    if metric_name not in {
        "mae",
        "rmse",
        "qlike",
    }:
        raise BootstrapError(
            "metric must be 'mae', 'rmse', or 'qlike'."
        )

    if metric_name == "qlike":
        if np.any(actual_array <= 0):
            raise BootstrapError(
                "Actual values must be strictly positive "
                "for QLIKE."
            )

        if np.any(forecast_array <= 0):
            raise BootstrapError(
                "Forecast values must be strictly positive "
                "for QLIKE."
            )

    def metric_statistic(
        indices: np.ndarray,
    ) -> float:
        sample_actual = actual_array[indices]
        sample_forecast = forecast_array[indices]

        if metric_name == "mae":
            return float(
                np.mean(
                    np.abs(
                        sample_actual
                        - sample_forecast
                    )
                )
            )

        if metric_name == "rmse":
            return float(
                np.sqrt(
                    np.mean(
                        np.square(
                            sample_actual
                            - sample_forecast
                        )
                    )
                )
            )

        ratio = (
            sample_actual
            / sample_forecast
        )

        return float(
            np.mean(
                ratio
                - np.log(ratio)
                - 1.0
            )
        )

    return _bootstrap_indices(
        observations=len(actual_array),
        statistic=metric_statistic,
        point_estimate=metric_statistic(
            np.arange(len(actual_array))
        ),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def bootstrap_difference(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2_000,
    random_state: int | None = 42,
) -> BootstrapResult:
    """
    Bootstrap the mean difference between two paired series.

    Difference is defined as:

        mean(values_a - values_b)
    """
    array_a = _to_array(values_a)
    array_b = _to_array(values_b)

    _validate_matching_lengths(
        array_a,
        array_b,
    )

    differences = array_a - array_b

    return bootstrap_confidence_interval(
        differences,
        statistic=np.mean,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def bootstrap_sharpe_ratio(
    returns: np.ndarray | list[float],
    *,
    annualization_factor: float = 252.0,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2_000,
    random_state: int | None = 42,
) -> BootstrapResult:
    """
    Bootstrap the annualized Sharpe ratio.

    Sharpe is calculated as:

        mean(return) / std(return)
        × sqrt(annualization_factor)

    A zero standard deviation is rejected because the
    Sharpe ratio would be undefined.
    """
    array = _to_array(returns)

    if annualization_factor <= 0:
        raise BootstrapError(
            "annualization_factor must be greater than zero."
        )

    def sharpe_statistic(
        sample: np.ndarray,
    ) -> float:
        standard_deviation = float(
            np.std(
                sample,
                ddof=1,
            )
        )

        if standard_deviation <= 0:
            raise BootstrapError(
                "Sharpe ratio is undefined for zero "
                "standard deviation."
            )

        return float(
            (
                np.mean(sample)
                / standard_deviation
            )
            * np.sqrt(
                annualization_factor
            )
        )

    return bootstrap_confidence_interval(
        array,
        statistic=sharpe_statistic,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def _bootstrap_indices(
    *,
    observations: int,
    statistic: Callable[[np.ndarray], float],
    point_estimate: float,
    confidence_level: float,
    n_bootstrap: int,
    random_state: int | None,
) -> BootstrapResult:
    """Bootstrap a statistic using resampled observation indices."""

    _validate_parameters(
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )

    if observations < 2:
        raise BootstrapError(
            "At least two observations are required."
        )

    rng = np.random.default_rng(random_state)

    bootstrap_statistics = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        sample_indices = rng.integers(
            low=0,
            high=observations,
            size=observations,
        )

        try:
            value = float(
                statistic(sample_indices)
            )
        except BootstrapError:
            raise
        except Exception as exc:
            raise BootstrapError(
                "Bootstrap statistic failed during resampling."
            ) from exc

        if not np.isfinite(value):
            raise BootstrapError(
                "Bootstrap statistic returned a "
                "non-finite value."
            )

        bootstrap_statistics[index] = value

    alpha = 1.0 - confidence_level

    lower_bound = float(
        np.quantile(
            bootstrap_statistics,
            alpha / 2.0,
        )
    )

    upper_bound = float(
        np.quantile(
            bootstrap_statistics,
            1.0 - alpha / 2.0,
        )
    )

    standard_error = float(
        np.std(
            bootstrap_statistics,
            ddof=1,
        )
    )

    return BootstrapResult(
        estimate=float(point_estimate),
        standard_error=standard_error,
        confidence_level=confidence_level,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bootstrap_samples=n_bootstrap,
    )


def _to_array(
    values: np.ndarray | list[float],
) -> np.ndarray:
    """Convert values to a validated one-dimensional array."""

    if isinstance(values, np.ndarray):
        try:
            array = values.astype(
                float,
                copy=False,
            )
        except (TypeError, ValueError) as exc:
            raise BootstrapError(
                "Values contain non-numeric data."
            ) from exc

    elif isinstance(values, list):
        try:
            array = np.asarray(
                values,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise BootstrapError(
                "Values contain non-numeric data."
            ) from exc

    else:
        raise TypeError(
            "Values must be a numpy array or list."
        )

    if array.ndim != 1:
        raise BootstrapError(
            "Values must be one-dimensional."
        )

    if len(array) < 2:
        raise BootstrapError(
            "At least two observations are required."
        )

    if not np.isfinite(array).all():
        raise BootstrapError(
            "Values contain NaN or infinite values."
        )

    return array


def _validate_matching_lengths(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> None:
    """Validate paired-series lengths."""

    if len(values_a) != len(values_b):
        raise BootstrapError(
            "Both series must have the same number "
            "of observations."
        )


def _validate_parameters(
    *,
    confidence_level: float,
    n_bootstrap: int,
) -> None:
    """Validate bootstrap parameters."""

    if not 0.0 < confidence_level < 1.0:
        raise BootstrapError(
            "confidence_level must be between 0 and 1."
        )

    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, int)
    ):
        raise BootstrapError(
            "n_bootstrap must be an integer."
        )

    if n_bootstrap < 100:
        raise BootstrapError(
            "n_bootstrap must be at least 100."
        )