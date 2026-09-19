from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backend.app.research.evaluation.forecast_metrics import (
    ForecastMetricError,
)


class DieboldMarianoError(ValueError):
    """Raised when the Diebold-Mariano test cannot be calculated."""


LossFunction = Literal[
    "absolute",
    "squared",
    "qlike",
]


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Result of a Diebold-Mariano forecast comparison."""

    statistic: float
    p_value: float
    mean_loss_difference: float
    observations: int
    loss_function: str
    alternative: str


def diebold_mariano_test(
    actual: pd.Series | np.ndarray,
    forecast_a: pd.Series | np.ndarray,
    forecast_b: pd.Series | np.ndarray,
    *,
    loss: LossFunction = "squared",
    alternative: Literal[
        "two_sided",
        "less",
        "greater",
    ] = "two_sided",
    lag: int | None = None,
) -> DieboldMarianoResult:
    """
    Perform a Diebold-Mariano test comparing two forecasts.

    The loss differential is:

        d_t = L(actual_t, forecast_a_t)
              - L(actual_t, forecast_b_t)

    The null hypothesis is:

        H0: E[d_t] = 0

    Parameters
    ----------
    actual:
        Realized out-of-sample values.

    forecast_a:
        Forecast series for model A.

    forecast_b:
        Forecast series for model B.

    loss:
        Forecast loss function:
        - "absolute"
        - "squared"
        - "qlike"

    alternative:
        Alternative hypothesis:
        - "two_sided"
        - "less"
        - "greater"

    lag:
        HAC autocovariance truncation lag. If None, an automatic
        lag based on the sample size is used.
    """

    y_true = _to_numpy(
        actual,
        "actual",
    )

    y_a = _to_numpy(
        forecast_a,
        "forecast_a",
    )

    y_b = _to_numpy(
        forecast_b,
        "forecast_b",
    )

    _validate_inputs(
        y_true,
        y_a,
        y_b,
    )

    if loss not in {
        "absolute",
        "squared",
        "qlike",
    }:
        raise DieboldMarianoError(
            "loss must be 'absolute', 'squared', or 'qlike'."
        )

    if alternative not in {
        "two_sided",
        "less",
        "greater",
    }:
        raise DieboldMarianoError(
            "alternative must be 'two_sided', 'less', "
            "or 'greater'."
        )

    effective_lag = _resolve_lag(
        len(y_true),
        lag,
    )

    loss_a = _calculate_loss(
        y_true,
        y_a,
        loss,
    )

    loss_b = _calculate_loss(
        y_true,
        y_b,
        loss,
    )

    loss_difference = loss_a - loss_b

    mean_difference = float(
        np.mean(loss_difference)
    )

    variance = _newey_west_long_run_variance(
        loss_difference,
        effective_lag,
    )

    if not np.isfinite(variance):
        raise DieboldMarianoError(
            "Estimated long-run variance is non-finite."
        )

    if variance <= 0:
        if np.isclose(
            mean_difference,
            0.0,
            atol=1e-15,
        ):
            statistic = 0.0
            p_value = 1.0

            return DieboldMarianoResult(
                statistic=statistic,
                p_value=p_value,
                mean_loss_difference=mean_difference,
                observations=len(y_true),
                loss_function=loss,
                alternative=alternative,
            )

        raise DieboldMarianoError(
            "Long-run variance is zero or negative while "
            "the mean loss difference is non-zero."
        )

    standard_error = np.sqrt(
        variance / len(loss_difference)
    )

    statistic = mean_difference / standard_error

    p_value = _normal_p_value(
        statistic,
        alternative,
    )

    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_loss_difference=mean_difference,
        observations=len(y_true),
        loss_function=loss,
        alternative=alternative,
    )


def calculate_loss_difference(
    actual: pd.Series | np.ndarray,
    forecast_a: pd.Series | np.ndarray,
    forecast_b: pd.Series | np.ndarray,
    *,
    loss: LossFunction = "squared",
) -> np.ndarray:
    """
    Calculate the Diebold-Mariano loss differential:

        L(A) - L(B)
    """

    y_true = _to_numpy(
        actual,
        "actual",
    )

    y_a = _to_numpy(
        forecast_a,
        "forecast_a",
    )

    y_b = _to_numpy(
        forecast_b,
        "forecast_b",
    )

    _validate_inputs(
        y_true,
        y_a,
        y_b,
    )

    if loss not in {
        "absolute",
        "squared",
        "qlike",
    }:
        raise DieboldMarianoError(
            "loss must be 'absolute', 'squared', or 'qlike'."
        )

    return (
        _calculate_loss(
            y_true,
            y_a,
            loss,
        )
        - _calculate_loss(
            y_true,
            y_b,
            loss,
        )
    )


def _calculate_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    loss: LossFunction,
) -> np.ndarray:
    """Calculate the selected pointwise forecast loss."""

    errors = actual - forecast

    if loss == "absolute":
        values = np.abs(errors)

    elif loss == "squared":
        values = errors**2

    elif loss == "qlike":
        if (actual <= 0).any():
            raise ForecastMetricError(
                "QLIKE requires actual values greater than zero."
            )

        if (forecast <= 0).any():
            raise ForecastMetricError(
                "QLIKE requires forecast values greater than zero."
            )

        ratio = actual / forecast

        values = (
            ratio
            - np.log(ratio)
            - 1.0
        )

    else:
        raise DieboldMarianoError(
            f"Unsupported loss function: {loss}"
        )

    if not np.isfinite(values).all():
        raise DieboldMarianoError(
            "Forecast loss contains non-finite values."
        )

    return values


def _newey_west_long_run_variance(
    values: np.ndarray,
    lag: int,
) -> float:
    """
    Estimate the long-run variance of the loss differential.

    Uses a Bartlett/Newey-West weighting scheme.
    """

    centered = values - np.mean(values)

    n = len(centered)

    gamma_0 = float(
        np.mean(centered * centered)
    )

    long_run_variance = gamma_0

    for current_lag in range(
        1,
        lag + 1,
    ):
        autocovariance = float(
            np.mean(
                centered[current_lag:]
                * centered[:-current_lag]
            )
        )

        weight = 1.0 - (
            current_lag / (lag + 1.0)
        )

        long_run_variance += (
            2.0
            * weight
            * autocovariance
        )

    return float(
        max(long_run_variance, 0.0)
    )


def _resolve_lag(
    observations: int,
    lag: int | None,
) -> int:
    """Validate or derive the HAC truncation lag."""

    if observations < 2:
        raise DieboldMarianoError(
            "At least two observations are required."
        )

    if lag is None:
        return max(
            0,
            int(
                np.floor(
                    observations ** (1.0 / 3.0)
                )
            ),
        )

    if isinstance(lag, bool) or not isinstance(
        lag,
        int,
    ):
        raise DieboldMarianoError(
            "lag must be an integer or None."
        )

    if lag < 0:
        raise DieboldMarianoError(
            "lag must not be negative."
        )

    if lag >= observations:
        raise DieboldMarianoError(
            "lag must be smaller than the number "
            "of observations."
        )

    return lag


def _normal_p_value(
    statistic: float,
    alternative: str,
) -> float:
    """Calculate a normal-approximation p-value."""

    normal_cdf = 0.5 * (
        1.0
        + _erf(
            statistic / np.sqrt(2.0)
        )
    )

    if alternative == "two_sided":
        p_value = 2.0 * min(
            normal_cdf,
            1.0 - normal_cdf,
        )

    elif alternative == "less":
        p_value = normal_cdf

    elif alternative == "greater":
        p_value = 1.0 - normal_cdf

    else:
        raise DieboldMarianoError(
            f"Unsupported alternative: {alternative}"
        )

    return float(
        np.clip(
            p_value,
            0.0,
            1.0,
        )
    )


def _erf(value: float) -> float:
    """Scalar error function using scipy."""

    from scipy.special import erf

    return float(
        erf(value)
    )


def _validate_inputs(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
) -> None:
    if actual.ndim != 1:
        raise DieboldMarianoError(
            "actual must be one-dimensional."
        )

    if forecast_a.ndim != 1:
        raise DieboldMarianoError(
            "forecast_a must be one-dimensional."
        )

    if forecast_b.ndim != 1:
        raise DieboldMarianoError(
            "forecast_b must be one-dimensional."
        )

    if len(actual) != len(forecast_a):
        raise DieboldMarianoError(
            "actual and forecast_a must have the same "
            "number of observations."
        )

    if len(actual) != len(forecast_b):
        raise DieboldMarianoError(
            "actual and forecast_b must have the same "
            "number of observations."
        )

    if len(actual) < 2:
        raise DieboldMarianoError(
            "At least two observations are required."
        )

    if not np.isfinite(actual).all():
        raise DieboldMarianoError(
            "actual contains non-finite values."
        )

    if not np.isfinite(forecast_a).all():
        raise DieboldMarianoError(
            "forecast_a contains non-finite values."
        )

    if not np.isfinite(forecast_b).all():
        raise DieboldMarianoError(
            "forecast_b contains non-finite values."
        )


def _to_numpy(
    values: pd.Series | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert supported input to a one-dimensional NumPy array."""

    if isinstance(
        values,
        pd.Series,
    ):
        array = values.to_numpy(
            dtype=float
        )

    elif isinstance(
        values,
        np.ndarray,
    ):
        array = np.asarray(
            values,
            dtype=float,
        )

    else:
        raise TypeError(
            f"{name} must be a pandas.Series or numpy.ndarray."
        )

    return array