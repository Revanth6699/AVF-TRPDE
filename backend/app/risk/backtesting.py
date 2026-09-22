from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BacktestingError(ValueError):
    """Raised when VaR backtesting inputs are invalid."""


@dataclass(frozen=True)
class VaRBacktestData:
    """Validated data required for VaR backtesting."""

    actual_returns: np.ndarray
    var_losses: np.ndarray
    exceptions: np.ndarray
    observation_count: int
    exception_count: int
    exception_rate: float


def prepare_var_backtest(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
) -> VaRBacktestData:
    """
    Prepare realized returns and VaR loss thresholds for backtesting.

    Parameters
    ----------
    actual_returns:
        Realized returns for the out-of-sample period.

    var_losses:
        VaR expressed as a positive loss magnitude.

    Exception definition
    --------------------
    An exception occurs when the realized return is below
    the negative VaR loss threshold:

        return_t <= -VaR_t
    """

    actual = _validate_array(
        actual_returns,
        name="actual_returns",
    )

    var = _validate_array(
        var_losses,
        name="var_losses",
    )

    if actual.size != var.size:
        raise BacktestingError(
            "actual_returns and var_losses must have the same length."
        )

    if (var < 0).any():
        raise BacktestingError(
            "var_losses must contain non-negative loss magnitudes."
        )

    exceptions = actual <= -var

    exception_count = int(exceptions.sum())
    observation_count = int(exceptions.size)

    exception_rate = (
        exception_count / observation_count
    )

    return VaRBacktestData(
        actual_returns=actual,
        var_losses=var,
        exceptions=exceptions.astype(bool),
        observation_count=observation_count,
        exception_count=exception_count,
        exception_rate=exception_rate,
    )


def identify_exceptions(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
) -> np.ndarray:
    """
    Identify VaR exceptions.

    Returns
    -------
    np.ndarray
        Boolean array where True indicates a VaR exception.
    """

    result = prepare_var_backtest(
        actual_returns,
        var_losses,
    )

    return result.exceptions.copy()


def exception_count(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
) -> int:
    """Return the number of VaR exceptions."""

    result = prepare_var_backtest(
        actual_returns,
        var_losses,
    )

    return result.exception_count


def exception_rate(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
) -> float:
    """Return the observed VaR exception rate."""

    result = prepare_var_backtest(
        actual_returns,
        var_losses,
    )

    return result.exception_rate


def _validate_array(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise BacktestingError(
            f"{name} must contain numeric values."
        ) from exc

    if array.ndim != 1:
        raise BacktestingError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise BacktestingError(
            f"{name} must not be empty."
        )

    if not np.isfinite(array).all():
        raise BacktestingError(
            f"{name} must contain only finite values."
        )

    return array