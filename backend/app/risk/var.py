from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class VaRError(ValueError):
    """Raised when VaR cannot be calculated."""


@dataclass(frozen=True)
class VaRResult:
    """Value-at-Risk result for a single confidence level."""

    confidence_level: float
    quantile: float
    var: float


@dataclass(frozen=True)
class VaRLevelsResult:
    """VaR results for the required 95% and 99% confidence levels."""

    var_95: VaRResult
    var_99: VaRResult


def calculate_var(
    simulated_returns: np.ndarray,
    *,
    confidence_level: float,
) -> VaRResult:
    """
    Calculate Value-at-Risk from a simulated return distribution.

    VaR is reported as a positive loss magnitude:

        VaR_c = -Q_(1-c)(r)

    where:
        c = confidence level
        r = simulated returns
    """

    returns = _validate_returns(simulated_returns)
    confidence_level = _validate_confidence_level(
        confidence_level
    )

    tail_probability = 1.0 - confidence_level

    quantile = float(
        np.quantile(
            returns,
            tail_probability,
        )
    )

    var = -quantile

    return VaRResult(
        confidence_level=confidence_level,
        quantile=quantile,
        var=float(var),
    )


def calculate_var_95(
    simulated_returns: np.ndarray,
) -> VaRResult:
    """Calculate 95% Value-at-Risk."""

    return calculate_var(
        simulated_returns,
        confidence_level=0.95,
    )


def calculate_var_99(
    simulated_returns: np.ndarray,
) -> VaRResult:
    """Calculate 99% Value-at-Risk."""

    return calculate_var(
        simulated_returns,
        confidence_level=0.99,
    )


def calculate_required_var_levels(
    simulated_returns: np.ndarray,
) -> VaRLevelsResult:
    """
    Calculate the required 95% and 99% VaR levels.
    """

    return VaRLevelsResult(
        var_95=calculate_var_95(
            simulated_returns
        ),
        var_99=calculate_var_99(
            simulated_returns
        ),
    )


def _validate_returns(
    simulated_returns: np.ndarray,
) -> np.ndarray:
    returns = np.asarray(
        simulated_returns,
        dtype=float,
    )

    if returns.ndim != 1:
        raise VaRError(
            "simulated_returns must be one-dimensional."
        )

    if returns.size == 0:
        raise VaRError(
            "simulated_returns must not be empty."
        )

    if not np.isfinite(returns).all():
        raise VaRError(
            "simulated_returns contains non-finite values."
        )

    return returns


def _validate_confidence_level(
    confidence_level: float,
) -> float:
    if not isinstance(
        confidence_level,
        (int, float),
    ):
        raise TypeError(
            "confidence_level must be numeric."
        )

    confidence_level = float(
        confidence_level
    )

    if not np.isfinite(confidence_level):
        raise VaRError(
            "confidence_level must be finite."
        )

    if not 0.0 < confidence_level < 1.0:
        raise VaRError(
            "confidence_level must be between 0 and 1."
        )

    return confidence_level