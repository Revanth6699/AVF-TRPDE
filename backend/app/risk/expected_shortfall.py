from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ExpectedShortfallError(ValueError):
    """Raised when Expected Shortfall cannot be calculated."""


@dataclass(frozen=True)
class ExpectedShortfallResult:
    """Expected Shortfall result for a single confidence level."""

    confidence_level: float
    var: float
    expected_shortfall: float
    tail_observation_count: int


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float,
) -> ExpectedShortfallResult:
    """
    Calculate Expected Shortfall from a return distribution.

    Parameters
    ----------
    returns:
        One-dimensional array of simulated or historical returns.

    confidence_level:
        Confidence level, for example 0.95 or 0.99.

    Returns
    -------
    ExpectedShortfallResult
        VaR threshold and mean loss beyond that threshold.

    Notes
    -----
    VaR is expressed as a positive loss magnitude:

        VaR_c = -Q_(1-c)(returns)

    Expected Shortfall is:

        ES_c = -E[r | r <= Q_(1-c)]

    Therefore both VaR and Expected Shortfall are positive
    loss magnitudes.
    """

    clean_returns = _validate_returns(returns)
    _validate_confidence_level(confidence_level)

    quantile = float(
        np.quantile(
            clean_returns,
            1.0 - confidence_level,
        )
    )

    tail_returns = clean_returns[
        clean_returns <= quantile
    ]

    if tail_returns.size == 0:
        raise ExpectedShortfallError(
            "No observations are available in the loss tail."
        )

    var = -quantile
    expected_shortfall = -float(
        np.mean(tail_returns)
    )

    return ExpectedShortfallResult(
        confidence_level=confidence_level,
        var=var,
        expected_shortfall=expected_shortfall,
        tail_observation_count=int(tail_returns.size),
    )


def calculate_es_95(
    returns: np.ndarray,
) -> ExpectedShortfallResult:
    """Calculate Expected Shortfall at 95% confidence."""
    return calculate_expected_shortfall(
        returns,
        confidence_level=0.95,
    )


def calculate_es_99(
    returns: np.ndarray,
) -> ExpectedShortfallResult:
    """Calculate Expected Shortfall at 99% confidence."""
    return calculate_expected_shortfall(
        returns,
        confidence_level=0.99,
    )


def calculate_required_es_levels(
    returns: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> dict[float, ExpectedShortfallResult]:
    """
    Calculate Expected Shortfall for multiple confidence levels.
    """

    if not confidence_levels:
        raise ExpectedShortfallError(
            "At least one confidence level is required."
        )

    results: dict[float, ExpectedShortfallResult] = {}

    for confidence_level in confidence_levels:
        if confidence_level in results:
            raise ExpectedShortfallError(
                "Confidence levels must not contain duplicates."
            )

        results[confidence_level] = calculate_expected_shortfall(
            returns,
            confidence_level,
        )

    return results


def _validate_returns(
    returns: np.ndarray,
) -> np.ndarray:
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ExpectedShortfallError(
            "returns must be a one-dimensional array."
        )

    if returns.size == 0:
        raise ExpectedShortfallError(
            "returns must not be empty."
        )

    try:
        clean_returns = returns.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ExpectedShortfallError(
            "returns must contain numeric values."
        ) from exc

    if not np.isfinite(clean_returns).all():
        raise ExpectedShortfallError(
            "returns must contain only finite values."
        )

    return clean_returns


def _validate_confidence_level(
    confidence_level: float,
) -> None:
    if not np.isfinite(confidence_level):
        raise ExpectedShortfallError(
            "confidence_level must be finite."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ExpectedShortfallError(
            "confidence_level must be between 0 and 1."
        )