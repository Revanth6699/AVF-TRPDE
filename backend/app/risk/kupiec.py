from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from backend.app.risk.backtesting import (
    prepare_var_backtest,
)


class KupiecTestError(ValueError):
    """Raised when the Kupiec test cannot be calculated."""


@dataclass(frozen=True)
class KupiecResult:
    """Result of the Kupiec unconditional coverage test."""

    confidence_level: float
    expected_exception_rate: float
    observed_exception_rate: float
    observation_count: int
    exception_count: int
    lr_statistic: float
    p_value: float
    degrees_of_freedom: int
    reject_null: bool


def kupiec_unconditional_coverage(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    confidence_level: float,
    *,
    significance_level: float = 0.05,
) -> KupiecResult:
    """
    Perform the Kupiec unconditional coverage test.

    Null hypothesis
    ---------------
    The observed VaR exception probability equals the
    theoretically expected probability:

        H0: p_observed = 1 - confidence_level

    Alternative
    -----------
    The exception probability differs from the expected rate.

    Likelihood-ratio statistic:

        LR_uc = -2 log(L0 / L1)

    where L0 uses the expected exception probability and
    L1 uses the observed exception probability.
    """

    _validate_confidence_level(confidence_level)
    _validate_significance_level(significance_level)

    data = prepare_var_backtest(
        actual_returns,
        var_losses,
    )

    expected_exception_rate = 1.0 - confidence_level
    observed_exception_rate = data.exception_rate

    lr_statistic = _calculate_lr_uc(
        exception_count=data.exception_count,
        observation_count=data.observation_count,
        expected_probability=expected_exception_rate,
    )

    p_value = float(
        chi2.sf(
            lr_statistic,
            df=1,
        )
    )

    reject_null = p_value < significance_level

    return KupiecResult(
        confidence_level=confidence_level,
        expected_exception_rate=expected_exception_rate,
        observed_exception_rate=observed_exception_rate,
        observation_count=data.observation_count,
        exception_count=data.exception_count,
        lr_statistic=lr_statistic,
        p_value=p_value,
        degrees_of_freedom=1,
        reject_null=reject_null,
    )


def kupiec_95(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    *,
    significance_level: float = 0.05,
) -> KupiecResult:
    """Run the Kupiec test for 95% VaR."""

    return kupiec_unconditional_coverage(
        actual_returns,
        var_losses,
        confidence_level=0.95,
        significance_level=significance_level,
    )


def kupiec_99(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    *,
    significance_level: float = 0.05,
) -> KupiecResult:
    """Run the Kupiec test for 99% VaR."""

    return kupiec_unconditional_coverage(
        actual_returns,
        var_losses,
        confidence_level=0.99,
        significance_level=significance_level,
    )


def _calculate_lr_uc(
    *,
    exception_count: int,
    observation_count: int,
    expected_probability: float,
) -> float:
    """
    Calculate the Kupiec likelihood-ratio statistic.

    Uses log-likelihood form with explicit boundary handling
    for zero or complete exception counts.
    """

    x = exception_count
    n = observation_count
    p = expected_probability

    observed_probability = x / n

    log_likelihood_null = (
        _binomial_log_likelihood(
            successes=x,
            trials=n,
            probability=p,
        )
    )

    log_likelihood_alternative = (
        _binomial_log_likelihood(
            successes=x,
            trials=n,
            probability=observed_probability,
        )
    )

    statistic = -2.0 * (
        log_likelihood_null
        - log_likelihood_alternative
    )

    return float(
        max(
            statistic,
            0.0,
        )
    )


def _binomial_log_likelihood(
    *,
    successes: int,
    trials: int,
    probability: float,
) -> float:
    """Calculate a Bernoulli/binomial log-likelihood."""

    failures = trials - successes

    if probability <= 0.0:
        if successes > 0:
            return -np.inf
        return 0.0

    if probability >= 1.0:
        if failures > 0:
            return -np.inf
        return 0.0

    return float(
        successes * np.log(probability)
        + failures * np.log1p(-probability)
    )


def _validate_confidence_level(
    confidence_level: float,
) -> None:
    if not np.isfinite(confidence_level):
        raise KupiecTestError(
            "confidence_level must be finite."
        )

    if not 0.0 < confidence_level < 1.0:
        raise KupiecTestError(
            "confidence_level must be between 0 and 1."
        )


def _validate_significance_level(
    significance_level: float,
) -> None:
    if not np.isfinite(significance_level):
        raise KupiecTestError(
            "significance_level must be finite."
        )

    if not 0.0 < significance_level < 1.0:
        raise KupiecTestError(
            "significance_level must be between 0 and 1."
        )