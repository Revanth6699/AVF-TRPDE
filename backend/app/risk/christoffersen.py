from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from backend.app.risk.backtesting import (
    prepare_var_backtest,
)
from backend.app.risk.kupiec import (
    _validate_confidence_level,
    _validate_significance_level,
    _calculate_lr_uc,
)


class ChristoffersenTestError(ValueError):
    """Raised when the Christoffersen test cannot be calculated."""


@dataclass(frozen=True)
class TransitionCounts:
    """Transition counts for consecutive VaR exceptions."""

    n00: int
    n01: int
    n10: int
    n11: int


@dataclass(frozen=True)
class ChristoffersenResult:
    """Result of the Christoffersen independence test."""

    confidence_level: float
    observation_count: int
    exception_count: int

    n00: int
    n01: int
    n10: int
    n11: int

    transition_probability_0: float
    transition_probability_1: float

    lr_independence: float
    independence_p_value: float

    lr_unconditional_coverage: float
    unconditional_coverage_p_value: float

    lr_conditional_coverage: float
    conditional_coverage_p_value: float

    degrees_of_freedom_independence: int
    degrees_of_freedom_conditional_coverage: int

    reject_independence_null: bool
    reject_conditional_coverage_null: bool


def christoffersen_test(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    confidence_level: float,
    *,
    significance_level: float = 0.05,
) -> ChristoffersenResult:
    """
    Perform Christoffersen independence and conditional
    coverage tests.

    Independence null hypothesis
    ----------------------------
    VaR exceptions occur independently over time.

    Conditional coverage
    ---------------------
    Combines:

        1. Unconditional coverage
        2. Independence

    Therefore:

        LR_cc = LR_uc + LR_ind

    with two degrees of freedom.
    """

    _validate_confidence_level(confidence_level)
    _validate_significance_level(significance_level)

    data = prepare_var_backtest(
        actual_returns,
        var_losses,
    )

    if data.observation_count < 2:
        raise ChristoffersenTestError(
            "At least two observations are required "
            "for the Christoffersen test."
        )

    transitions = calculate_transition_counts(
        data.exceptions,
    )

    lr_independence = _calculate_lr_independence(
        transitions,
    )

    independence_p_value = float(
        chi2.sf(
            lr_independence,
            df=1,
        )
    )

    expected_exception_rate = (
        1.0 - confidence_level
    )

    lr_unconditional_coverage = _calculate_lr_uc(
        exception_count=data.exception_count,
        observation_count=data.observation_count,
        expected_probability=expected_exception_rate,
    )

    unconditional_coverage_p_value = float(
        chi2.sf(
            lr_unconditional_coverage,
            df=1,
        )
    )

    lr_conditional_coverage = (
        lr_unconditional_coverage
        + lr_independence
    )

    conditional_coverage_p_value = float(
        chi2.sf(
            lr_conditional_coverage,
            df=2,
        )
    )

    reject_independence_null = (
        independence_p_value < significance_level
    )

    reject_conditional_coverage_null = (
        conditional_coverage_p_value
        < significance_level
    )

    transition_probability_0 = _transition_probability(
        transitions.n01,
        transitions.n00,
    )

    transition_probability_1 = _transition_probability(
        transitions.n11,
        transitions.n10,
    )

    return ChristoffersenResult(
        confidence_level=confidence_level,
        observation_count=data.observation_count,
        exception_count=data.exception_count,
        n00=transitions.n00,
        n01=transitions.n01,
        n10=transitions.n10,
        n11=transitions.n11,
        transition_probability_0=transition_probability_0,
        transition_probability_1=transition_probability_1,
        lr_independence=lr_independence,
        independence_p_value=independence_p_value,
        lr_unconditional_coverage=lr_unconditional_coverage,
        unconditional_coverage_p_value=(
            unconditional_coverage_p_value
        ),
        lr_conditional_coverage=lr_conditional_coverage,
        conditional_coverage_p_value=(
            conditional_coverage_p_value
        ),
        degrees_of_freedom_independence=1,
        degrees_of_freedom_conditional_coverage=2,
        reject_independence_null=(
            reject_independence_null
        ),
        reject_conditional_coverage_null=(
            reject_conditional_coverage_null
        ),
    )


def christoffersen_95(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    *,
    significance_level: float = 0.05,
) -> ChristoffersenResult:
    """Run Christoffersen tests for 95% VaR."""

    return christoffersen_test(
        actual_returns,
        var_losses,
        confidence_level=0.95,
        significance_level=significance_level,
    )


def christoffersen_99(
    actual_returns: np.ndarray,
    var_losses: np.ndarray,
    *,
    significance_level: float = 0.05,
) -> ChristoffersenResult:
    """Run Christoffersen tests for 99% VaR."""

    return christoffersen_test(
        actual_returns,
        var_losses,
        confidence_level=0.99,
        significance_level=significance_level,
    )


def calculate_transition_counts(
    exceptions: np.ndarray,
) -> TransitionCounts:
    """
    Calculate consecutive exception transition counts.

    States:

        0 = no VaR exception
        1 = VaR exception

    Counts:

        n00: 0 -> 0
        n01: 0 -> 1
        n10: 1 -> 0
        n11: 1 -> 1
    """

    states = np.asarray(
        exceptions,
        dtype=bool,
    )

    if states.ndim != 1:
        raise ChristoffersenTestError(
            "exceptions must be one-dimensional."
        )

    if states.size < 2:
        raise ChristoffersenTestError(
            "At least two exception observations are required."
        )

    previous = states[:-1]
    current = states[1:]

    n00 = int(
        np.sum(
            (~previous) & (~current)
        )
    )

    n01 = int(
        np.sum(
            (~previous) & current
        )
    )

    n10 = int(
        np.sum(
            previous & (~current)
        )
    )

    n11 = int(
        np.sum(
            previous & current
        )
    )

    return TransitionCounts(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
    )


def _calculate_lr_independence(
    transitions: TransitionCounts,
) -> float:
    """
    Calculate the Christoffersen independence
    likelihood-ratio statistic.
    """

    n00 = transitions.n00
    n01 = transitions.n01
    n10 = transitions.n10
    n11 = transitions.n11

    total_transitions = (
        n00 + n01 + n10 + n11
    )

    if total_transitions <= 0:
        raise ChristoffersenTestError(
            "No transitions are available."
        )

    total_exceptions_as_current = (
        n01 + n11
    )

    total_previous_exceptions = (
        n10 + n11
    )

    pi = (
        total_exceptions_as_current
        / total_transitions
    )

    pi_0 = _transition_probability(
        n01,
        n00,
    )

    pi_1 = _transition_probability(
        n11,
        n10,
    )

    log_likelihood_independent = (
        _transition_log_likelihood(
            n00=n00,
            n01=n01,
            n10=n10,
            n11=n11,
            probability=pi,
        )
    )

    log_likelihood_conditional = (
        _transition_log_likelihood_conditional(
            n00=n00,
            n01=n01,
            n10=n10,
            n11=n11,
            pi_0=pi_0,
            pi_1=pi_1,
        )
    )

    statistic = -2.0 * (
        log_likelihood_independent
        - log_likelihood_conditional
    )

    return float(
        max(
            statistic,
            0.0,
        )
    )


def _transition_log_likelihood(
    *,
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    probability: float,
) -> float:
    """Log-likelihood under independent exceptions."""

    count_zero = n00 + n10
    count_one = n01 + n11

    return _bernoulli_log_likelihood(
        zero_count=count_zero,
        one_count=count_one,
        probability=probability,
    )


def _transition_log_likelihood_conditional(
    *,
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    pi_0: float,
    pi_1: float,
) -> float:
    """Log-likelihood under the first-order Markov model."""

    return (
        _bernoulli_log_likelihood(
            zero_count=n00,
            one_count=n01,
            probability=pi_0,
        )
        + _bernoulli_log_likelihood(
            zero_count=n10,
            one_count=n11,
            probability=pi_1,
        )
    )


def _bernoulli_log_likelihood(
    *,
    zero_count: int,
    one_count: int,
    probability: float,
) -> float:
    """Numerically safe Bernoulli log-likelihood."""

    if probability <= 0.0:
        if one_count > 0:
            return -np.inf
        return 0.0

    if probability >= 1.0:
        if zero_count > 0:
            return -np.inf
        return 0.0

    return float(
        zero_count * np.log1p(-probability)
        + one_count * np.log(probability)
    )


def _transition_probability(
    numerator: int,
    denominator: int,
) -> float:
    """Calculate a transition probability safely."""

    total = numerator + denominator

    if total == 0:
        return 0.0

    return float(
        numerator / total
    )