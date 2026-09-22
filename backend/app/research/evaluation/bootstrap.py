from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


class BootstrapError(ValueError):
    """Raised when bootstrap inputs or configuration are invalid."""


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap estimation."""

    samples: int = 2000
    confidence_level: float = 0.95
    block_length: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    samples: int


@dataclass(frozen=True)
class BootstrapDifference:
    """Bootstrap confidence interval for a paired difference."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    samples: int


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapCI:
    """
    Estimate the mean and its bootstrap confidence interval.
    """

    data = _validate_vector(values)
    cfg = config or BootstrapConfig()

    _validate_config(cfg)

    statistic = lambda sample: float(np.mean(sample))

    estimate = statistic(data)

    bootstrap_statistics = _bootstrap_statistics(
        data,
        statistic=statistic,
        config=cfg,
    )

    lower, upper = _percentile_interval(
        bootstrap_statistics,
        confidence_level=cfg.confidence_level,
    )

    return BootstrapCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=cfg.confidence_level,
        samples=cfg.samples,
    )


def bootstrap_metric(
    values: Sequence[float] | np.ndarray,
    metric: Callable[[np.ndarray], float],
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapCI:
    """
    Bootstrap an arbitrary scalar metric.

    The metric must accept a one-dimensional NumPy array and return
    one finite scalar value.
    """

    data = _validate_vector(values)
    cfg = config or BootstrapConfig()

    _validate_config(cfg)

    estimate = _evaluate_metric(
        metric,
        data,
    )

    bootstrap_statistics = _bootstrap_statistics(
        data,
        statistic=metric,
        config=cfg,
    )

    lower, upper = _percentile_interval(
        bootstrap_statistics,
        confidence_level=cfg.confidence_level,
    )

    return BootstrapCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=cfg.confidence_level,
        samples=cfg.samples,
    )


def bootstrap_difference(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapDifference:
    """
    Bootstrap the mean paired difference between two series.

    The two series must be time-aligned and have identical lengths.

    The estimated difference is:

        mean(A - B)
    """

    data_a = _validate_vector(values_a)
    data_b = _validate_vector(values_b)

    if len(data_a) != len(data_b):
        raise BootstrapError(
            "values_a and values_b must contain the same number "
            "of observations."
        )

    cfg = config or BootstrapConfig()

    _validate_config(cfg)

    differences = data_a - data_b

    result = bootstrap_mean(
        differences,
        config=cfg,
    )

    return BootstrapDifference(
        estimate=result.estimate,
        lower=result.lower,
        upper=result.upper,
        confidence_level=result.confidence_level,
        samples=result.samples,
    )


def bootstrap_metric_difference(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    metric: Callable[[np.ndarray], float],
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapDifference:
    """
    Bootstrap the paired difference between two model metrics.

    Each bootstrap replicate resamples the same time indices for both
    series, preserving their paired structure.
    """

    data_a = _validate_vector(values_a)
    data_b = _validate_vector(values_b)

    if len(data_a) != len(data_b):
        raise BootstrapError(
            "values_a and values_b must contain the same number "
            "of observations."
        )

    cfg = config or BootstrapConfig()

    _validate_config(cfg)

    estimate = (
        _evaluate_metric(metric, data_a)
        - _evaluate_metric(metric, data_b)
    )

    rng = np.random.default_rng(
        cfg.random_state
    )

    bootstrap_statistics = np.empty(
        cfg.samples,
        dtype=float,
    )

    for index in range(cfg.samples):
        indices = _moving_block_indices(
            observations=len(data_a),
            block_length=cfg.block_length,
            rng=rng,
        )

        metric_a = _evaluate_metric(
            metric,
            data_a[indices],
        )

        metric_b = _evaluate_metric(
            metric,
            data_b[indices],
        )

        bootstrap_statistics[index] = (
            metric_a - metric_b
        )

    lower, upper = _percentile_interval(
        bootstrap_statistics,
        confidence_level=cfg.confidence_level,
    )

    return BootstrapDifference(
        estimate=float(estimate),
        lower=lower,
        upper=upper,
        confidence_level=cfg.confidence_level,
        samples=cfg.samples,
    )


def bootstrap_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
    config: BootstrapConfig | None = None,
) -> BootstrapCI:
    """
    Bootstrap an annualized Sharpe ratio.

    Returns are assumed to be periodic returns.

    Sharpe:

        sqrt(annualization_factor)
        * mean(excess_return)
        / std(excess_return)
    """

    data = _validate_vector(returns)
    cfg = config or BootstrapConfig()

    _validate_config(cfg)

    if annualization_factor <= 0:
        raise BootstrapError(
            "annualization_factor must be greater than zero."
        )

    def sharpe(sample: np.ndarray) -> float:
        excess = sample - risk_free_rate
        volatility = float(
            np.std(
                excess,
                ddof=1,
            )
        )

        if volatility <= 0:
            return 0.0

        return float(
            np.sqrt(annualization_factor)
            * np.mean(excess)
            / volatility
        )

    estimate = _evaluate_metric(
        sharpe,
        data,
    )

    bootstrap_statistics = _bootstrap_statistics(
        data,
        statistic=sharpe,
        config=cfg,
    )

    lower, upper = _percentile_interval(
        bootstrap_statistics,
        confidence_level=cfg.confidence_level,
    )

    return BootstrapCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=cfg.confidence_level,
        samples=cfg.samples,
    )


def _bootstrap_statistics(
    data: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float],
    config: BootstrapConfig,
) -> np.ndarray:
    """
    Generate bootstrap statistics using moving blocks.
    """

    rng = np.random.default_rng(
        config.random_state
    )

    bootstrap_statistics = np.empty(
        config.samples,
        dtype=float,
    )

    for index in range(config.samples):
        indices = _moving_block_indices(
            observations=len(data),
            block_length=config.block_length,
            rng=rng,
        )

        bootstrap_statistics[index] = _evaluate_metric(
            statistic,
            data[indices],
        )

    return bootstrap_statistics


def _moving_block_indices(
    *,
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate time-series bootstrap indices using moving blocks.
    """

    effective_block_length = min(
        block_length,
        observations,
    )

    block_count = int(
        np.ceil(
            observations
            / effective_block_length
        )
    )

    max_start = (
        observations
        - effective_block_length
    )

    if max_start == 0:
        starts = np.zeros(
            block_count,
            dtype=int,
        )
    else:
        starts = rng.integers(
            0,
            max_start + 1,
            size=block_count,
        )

    blocks = [
        np.arange(
            start,
            start + effective_block_length,
        )
        for start in starts
    ]

    indices = np.concatenate(blocks)

    return indices[:observations]


def _percentile_interval(
    values: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level

    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (
        1.0 - alpha / 2.0
    )

    lower, upper = np.percentile(
        values,
        [lower_percentile, upper_percentile],
    )

    return float(lower), float(upper)


def _evaluate_metric(
    metric: Callable[[np.ndarray], float],
    values: np.ndarray,
) -> float:
    try:
        result = metric(values)
    except Exception as exc:
        raise BootstrapError(
            "Metric evaluation failed."
        ) from exc

    try:
        scalar = float(result)
    except (TypeError, ValueError) as exc:
        raise BootstrapError(
            "Metric must return a scalar numeric value."
        ) from exc

    if not np.isfinite(scalar):
        raise BootstrapError(
            "Metric returned a non-finite value."
        )

    return scalar


def _validate_vector(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    try:
        data = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapError(
            "Values must be numeric."
        ) from exc

    if data.ndim != 1:
        raise BootstrapError(
            "Values must be one-dimensional."
        )

    if len(data) < 2:
        raise BootstrapError(
            "At least two observations are required."
        )

    if not np.isfinite(data).all():
        raise BootstrapError(
            "Values must not contain NaN or infinite values."
        )

    return data


def _validate_config(
    config: BootstrapConfig,
) -> None:
    if config.samples < 100:
        raise BootstrapError(
            "samples must be at least 100."
        )

    if not 0.0 < config.confidence_level < 1.0:
        raise BootstrapError(
            "confidence_level must be strictly between 0 and 1."
        )

    if config.block_length < 1:
        raise BootstrapError(
            "block_length must be at least 1."
        )

    if config.random_state < 0:
        raise BootstrapError(
            "random_state must be non-negative."
        )