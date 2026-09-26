from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class MCSError(ValueError):
    """Raised when Model Confidence Set inputs are invalid."""


@dataclass(frozen=True)
class MCSElimination:
    """Record of one MCS elimination step."""

    model: str
    p_value: float
    test_statistic: float


@dataclass(frozen=True)
class MCSResult:
    """Result of the Model Confidence Set procedure."""

    included_models: tuple[str, ...]
    eliminated_models: tuple[str, ...]
    alpha: float
    p_value: float
    test_statistic: float
    bootstrap_samples: int
    block_length: int
    elimination_history: tuple[MCSElimination, ...]

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha

    @property
    def model_count(self) -> int:
        return len(self.included_models)

    @property
    def is_singleton(self) -> bool:
        return self.model_count == 1


def model_confidence_set(
    losses: np.ndarray,
    models: Sequence[str] | None = None,
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    random_state: int | None = 42,
) -> MCSResult:
    """
    Compute a bootstrap Model Confidence Set.

    Parameters
    ----------
    losses:
        Two-dimensional array with shape (n_observations, n_models).
        Each column contains the OOS loss series for one model.
        Lower loss is better.

    models:
        Model names corresponding to the columns of ``losses``.
        If omitted, names are generated as Model_0, Model_1, ...

    alpha:
        MCS significance level. For a 90% MCS use alpha=0.10.

    n_bootstrap:
        Number of bootstrap replications.

    block_length:
        Moving-block bootstrap block length. If omitted, a data-dependent
        default of ceil(sqrt(n_observations)) is used.

    random_state:
        Random seed.

    Returns
    -------
    MCSResult
        Statistically supported model set and elimination history.

    Notes
    -----
    The procedure uses a moving-block bootstrap so that temporal dependence
    in the out-of-sample loss sequence is not treated as iid noise.

    At each elimination step:

    1. Compute the active-model loss means.
    2. Construct the maximum studentized pairwise loss-difference statistic.
    3. Center active losses under the null of equal predictive ability.
    4. Generate moving-block bootstrap samples.
    5. Estimate the MCS p-value.
    6. If the null is rejected, eliminate the active model with the largest
       average loss.
    7. Continue until the equal-predictive-ability null is not rejected.

    This implementation is intended for the project's out-of-sample
    forecast-loss comparison layer.
    """
    matrix, model_names = _validate_inputs(
        losses,
        models,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )

    n_observations, n_models = matrix.shape

    if block_length is None:
        block_length = max(1, int(np.ceil(np.sqrt(n_observations))))

    rng = np.random.default_rng(random_state)

    active = list(range(n_models))
    eliminated: list[str] = []
    history: list[MCSElimination] = []

    final_p_value = 1.0
    final_statistic = 0.0

    while len(active) > 1:
        active_losses = matrix[:, active]

        statistic, pair_se = _observed_test_statistic(active_losses)

        bootstrap_statistics = _bootstrap_test_statistics(
            active_losses,
            pair_se,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            rng=rng,
        )

        p_value = float(
            (1.0 + np.sum(bootstrap_statistics >= statistic))
            / (n_bootstrap + 1.0)
        )

        final_p_value = p_value
        final_statistic = statistic

        if p_value > alpha:
            break

        means = np.mean(active_losses, axis=0)
        worst_position = int(np.argmax(means))
        worst_index = active.pop(worst_position)
        worst_name = model_names[worst_index]

        history.append(
            MCSElimination(
                model=worst_name,
                p_value=p_value,
                test_statistic=statistic,
            )
        )

        eliminated.append(worst_name)

    included = tuple(model_names[index] for index in active)

    return MCSResult(
        included_models=included,
        eliminated_models=tuple(eliminated),
        alpha=float(alpha),
        p_value=float(final_p_value),
        test_statistic=float(final_statistic),
        bootstrap_samples=int(n_bootstrap),
        block_length=int(block_length),
        elimination_history=tuple(history),
    )


def _validate_inputs(
    losses: np.ndarray,
    models: Sequence[str] | None,
    *,
    alpha: float,
    n_bootstrap: int,
    block_length: int | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(losses, dtype=float)

    if values.ndim != 2:
        raise MCSError("losses must be a two-dimensional array.")

    n_observations, n_models = values.shape

    if n_observations < 10:
        raise MCSError("At least 10 observations are required for MCS.")

    if n_models < 2:
        raise MCSError("At least two models are required for MCS.")

    if not np.isfinite(values).all():
        raise MCSError("losses must contain only finite values.")

    if alpha <= 0.0 or alpha >= 1.0:
        raise MCSError("alpha must be strictly between 0 and 1.")

    if n_bootstrap < 100:
        raise MCSError("n_bootstrap must be at least 100.")

    if block_length is not None:
        if block_length < 1:
            raise MCSError("block_length must be at least 1.")
        if block_length > n_observations:
            raise MCSError(
                "block_length cannot exceed the number of observations."
            )

    if models is None:
        names = tuple(f"Model_{i}" for i in range(n_models))
    else:
        names = tuple(str(name) for name in models)

        if len(names) != n_models:
            raise MCSError(
                "Number of model names must match the loss columns."
            )

        if any(not name.strip() for name in names):
            raise MCSError("Model names must be non-empty.")

        if len(set(names)) != len(names):
            raise MCSError("Model names must be unique.")

    return values, names


def _observed_test_statistic(
    losses: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Maximum absolute studentized pairwise mean loss difference.
    """
    n_observations, n_models = losses.shape

    means = np.mean(losses, axis=0)

    pairwise_se = np.zeros((n_models, n_models), dtype=float)
    statistic = 0.0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            difference = losses[:, i] - losses[:, j]

            standard_error = _block_robust_scale(difference)

            if standard_error <= np.finfo(float).eps:
                if np.isclose(means[i], means[j]):
                    t_value = 0.0
                else:
                    t_value = np.inf
            else:
                t_value = (
                    np.sqrt(n_observations)
                    * abs(means[i] - means[j])
                    / standard_error
                )

            pairwise_se[i, j] = standard_error
            pairwise_se[j, i] = standard_error

            statistic = max(statistic, float(t_value))

    return float(statistic), pairwise_se


def _bootstrap_test_statistics(
    losses: np.ndarray,
    pairwise_se: np.ndarray,
    *,
    n_bootstrap: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate null bootstrap maximum studentized statistics.

    Each model's loss series is centered by its own sample mean, enforcing
    the null hypothesis of equal expected predictive loss while retaining
    temporal dependence through moving-block resampling.
    """
    n_observations, n_models = losses.shape

    centered = losses - np.mean(losses, axis=0, keepdims=True)

    result = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        indices = _moving_block_indices(
            n_observations,
            block_length,
            rng,
        )

        sample = centered[indices]

        means = np.mean(sample, axis=0)
        statistic = 0.0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                standard_error = pairwise_se[i, j]

                if standard_error <= np.finfo(float).eps:
                    if np.isclose(means[i], means[j]):
                        t_value = 0.0
                    else:
                        t_value = np.inf
                else:
                    t_value = (
                        np.sqrt(n_observations)
                        * abs(means[i] - means[j])
                        / standard_error
                    )

                statistic = max(statistic, float(t_value))

        result[b] = statistic

    return result


def _moving_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap indices using a moving-block bootstrap.
    """
    n_blocks = int(np.ceil(n_observations / block_length))

    starts = rng.integers(
        0,
        n_observations - block_length + 1,
        size=n_blocks,
    )

    indices = np.concatenate(
        [
            np.arange(start, start + block_length, dtype=int)
            for start in starts
        ]
    )

    return indices[:n_observations]


def _block_robust_scale(values: np.ndarray) -> float:
    """
    Estimate the standard deviation of the sample mean using a
    simple autocorrelation-aware variance estimate.

    The estimator uses the Bartlett/Newey-West form with a bandwidth
    selected from the square-root sample-size rule.
    """
    x = np.asarray(values, dtype=float)

    n = x.size

    if n < 2:
        return 0.0

    centered = x - np.mean(x)

    gamma0 = float(np.mean(centered * centered))

    bandwidth = max(1, int(np.floor(np.sqrt(n))))

    long_run_variance = gamma0

    for lag in range(1, bandwidth + 1):
        covariance = float(
            np.mean(
                centered[lag:]
                * centered[:-lag]
            )
        )

        weight = 1.0 - lag / (bandwidth + 1.0)

        long_run_variance += 2.0 * weight * covariance

    long_run_variance = max(
        long_run_variance,
        np.finfo(float).eps,
    )

    return float(np.sqrt(long_run_variance / n))


def mcs(
    losses: np.ndarray,
    models: Sequence[str] | None = None,
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    random_state: int | None = 42,
) -> MCSResult:
    """Alias for :func:`model_confidence_set`."""
    return model_confidence_set(
        losses,
        models,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        random_state=random_state,
    )


__all__ = [
    "MCSError",
    "MCSElimination",
    "MCSResult",
    "mcs",
    "model_confidence_set",
]