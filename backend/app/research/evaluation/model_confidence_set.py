from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


class ModelConfidenceSetError(ValueError):
    """Raised when MCS inputs or configuration are invalid."""


@dataclass(frozen=True)
class MCSConfig:
    """Configuration for the Model Confidence Set procedure."""

    alpha: float = 0.05
    bootstrap_samples: int = 1000
    block_length: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class MCSStep:
    """Record of one model-elimination step."""

    models: tuple[str, ...]
    eliminated_model: str
    test_statistic: float
    p_value: float


@dataclass(frozen=True)
class MCSResult:
    """Result returned by the Model Confidence Set procedure."""

    included_models: tuple[str, ...]
    eliminated_models: tuple[str, ...]
    elimination_order: tuple[str, ...]
    p_values: dict[str, float]
    test_statistics: dict[str, float]
    steps: tuple[MCSStep, ...]
    observations: int
    alpha: float


def calculate_mcs(
    losses: pd.DataFrame | np.ndarray,
    *,
    model_names: Sequence[str] | None = None,
    config: MCSConfig | None = None,
) -> MCSResult:
    """
    Calculate a Model Confidence Set using the range statistic.

    Parameters
    ----------
    losses:
        Time-aligned loss observations.

        Rows represent observations through time.
        Columns represent competing models.

        Lower loss indicates better forecast performance.

    model_names:
        Optional model names when `losses` is supplied as a NumPy array.

    config:
        MCS configuration.

    Returns
    -------
    MCSResult
        Models remaining in the Model Confidence Set, eliminated models,
        elimination statistics, and elimination history.

    Notes
    -----
    The procedure starts with all candidate models.

    At each iteration:

    1. Calculate pairwise loss differentials.
    2. Calculate the range test statistic.
    3. Estimate its bootstrap null distribution.
    4. Calculate the p-value.
    5. If the equal-predictive-ability null is rejected, eliminate the
       model with the largest average pairwise loss differential.
    6. Repeat until the null is not rejected or only one model remains.

    The procedure is designed for model comparison and does not assume
    that a particular model belongs to the final confidence set.
    """

    cfg = config or MCSConfig()

    _validate_config(cfg)

    loss_matrix, names = _prepare_losses(
        losses,
        model_names=model_names,
    )

    rng = np.random.default_rng(cfg.random_state)

    current_indices = list(range(loss_matrix.shape[1]))
    eliminated_models: list[str] = []
    elimination_order: list[str] = []
    p_values: dict[str, float] = {}
    test_statistics: dict[str, float] = {}
    steps: list[MCSStep] = []

    while len(current_indices) > 1:
        current_losses = loss_matrix[:, current_indices]
        current_names = tuple(names[index] for index in current_indices)

        statistic, average_statistics = _range_test_statistic(
            current_losses
        )

        p_value = _bootstrap_p_value(
            current_losses,
            observed_statistic=statistic,
            bootstrap_samples=cfg.bootstrap_samples,
            block_length=cfg.block_length,
            rng=rng,
        )

        if p_value > cfg.alpha:
            break

        elimination_position = int(
            np.argmax(average_statistics)
        )

        eliminated_index = current_indices[elimination_position]
        eliminated_name = names[eliminated_index]

        eliminated_models.append(eliminated_name)
        elimination_order.append(eliminated_name)

        p_values[eliminated_name] = float(p_value)
        test_statistics[eliminated_name] = float(statistic)

        steps.append(
            MCSStep(
                models=current_names,
                eliminated_model=eliminated_name,
                test_statistic=float(statistic),
                p_value=float(p_value),
            )
        )

        current_indices.pop(elimination_position)

    included_models = tuple(
        names[index]
        for index in current_indices
    )

    return MCSResult(
        included_models=included_models,
        eliminated_models=tuple(eliminated_models),
        elimination_order=tuple(elimination_order),
        p_values=p_values,
        test_statistics=test_statistics,
        steps=tuple(steps),
        observations=int(loss_matrix.shape[0]),
        alpha=cfg.alpha,
    )


def calculate_mcs_from_dict(
    losses: dict[str, Sequence[float]],
    *,
    config: MCSConfig | None = None,
) -> MCSResult:
    """
    Calculate MCS from a mapping of model names to loss observations.
    """

    if not losses:
        raise ModelConfidenceSetError(
            "losses must contain at least two models."
        )

    dataframe = pd.DataFrame(losses)

    return calculate_mcs(
        dataframe,
        config=config,
    )


def _prepare_losses(
    losses: pd.DataFrame | np.ndarray,
    *,
    model_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(losses, pd.DataFrame):
        if losses.empty:
            raise ModelConfidenceSetError(
                "losses must not be empty."
            )

        dataframe = losses.copy()

        if model_names is not None:
            raise ModelConfidenceSetError(
                "model_names must not be supplied when losses is a DataFrame."
            )

        names = tuple(str(column) for column in dataframe.columns)

        values = dataframe.to_numpy(dtype=float)

    elif isinstance(losses, np.ndarray):
        if losses.ndim != 2:
            raise ModelConfidenceSetError(
                "NumPy losses must be a two-dimensional array."
            )

        values = np.asarray(losses, dtype=float)

        if model_names is None:
            names = tuple(
                f"model_{index}"
                for index in range(values.shape[1])
            )
        else:
            names = tuple(str(name) for name in model_names)

    else:
        raise TypeError(
            "losses must be a pandas.DataFrame or numpy.ndarray."
        )

    if values.ndim != 2:
        raise ModelConfidenceSetError(
            "Loss matrix must be two-dimensional."
        )

    observations, model_count = values.shape

    if observations < 2:
        raise ModelConfidenceSetError(
            "At least two observations are required."
        )

    if model_count < 2:
        raise ModelConfidenceSetError(
            "At least two competing models are required."
        )

    if len(names) != model_count:
        raise ModelConfidenceSetError(
            "The number of model names must match the number of columns."
        )

    if len(set(names)) != len(names):
        raise ModelConfidenceSetError(
            "Model names must be unique."
        )

    if any(not name.strip() for name in names):
        raise ModelConfidenceSetError(
            "Model names must not be empty."
        )

    if not np.isfinite(values).all():
        raise ModelConfidenceSetError(
            "Loss matrix contains NaN or infinite values."
        )

    return values, names


def _validate_config(config: MCSConfig) -> None:
    if not 0.0 < config.alpha < 1.0:
        raise ModelConfidenceSetError(
            "alpha must be strictly between 0 and 1."
        )

    if config.bootstrap_samples < 100:
        raise ModelConfidenceSetError(
            "bootstrap_samples must be at least 100."
        )

    if config.block_length < 1:
        raise ModelConfidenceSetError(
            "block_length must be at least 1."
        )

    if config.random_state < 0:
        raise ModelConfidenceSetError(
            "random_state must be non-negative."
        )


def _pairwise_loss_differentials(
    losses: np.ndarray,
) -> np.ndarray:
    """
    Return pairwise loss differentials.

    d[i, j, t] = loss[i, t] - loss[j, t]
    """

    return (
        losses[:, :, None]
        - losses[:, None, :]
    )


def _range_test_statistic(
    losses: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Calculate the MCS range statistic.

    For each model i:

        t_i = max_j |mean(d_ij)| / se(d_ij)

    The overall range statistic is:

        T_R = max_i t_i

    The returned average_statistics are the model-level statistics used
    to determine which model is eliminated after rejection.
    """

    observations = losses.shape[0]

    differentials = _pairwise_loss_differentials(losses)

    mean_differentials = differentials.mean(axis=0)

    centered = (
        differentials
        - mean_differentials[None, :, :]
    )

    variance = np.sum(
        centered**2,
        axis=0,
    ) / max(observations - 1, 1)

    standard_error = np.sqrt(
        variance / observations
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        t_statistics = np.divide(
            np.abs(mean_differentials),
            standard_error,
            out=np.zeros_like(mean_differentials),
            where=standard_error > 0,
        )

    np.fill_diagonal(
        t_statistics,
        0.0,
    )

    model_statistics = np.max(
        t_statistics,
        axis=1,
    )

    range_statistic = float(
        np.max(model_statistics)
    )

    return range_statistic, model_statistics


def _bootstrap_p_value(
    losses: np.ndarray,
    *,
    observed_statistic: float,
    bootstrap_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate the MCS range-statistic p-value using a moving-block bootstrap.

    The bootstrap samples are centered under the equal-predictive-ability
    null by subtracting each model's sample mean loss.
    """

    observations = losses.shape[0]

    centered_losses = (
        losses
        - losses.mean(axis=0, keepdims=True)
    )

    bootstrap_statistics = np.empty(
        bootstrap_samples,
        dtype=float,
    )

    for bootstrap_index in range(bootstrap_samples):
        indices = _moving_block_indices(
            observations=observations,
            block_length=block_length,
            rng=rng,
        )

        sample = centered_losses[indices]

        bootstrap_statistics[bootstrap_index], _ = (
            _range_test_statistic(sample)
        )

    exceedances = np.count_nonzero(
        bootstrap_statistics >= observed_statistic
    )

    # Add-one correction prevents a zero p-value.
    return float(
        (exceedances + 1)
        / (bootstrap_samples + 1)
    )


def _moving_block_indices(
    *,
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap indices using a moving-block bootstrap.
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

    if max_start <= 0:
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

    indices = np.concatenate(
        [
            np.arange(
                start,
                start + effective_block_length,
            )
            for start in starts
        ]
    )

    return indices[:observations]