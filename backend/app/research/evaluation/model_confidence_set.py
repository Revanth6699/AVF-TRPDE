from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ModelConfidenceSetError(ValueError):
    """Raised when Model Confidence Set estimation fails."""


@dataclass(frozen=True)
class ModelConfidenceSetResult:
    """Result of a Model Confidence Set procedure."""

    included_models: tuple[str, ...]
    eliminated_models: tuple[str, ...]
    confidence_level: float
    observations: int
    bootstrap_samples: int
    statistic: float
    p_value: float

    def as_dict(self) -> dict[str, object]:
        """Return the result as a dictionary."""
        return {
            "included_models": self.included_models,
            "eliminated_models": self.eliminated_models,
            "confidence_level": self.confidence_level,
            "observations": self.observations,
            "bootstrap_samples": self.bootstrap_samples,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }


def model_confidence_set(
    loss_matrix: pd.DataFrame,
    *,
    confidence_level: float = 0.90,
    n_bootstrap: int = 2_000,
    block_length: int = 5,
    random_state: int | None = 42,
) -> ModelConfidenceSetResult:
    """
    Estimate a Model Confidence Set from model loss series.

    Rows represent common out-of-sample observations.
    Columns represent competing models.

    Lower loss indicates better predictive performance.

    The procedure repeatedly tests the null hypothesis that
    the remaining models have equal predictive ability.

    Models are removed only when the null is rejected.

    Args:
        loss_matrix:
            DataFrame where each column contains the loss
            sequence for one model.
        confidence_level:
            Confidence level for the MCS.
        n_bootstrap:
            Number of bootstrap samples.
        block_length:
            Moving-block bootstrap length.
        random_state:
            Random seed.

    Returns:
        ModelConfidenceSetResult.
    """
    losses = _validate_loss_matrix(loss_matrix)

    _validate_parameters(
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )

    rng = np.random.default_rng(random_state)

    active_models = list(losses.columns)
    eliminated_models: list[str] = []

    final_statistic = 0.0
    final_p_value = 1.0

    while len(active_models) > 1:
        active_losses = losses.loc[:, active_models]

        statistic, p_value, worst_model = _mcs_step(
            active_losses,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            rng=rng,
        )

        final_statistic = statistic
        final_p_value = p_value

        if p_value >= 1.0 - confidence_level:
            break

        if worst_model is None:
            break

        active_models.remove(worst_model)
        eliminated_models.append(worst_model)

    return ModelConfidenceSetResult(
        included_models=tuple(active_models),
        eliminated_models=tuple(eliminated_models),
        confidence_level=confidence_level,
        observations=len(losses),
        bootstrap_samples=n_bootstrap,
        statistic=final_statistic,
        p_value=final_p_value,
    )


def build_loss_matrix(
    actual: np.ndarray | list[float],
    forecasts: dict[str, np.ndarray | list[float]],
    *,
    loss: str = "qlike",
) -> pd.DataFrame:
    """
    Build a model-loss matrix.

    Supported losses:
        - mae
        - rmse
        - qlike
    """
    actual_array = _to_array(
        actual,
        "actual",
    )

    if not forecasts:
        raise ModelConfidenceSetError(
            "At least two model forecasts are required."
        )

    if len(forecasts) < 2:
        raise ModelConfidenceSetError(
            "Model Confidence Set requires at least two models."
        )

    loss_name = loss.strip().lower()

    if loss_name not in {
        "mae",
        "rmse",
        "qlike",
    }:
        raise ModelConfidenceSetError(
            "loss must be 'mae', 'rmse', or 'qlike'."
        )

    matrix: dict[str, np.ndarray] = {}

    for model_name, forecast in forecasts.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ModelConfidenceSetError(
                "Model names must be non-empty strings."
            )

        forecast_array = _to_array(
            forecast,
            f"forecast[{model_name}]",
        )

        if len(actual_array) != len(forecast_array):
            raise ModelConfidenceSetError(
                "Actual observations and every forecast "
                "must have the same length."
            )

        matrix[model_name] = _pointwise_loss(
            actual_array,
            forecast_array,
            loss_name,
        )

    return pd.DataFrame(
        matrix,
        index=np.arange(len(actual_array)),
    )


def _mcs_step(
    losses: pd.DataFrame,
    *,
    confidence_level: float,
    n_bootstrap: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[float, float, str | None]:
    """Perform one MCS elimination step."""

    model_names = list(losses.columns)
    observations = len(losses)
    model_count = len(model_names)

    mean_losses = losses.mean(axis=0)

    pairwise_difference = np.zeros(
        (model_count, model_count),
        dtype=float,
    )

    pairwise_t = np.zeros(
        (model_count, model_count),
        dtype=float,
    )

    for i in range(model_count):
        for j in range(model_count):
            if i == j:
                continue

            difference = (
                losses.iloc[:, i]
                - losses.iloc[:, j]
            )

            pairwise_difference[i, j] = float(
                difference.mean()
            )

            variance = _long_run_variance(
                difference.to_numpy(),
                block_length=block_length,
            )

            if variance > 0:
                pairwise_t[i, j] = (
                    pairwise_difference[i, j]
                    / np.sqrt(
                        variance / observations
                    )
                )

    statistic = float(
        np.max(
            np.abs(
                pairwise_t
            )
        )
    )

    bootstrap_statistics = np.empty(
        n_bootstrap,
        dtype=float,
    )

    loss_array = losses.to_numpy()

    for bootstrap_index in range(n_bootstrap):
        indices = _moving_block_indices(
            observations=observations,
            block_length=block_length,
            rng=rng,
        )

        sample = loss_array[indices]

        centered = (
            sample
            - sample.mean(axis=0)
        )

        bootstrap_mean = centered.mean(axis=0)

        bootstrap_max = 0.0

        for i in range(model_count):
            for j in range(model_count):
                if i == j:
                    continue

                difference = (
                    centered[:, i]
                    - centered[:, j]
                )

                standard_error = np.std(
                    difference,
                    ddof=1,
                ) / np.sqrt(observations)

                if standard_error > 0:
                    value = abs(
                        bootstrap_mean[i]
                        - bootstrap_mean[j]
                    ) / standard_error

                    bootstrap_max = max(
                        bootstrap_max,
                        float(value),
                    )

        bootstrap_statistics[
            bootstrap_index
        ] = bootstrap_max

    p_value = float(
        np.mean(
            bootstrap_statistics >= statistic
        )
    )

    worst_model = _identify_worst_model(
        losses,
        pairwise_t,
        mean_losses,
    )

    return (
        statistic,
        p_value,
        worst_model,
    )


def _identify_worst_model(
    losses: pd.DataFrame,
    pairwise_t: np.ndarray,
    mean_losses: pd.Series,
) -> str | None:
    """
    Identify the model contributing most strongly to
    the rejection statistic.

    This is an elimination step, not a final ranking.
    """
    model_names = list(losses.columns)

    if len(model_names) <= 1:
        return None

    scores = np.max(
        pairwise_t,
        axis=1,
    )

    candidate_index = int(
        np.argmax(scores)
    )

    candidate = model_names[
        candidate_index
    ]

    if scores[candidate_index] <= 0:
        candidate = str(
            mean_losses.idxmax()
        )

    return candidate


def _pointwise_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    loss: str,
) -> np.ndarray:
    """Calculate pointwise forecast losses."""

    if loss == "mae":
        return np.abs(
            actual - forecast
        )

    if loss == "rmse":
        return np.square(
            actual - forecast
        )

    if np.any(actual <= 0):
        raise ModelConfidenceSetError(
            "Actual values must be strictly positive "
            "for QLIKE."
        )

    if np.any(forecast <= 0):
        raise ModelConfidenceSetError(
            "Forecast values must be strictly positive "
            "for QLIKE."
        )

    ratio = actual / forecast

    return (
        ratio
        - np.log(ratio)
        - 1.0
    )


def _long_run_variance(
    values: np.ndarray,
    *,
    block_length: int,
) -> float:
    """Estimate long-run variance using autocovariances."""

    observations = len(values)

    if observations < 2:
        return 0.0

    centered = (
        values
        - np.mean(values)
    )

    max_lag = min(
        block_length - 1,
        observations - 1,
    )

    variance = float(
        np.mean(
            centered * centered
        )
    )

    for lag in range(1, max_lag + 1):
        covariance = float(
            np.mean(
                centered[lag:]
                * centered[:-lag]
            )
        )

        weight = 1.0 - (
            lag / (max_lag + 1)
        )

        variance += (
            2.0
            * weight
            * covariance
        )

    return max(
        variance,
        0.0,
    )


def _moving_block_indices(
    *,
    observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate indices using a moving-block bootstrap."""

    if observations <= 0:
        raise ModelConfidenceSetError(
            "Number of observations must be positive."
        )

    block_length = min(
        block_length,
        observations,
    )

    max_start = (
        observations
        - block_length
    )

    indices: list[int] = []

    while len(indices) < observations:
        start = int(
            rng.integers(
                0,
                max_start + 1,
            )
        )

        block = range(
            start,
            start + block_length,
        )

        indices.extend(block)

    return np.asarray(
        indices[:observations],
        dtype=int,
    )


def _validate_loss_matrix(
    loss_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and copy the loss matrix."""

    if not isinstance(
        loss_matrix,
        pd.DataFrame,
    ):
        raise TypeError(
            "loss_matrix must be a pandas.DataFrame."
        )

    if loss_matrix.empty:
        raise ModelConfidenceSetError(
            "loss_matrix must not be empty."
        )

    if loss_matrix.shape[1] < 2:
        raise ModelConfidenceSetError(
            "loss_matrix must contain at least two models."
        )

    if loss_matrix.isna().any().any():
        raise ModelConfidenceSetError(
            "loss_matrix contains missing values."
        )

    values = loss_matrix.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ModelConfidenceSetError(
            "loss_matrix contains NaN or infinite values."
        )

    return loss_matrix.copy()


def _validate_parameters(
    *,
    confidence_level: float,
    n_bootstrap: int,
    block_length: int,
) -> None:
    """Validate MCS parameters."""

    if not 0.0 < confidence_level < 1.0:
        raise ModelConfidenceSetError(
            "confidence_level must be between 0 and 1."
        )

    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, int)
    ):
        raise ModelConfidenceSetError(
            "n_bootstrap must be an integer."
        )

    if n_bootstrap < 100:
        raise ModelConfidenceSetError(
            "n_bootstrap must be at least 100."
        )

    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
    ):
        raise ModelConfidenceSetError(
            "block_length must be an integer."
        )

    if block_length < 1:
        raise ModelConfidenceSetError(
            "block_length must be at least 1."
        )


def _to_array(
    values: np.ndarray | list[float],
    name: str,
) -> np.ndarray:
    """Convert input values to a validated 1-D float array."""

    if isinstance(values, np.ndarray):
        try:
            array = values.astype(
                float,
                copy=False,
            )
        except (TypeError, ValueError) as exc:
            raise ModelConfidenceSetError(
                f"{name} contains non-numeric values."
            ) from exc

    elif isinstance(values, list):
        try:
            array = np.asarray(
                values,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ModelConfidenceSetError(
                f"{name} contains non-numeric values."
            ) from exc

    else:
        raise TypeError(
            f"{name} must be a numpy array or list."
        )

    if array.ndim != 1:
        raise ModelConfidenceSetError(
            f"{name} must be one-dimensional."
        )

    if len(array) < 2:
        raise ModelConfidenceSetError(
            f"{name} must contain at least two observations."
        )

    if not np.isfinite(array).all():
        raise ModelConfidenceSetError(
            f"{name} contains NaN or infinite values."
        )

    return array