from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ForecastMetricError(ValueError):
    """Raised when forecast metrics cannot be calculated."""


@dataclass(frozen=True)
class ForecastMetrics:
    """Forecast evaluation metrics."""

    mae: float
    rmse: float
    qlike: float


def calculate_mae(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Error.

    MAE = mean(|y - y_hat|)
    """

    y_true, y_pred = _prepare_inputs(
        actual,
        forecast,
    )

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def calculate_rmse(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> float:
    """
    Calculate Root Mean Squared Error.

    RMSE = sqrt(mean((y - y_hat)^2))
    """

    y_true, y_pred = _prepare_inputs(
        actual,
        forecast,
    )

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def calculate_qlike(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> float:
    """
    Calculate QLIKE loss.

    QLIKE = mean(
        y / y_hat
        - log(y / y_hat)
        - 1
    )

    Actual and forecast volatility/variance inputs must be
    strictly positive.
    """

    y_true, y_pred = _prepare_inputs(
        actual,
        forecast,
    )

    if (y_true <= 0).any():
        raise ForecastMetricError(
            "QLIKE requires actual values greater than zero."
        )

    if (y_pred <= 0).any():
        raise ForecastMetricError(
            "QLIKE requires forecast values greater than zero."
        )

    ratio = y_true / y_pred

    losses = (
        ratio
        - np.log(ratio)
        - 1.0
    )

    if not np.isfinite(losses).all():
        raise ForecastMetricError(
            "QLIKE calculation produced non-finite values."
        )

    return float(
        np.mean(losses)
    )


def calculate_forecast_metrics(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> ForecastMetrics:
    """Calculate MAE, RMSE, and QLIKE together."""

    y_true, y_pred = _prepare_inputs(
        actual,
        forecast,
    )

    return ForecastMetrics(
        mae=calculate_mae(
            y_true,
            y_pred,
        ),
        rmse=calculate_rmse(
            y_true,
            y_pred,
        ),
        qlike=calculate_qlike(
            y_true,
            y_pred,
        ),
    )


def evaluate_forecast_dataframe(
    dataframe: pd.DataFrame,
    *,
    actual_column: str,
    forecast_column: str,
) -> ForecastMetrics:
    """
    Evaluate a forecast directly from a dataframe.

    Rows with missing actual or forecast values are rejected rather
    than silently removed, because silent removal can hide problems
    in an out-of-sample evaluation.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise ForecastMetricError(
            "dataframe must not be empty."
        )

    required_columns = (
        actual_column,
        forecast_column,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ForecastMetricError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    actual = dataframe[
        actual_column
    ]

    forecast = dataframe[
        forecast_column
    ]

    return calculate_forecast_metrics(
        actual,
        forecast,
    )


def _prepare_inputs(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert metric inputs."""

    y_true = _to_numpy(
        actual,
        "actual",
    )

    y_pred = _to_numpy(
        forecast,
        "forecast",
    )

    if y_true.ndim != 1:
        raise ForecastMetricError(
            "actual must be one-dimensional."
        )

    if y_pred.ndim != 1:
        raise ForecastMetricError(
            "forecast must be one-dimensional."
        )

    if len(y_true) != len(y_pred):
        raise ForecastMetricError(
            "actual and forecast must contain the same "
            "number of observations."
        )

    if len(y_true) == 0:
        raise ForecastMetricError(
            "actual and forecast must not be empty."
        )

    if not np.isfinite(y_true).all():
        raise ForecastMetricError(
            "actual contains non-finite values."
        )

    if not np.isfinite(y_pred).all():
        raise ForecastMetricError(
            "forecast contains non-finite values."
        )

    return y_true, y_pred


def _to_numpy(
    values: pd.Series | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert supported input types to a one-dimensional NumPy array."""

    if isinstance(values, pd.Series):
        array = values.to_numpy(
            dtype=float
        )

    elif isinstance(values, np.ndarray):
        array = np.asarray(
            values,
            dtype=float,
        )

    else:
        raise TypeError(
            f"{name} must be a pandas.Series or numpy.ndarray."
        )

    return array