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

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "qlike": self.qlike,
        }


def calculate_mae(
    actual: np.ndarray | pd.Series | list[float],
    forecast: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Calculate Mean Absolute Error.

    MAE = mean(|y - y_hat|)
    """
    actual_array, forecast_array = _prepare_inputs(
        actual,
        forecast,
        require_positive_forecast=False,
    )

    return float(
        np.mean(
            np.abs(actual_array - forecast_array)
        )
    )


def calculate_rmse(
    actual: np.ndarray | pd.Series | list[float],
    forecast: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Calculate Root Mean Squared Error.

    RMSE = sqrt(mean((y - y_hat)^2))
    """
    actual_array, forecast_array = _prepare_inputs(
        actual,
        forecast,
        require_positive_forecast=False,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(actual_array - forecast_array)
            )
        )
    )


def calculate_qlike(
    actual: np.ndarray | pd.Series | list[float],
    forecast: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Calculate QLIKE loss for volatility forecasts.

    QLIKE = mean(
        actual / forecast
        - log(actual / forecast)
        - 1
    )

    Both actual and forecast volatility values must be
    strictly positive.
    """
    actual_array, forecast_array = _prepare_inputs(
        actual,
        forecast,
        require_positive_forecast=True,
    )

    if np.any(actual_array <= 0):
        raise ForecastMetricError(
            "Actual volatility values must be strictly positive "
            "for QLIKE."
        )

    ratio = actual_array / forecast_array

    qlike_values = (
        ratio
        - np.log(ratio)
        - 1.0
    )

    return float(np.mean(qlike_values))


def calculate_forecast_metrics(
    actual: np.ndarray | pd.Series | list[float],
    forecast: np.ndarray | pd.Series | list[float],
) -> ForecastMetrics:
    """
    Calculate all locked forecast evaluation metrics.

    Returns:
        ForecastMetrics containing MAE, RMSE, and QLIKE.
    """
    mae = calculate_mae(
        actual,
        forecast,
    )

    rmse = calculate_rmse(
        actual,
        forecast,
    )

    qlike = calculate_qlike(
        actual,
        forecast,
    )

    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        qlike=qlike,
    )


def evaluate_forecast_dataframe(
    dataframe: pd.DataFrame,
    *,
    actual_column: str = "actual",
    forecast_column: str = "forecast",
) -> ForecastMetrics:
    """
    Calculate forecast metrics from a DataFrame.

    The DataFrame must contain the specified actual and
    forecast columns.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise ForecastMetricError(
            "Forecast dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in (
            actual_column,
            forecast_column,
        )
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ForecastMetricError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return calculate_forecast_metrics(
        actual=dataframe[actual_column],
        forecast=dataframe[forecast_column],
    )


def compare_forecasts(
    actual: np.ndarray | pd.Series | list[float],
    forecasts: dict[str, np.ndarray | pd.Series | list[float]],
) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, and QLIKE for multiple forecasts.

    Args:
        actual: Realized volatility observations.
        forecasts: Mapping of model name to forecast values.

    Returns:
        DataFrame containing one row per model.
    """
    if not forecasts:
        raise ForecastMetricError(
            "At least one forecast is required."
        )

    rows: list[dict[str, float | str]] = []

    for model_name, forecast in forecasts.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ForecastMetricError(
                "Forecast model names must be non-empty strings."
            )

        metrics = calculate_forecast_metrics(
            actual=actual,
            forecast=forecast,
        )

        rows.append(
            {
                "model": model_name,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "qlike": metrics.qlike,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "mae",
            "rmse",
            "qlike",
        ],
    )


def _prepare_inputs(
    actual: np.ndarray | pd.Series | list[float],
    forecast: np.ndarray | pd.Series | list[float],
    *,
    require_positive_forecast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert metric inputs to float arrays."""

    actual_array = _to_float_array(
        actual,
        name="actual",
    )

    forecast_array = _to_float_array(
        forecast,
        name="forecast",
    )

    if len(actual_array) != len(forecast_array):
        raise ForecastMetricError(
            "Actual and forecast arrays must have the same length."
        )

    if len(actual_array) == 0:
        raise ForecastMetricError(
            "Actual and forecast arrays must not be empty."
        )

    if require_positive_forecast and np.any(
        forecast_array <= 0
    ):
        raise ForecastMetricError(
            "Forecast volatility values must be strictly positive "
            "for QLIKE."
        )

    return actual_array, forecast_array


def _to_float_array(
    values: np.ndarray | pd.Series | list[float],
    *,
    name: str,
) -> np.ndarray:
    """Convert metric input to a validated one-dimensional array."""

    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype=float)

    elif isinstance(values, np.ndarray):
        try:
            array = values.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ForecastMetricError(
                f"{name} contains non-numeric values."
            ) from exc

    elif isinstance(values, list):
        try:
            array = np.asarray(
                values,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ForecastMetricError(
                f"{name} contains non-numeric values."
            ) from exc

    else:
        raise TypeError(
            f"{name} must be a numpy array, pandas Series, "
            "or list of floats."
        )

    if array.ndim != 1:
        raise ForecastMetricError(
            f"{name} must be one-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ForecastMetricError(
            f"{name} contains NaN or infinite values."
        )

    return array