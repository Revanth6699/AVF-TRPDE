from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.risk.expected_shortfall import calculate_es_95
from backend.app.risk.filtered_historical import FilteredHistoricalSimulation
from backend.app.risk.var import calculate_var_95, calculate_var_99


REQUIRED_COLUMNS = (
    "target_timestamp",
    "asset_id",
    "target_return",
    "prediction",
)


class RiskPipelineError(ValueError):
    """Raised when out-of-sample forecasts cannot produce FHS risk points."""


def calculate_fhs_risk_points(
    predictions: pd.DataFrame,
    *,
    assets: list[str] | tuple[str, ...],
    start_timestamp: Any,
    end_timestamp: Any,
    min_observations: int = 100,
    n_simulations: int = 10_000,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Generate expanding-window FHS VaR/ES forecasts independently per asset.

    Each volatility prediction is aligned to the next-day return through the
    target_timestamp and target_return columns. For a risk forecast at date t,
    only standardized out-of-sample returns strictly before t are used.
    The current day's realized return is added to the residual history only
    after its risk forecast has been generated.

    The date filter controls returned points, not the historical observations
    available to initialize the FHS residual distribution.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas.DataFrame.")
    if predictions.empty:
        raise RiskPipelineError("No out-of-sample predictions were supplied.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(predictions.columns))
    if missing:
        raise RiskPipelineError(
            "Prediction artifact is missing required risk-alignment columns: "
            + ", ".join(missing)
        )
    if not assets:
        raise RiskPipelineError("At least one asset is required.")
    if isinstance(min_observations, bool) or not isinstance(min_observations, int):
        raise TypeError("min_observations must be an integer.")
    if min_observations < 2:
        raise RiskPipelineError("min_observations must be at least 2.")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int):
        raise TypeError("n_simulations must be an integer.")
    if n_simulations < 100:
        raise RiskPipelineError("n_simulations must be at least 100.")

    start = _normalize_timestamp(start_timestamp, "start_timestamp")
    end = _normalize_timestamp(end_timestamp, "end_timestamp")
    if start > end:
        raise RiskPipelineError("start_timestamp must not be after end_timestamp.")

    requested_assets = tuple(
        dict.fromkeys(str(asset).strip().upper() for asset in assets)
    )
    if any(not asset for asset in requested_assets):
        raise RiskPipelineError("Asset identifiers must not be empty.")

    frame = predictions.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["target_timestamp"] = _normalize_timestamp_series(
        frame["target_timestamp"]
    )
    frame["asset_id"] = frame["asset_id"].astype("string").str.strip().str.upper()
    frame["target_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")

    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise RiskPipelineError("Risk input contains missing or invalid values.")
    values = frame[["target_return", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RiskPipelineError("Risk input contains non-finite numeric values.")
    if (frame["prediction"] <= 0).any():
        raise RiskPipelineError("Volatility forecasts must be strictly positive.")
    if frame.duplicated(["asset_id", "target_timestamp"]).any():
        raise RiskPipelineError(
            "Duplicate asset/target_timestamp forecast rows were found."
        )

    available_assets = set(frame["asset_id"].astype(str))
    missing_assets = sorted(set(requested_assets) - available_assets)
    if missing_assets:
        raise RiskPipelineError(
            "No forecast observations exist for requested asset(s): "
            + ", ".join(missing_assets)
        )

    risk_points: list[dict[str, Any]] = []
    for asset_index, asset_id in enumerate(requested_assets):
        asset_frame = (
            frame.loc[frame["asset_id"] == asset_id]
            .sort_values("target_timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        history: list[dict[str, Any]] = []

        for row_index, row in asset_frame.iterrows():
            forecast_timestamp = row["target_timestamp"]
            forecast_volatility = float(row["prediction"])
            actual_return = float(row["target_return"])

            # Build the forecast distribution from strictly prior OOS
            # standardized returns. Do not append this row until afterward.
            if len(history) >= min_observations:
                history_frame = pd.DataFrame(history)
                fhs = FilteredHistoricalSimulation(
                    min_observations=min_observations,
                    random_state=random_state + asset_index * 100_003 + row_index,
                )
                standardized_returns = fhs.fit(history_frame)
                simulated_returns = fhs.simulate(
                    standardized_returns,
                    forecast_volatility,
                    n_simulations=n_simulations,
                )
                var_95 = calculate_var_95(simulated_returns)
                var_99 = calculate_var_99(simulated_returns)
                es_95 = calculate_es_95(simulated_returns)

                if start <= forecast_timestamp <= end:
                    risk_points.append(
                        {
                            "timestamp": forecast_timestamp,
                            "asset_id": asset_id,
                            "var_95": float(var_95.var),
                            "var_99": float(var_99.var),
                            "es_95": float(es_95.expected_shortfall),
                            "actual_return": actual_return,
                            "forecast_volatility": forecast_volatility,
                            "history_count": len(history),
                        }
                    )

            # Add the realized observation only after today's risk forecast.
            history.append(
                {
                    "timestamp": forecast_timestamp,
                    "asset_id": asset_id,
                    "return": actual_return,
                    "forecast_volatility": forecast_volatility,
                }
            )

    if not risk_points:
        raise RiskPipelineError(
            "No risk points were produced in the requested date range. "
            f"At least {min_observations} prior out-of-sample observations "
            "per asset are required before the first FHS forecast."
        )

    return risk_points


def _normalize_timestamp(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise RiskPipelineError(f"{name} must be a valid timestamp.")
    if timestamp.tzinfo is not None:
        # Preserve exchange-local wall-clock time to match dataset resolver.
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise RiskPipelineError("target_timestamp contains invalid values.")
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    return timestamps
