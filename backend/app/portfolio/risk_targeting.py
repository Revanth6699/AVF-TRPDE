from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "forecast_volatility",
    "base_weight",
)


class RiskTargetingError(ValueError):
    """Raised when risk-targeted portfolio weights cannot be calculated."""


@dataclass(frozen=True)
class RiskTargetingResult:
    """Metadata and output for a risk-targeting calculation."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int


def calculate_risk_targeted_weights(
    dataframe: pd.DataFrame,
    *,
    target_volatility: float,
    max_position: float,
) -> pd.DataFrame:
    """
    Calculate risk-targeted portfolio weights.

    Locked specification:

        w_i,t = w_i,base * sigma_target / sigma_hat_i,t

    Position constraint:

        |w_i,t| <= max_position

    The function performs asset-level risk targeting only.
    It does not normalize weights across assets.

    Parameters
    ----------
    dataframe:
        DataFrame containing timestamp, asset_id,
        forecast_volatility, and base_weight.

    target_volatility:
        Desired portfolio/asset volatility target used by
        the locked risk-targeting formula.

    max_position:
        Maximum absolute position allowed for each asset.

    Returns
    -------
    pd.DataFrame
        Original identifying columns plus target_weight.
    """

    _validate_input(dataframe)
    _validate_parameters(
        target_volatility=target_volatility,
        max_position=max_position,
    )

    df = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["asset_id"] = df["asset_id"].astype("string").str.strip()

    df["forecast_volatility"] = pd.to_numeric(
        df["forecast_volatility"],
        errors="coerce",
    )

    df["base_weight"] = pd.to_numeric(
        df["base_weight"],
        errors="coerce",
    )

    df = df.sort_values(
        by=["timestamp", "asset_id"],
        ascending=True,
    ).reset_index(drop=True)

    raw_weights = (
        df["base_weight"]
        * target_volatility
        / df["forecast_volatility"]
    )

    # Apply only the locked per-position constraint.
    df["target_weight"] = raw_weights.clip(
        lower=-max_position,
        upper=max_position,
    )

    return df.loc[
        :,
        [
            "timestamp",
            "asset_id",
            "forecast_volatility",
            "base_weight",
            "target_weight",
        ],
    ]


def build_risk_targeting_result(
    dataframe: pd.DataFrame,
    *,
    target_volatility: float,
    max_position: float,
) -> RiskTargetingResult:
    """Calculate risk-targeted weights and return result metadata."""

    result = calculate_risk_targeted_weights(
        dataframe,
        target_volatility=target_volatility,
        max_position=max_position,
    )

    return RiskTargetingResult(
        dataframe=result,
        asset_count=int(result["asset_id"].nunique()),
        observation_count=len(result),
    )


def _validate_input(dataframe: pd.DataFrame) -> None:
    """Validate the input DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise RiskTargetingError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RiskTargetingError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise RiskTargetingError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise RiskTargetingError(
            "asset_id contains missing values."
        )

    asset_ids = (
        dataframe["asset_id"]
        .astype("string")
        .str.strip()
    )

    if asset_ids.eq("").any():
        raise RiskTargetingError(
            "asset_id contains empty values."
        )

    forecast_volatility = pd.to_numeric(
        dataframe["forecast_volatility"],
        errors="coerce",
    )

    if forecast_volatility.isna().any():
        raise RiskTargetingError(
            "forecast_volatility contains invalid or missing values."
        )

    if not np.isfinite(forecast_volatility.to_numpy()).all():
        raise RiskTargetingError(
            "forecast_volatility must contain only finite values."
        )

    if (forecast_volatility <= 0).any():
        raise RiskTargetingError(
            "forecast_volatility must be greater than zero."
        )

    base_weight = pd.to_numeric(
        dataframe["base_weight"],
        errors="coerce",
    )

    if base_weight.isna().any():
        raise RiskTargetingError(
            "base_weight contains invalid or missing values."
        )

    if not np.isfinite(base_weight.to_numpy()).all():
        raise RiskTargetingError(
            "base_weight must contain only finite values."
        )


def _validate_parameters(
    *,
    target_volatility: float,
    max_position: float,
) -> None:
    """Validate risk-targeting parameters."""

    if not isinstance(
        target_volatility,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "target_volatility must be numeric."
        )

    if not np.isfinite(target_volatility):
        raise RiskTargetingError(
            "target_volatility must be finite."
        )

    if target_volatility <= 0:
        raise RiskTargetingError(
            "target_volatility must be greater than zero."
        )

    if not isinstance(
        max_position,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "max_position must be numeric."
        )

    if not np.isfinite(max_position):
        raise RiskTargetingError(
            "max_position must be finite."
        )

    if max_position <= 0:
        raise RiskTargetingError(
            "max_position must be greater than zero."
        )