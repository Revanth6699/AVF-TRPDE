from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "garch_forecast",
)

OUTPUT_COLUMNS = (
    "timestamp",
    "asset_id",
    "garch_forecast",
)


class GARCHFeatureError(ValueError):
    """Raised when GARCH features cannot be constructed."""


@dataclass(frozen=True)
class GARCHFeatureResult:
    """Metadata describing a GARCH-feature dataset."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int
    feature_columns: tuple[str, ...]


def create_garch_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare GJR-GARCH forecasts as ML features.

    The input must contain one forecast for each asset/timestamp
    combination. The function does not fit a GARCH model.

    GARCH model fitting belongs to models/gjr_garch.py and is performed
    inside the walk-forward training process.

    Output:
    - timestamp
    - asset_id
    - garch_forecast
    """

    _validate_input(dataframe)

    df = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["garch_forecast"] = pd.to_numeric(
        df["garch_forecast"],
        errors="coerce",
    )

    df = df.sort_values(
        by=["timestamp", "asset_id"],
        ascending=True,
    ).reset_index(drop=True)

    return df.loc[:, OUTPUT_COLUMNS].copy()


def merge_garch_features(
    dataframe: pd.DataFrame,
    garch_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge GJR-GARCH forecasts into an existing feature dataset.

    The merge is performed using:
    - timestamp
    - asset_id

    The original row order of the feature dataframe is preserved.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if not isinstance(garch_features, pd.DataFrame):
        raise TypeError(
            "garch_features must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise GARCHFeatureError(
            "Input feature dataframe must not be empty."
        )

    prepared_garch = create_garch_features(
        garch_features,
    )

    required_base_columns = {
        "timestamp",
        "asset_id",
    }

    missing_base = [
        column
        for column in required_base_columns
        if column not in dataframe.columns
    ]

    if missing_base:
        raise GARCHFeatureError(
            "Input feature dataframe is missing required columns: "
            + ", ".join(missing_base)
        )

    result = dataframe.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["asset_id"] = (
        result["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if result["timestamp"].isna().any():
        raise GARCHFeatureError(
            "Input feature dataframe contains invalid timestamps."
        )

    if result["asset_id"].isna().any():
        raise GARCHFeatureError(
            "Input feature dataframe contains missing asset IDs."
        )

    if result.duplicated(
        subset=["timestamp", "asset_id"],
    ).any():
        raise GARCHFeatureError(
            "Input feature dataframe contains duplicate "
            "(timestamp, asset_id) combinations."
        )

    if prepared_garch.duplicated(
        subset=["timestamp", "asset_id"],
    ).any():
        raise GARCHFeatureError(
            "GARCH features contain duplicate "
            "(timestamp, asset_id) combinations."
        )

    if "garch_forecast" in result.columns:
        raise GARCHFeatureError(
            "Input feature dataframe already contains "
            "'garch_forecast'."
        )

    result = result.merge(
        prepared_garch,
        on=["timestamp", "asset_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    return result


def build_garch_feature_result(
    dataframe: pd.DataFrame,
) -> GARCHFeatureResult:
    """Build GARCH features together with result metadata."""

    result = create_garch_features(dataframe)

    feature_columns = tuple(
        column
        for column in result.columns
        if column not in {"timestamp", "asset_id"}
    )

    return GARCHFeatureResult(
        dataframe=result,
        asset_count=int(result["asset_id"].nunique()),
        observation_count=len(result),
        feature_columns=feature_columns,
    )


def _validate_input(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise GARCHFeatureError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise GARCHFeatureError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise GARCHFeatureError(
            "timestamp contains invalid or missing values."
        )

    asset_ids = (
        dataframe["asset_id"]
        .astype("string")
        .str.strip()
    )

    if asset_ids.isna().any() or asset_ids.eq("").any():
        raise GARCHFeatureError(
            "asset_id contains missing or empty values."
        )

    forecasts = pd.to_numeric(
        dataframe["garch_forecast"],
        errors="coerce",
    )

    if forecasts.isna().any():
        raise GARCHFeatureError(
            "garch_forecast contains invalid or missing values."
        )

    if not np.isfinite(
        forecasts.to_numpy()
    ).all():
        raise GARCHFeatureError(
            "garch_forecast contains non-finite values."
        )

    if (forecasts < 0).any():
        raise GARCHFeatureError(
            "garch_forecast must not contain negative values."
        )

    duplicate_keys = dataframe.duplicated(
        subset=["timestamp", "asset_id"],
    )

    if duplicate_keys.any():
        raise GARCHFeatureError(
            "Duplicate (timestamp, asset_id) combinations "
            "are not allowed."
        )