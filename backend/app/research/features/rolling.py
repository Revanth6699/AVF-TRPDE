from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class RollingFeatureError(ValueError):
    """Raised when rolling features cannot be constructed."""


@dataclass(frozen=True)
class RollingFeatureResult:
    """Metadata describing a rolling-feature dataset."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int
    feature_columns: tuple[str, ...]


def create_rolling_features(
    dataframe: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Create asset-wise rolling market features.

    Features:
    - rolling volatility from daily log returns
    - rolling average volume
    - rolling average normalized price range
    - momentum over each window

    All features at time t use information available through time t only.
    """

    _validate_input(dataframe)
    _validate_windows(windows)

    df = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["asset_id"] = df["asset_id"].astype("string")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_values(
        by=["asset_id", "timestamp"],
        ascending=True,
    ).reset_index(drop=True)

    grouped_close = df.groupby(
        "asset_id",
        sort=False,
    )["close"]

    df["log_return"] = np.log(
        grouped_close.transform(
            lambda series: series / series.shift(1)
        )
    )

    df["price_range"] = (
        (df["high"] - df["low"])
        / df["close"]
    )

    feature_columns: list[str] = []

    for window in windows:
        rolling_return_std = (
            df.groupby("asset_id", sort=False)["log_return"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(ddof=1)
            .reset_index(level=0, drop=True)
        )

        rolling_volume_mean = (
            df.groupby("asset_id", sort=False)["volume"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_range_mean = (
            df.groupby("asset_id", sort=False)["price_range"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(level=0, drop=True)
        )

        momentum = (
            grouped_close
            .transform(
                lambda series: series / series.shift(window) - 1.0
            )
        )

        df[f"rolling_volatility_{window}"] = rolling_return_std
        df[f"rolling_volume_{window}"] = rolling_volume_mean
        df[f"rolling_range_{window}"] = rolling_range_mean
        df[f"momentum_{window}"] = momentum

        feature_columns.extend(
            [
                f"rolling_volatility_{window}",
                f"rolling_volume_{window}",
                f"rolling_range_{window}",
                f"momentum_{window}",
            ]
        )

    result_columns = [
        "timestamp",
        "asset_id",
        *feature_columns,
    ]

    result = df.loc[:, result_columns].copy()

    result = result.sort_values(
        by=["timestamp", "asset_id"],
        ascending=True,
    ).reset_index(drop=True)

    return result


def build_rolling_feature_result(
    dataframe: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10, 20),
) -> RollingFeatureResult:
    """Build rolling features together with result metadata."""

    result = create_rolling_features(
        dataframe,
        windows=windows,
    )

    feature_columns = tuple(
        column
        for column in result.columns
        if column not in {"timestamp", "asset_id"}
    )

    return RollingFeatureResult(
        dataframe=result,
        asset_count=int(result["asset_id"].nunique()),
        observation_count=len(result),
        feature_columns=feature_columns,
    )


def _validate_input(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas.DataFrame.")

    if dataframe.empty:
        raise RollingFeatureError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RollingFeatureError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise RollingFeatureError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise RollingFeatureError(
            "asset_id contains missing values."
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if values.isna().any():
            raise RollingFeatureError(
                f"{column} contains invalid or missing values."
            )

    prices = dataframe[
        ["open", "high", "low", "close"]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if (prices <= 0).any().any():
        raise RollingFeatureError(
            "OHLC prices must be greater than zero."
        )

    volume = pd.to_numeric(
        dataframe["volume"],
        errors="coerce",
    )

    if (volume < 0).any():
        raise RollingFeatureError(
            "volume must not contain negative values."
        )


def _validate_windows(
    windows: tuple[int, ...],
) -> None:
    if not windows:
        raise RollingFeatureError(
            "At least one rolling window is required."
        )

    if any(
        not isinstance(window, int) or isinstance(window, bool)
        for window in windows
    ):
        raise RollingFeatureError(
            "Rolling windows must contain integers."
        )

    if any(window < 2 for window in windows):
        raise RollingFeatureError(
            "Rolling windows must be at least 2."
        )

    if len(set(windows)) != len(windows):
        raise RollingFeatureError(
            "Rolling windows must not contain duplicates."
        )