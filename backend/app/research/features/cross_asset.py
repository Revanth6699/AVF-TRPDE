from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "close",
)


class CrossAssetFeatureError(ValueError):
    """Raised when cross-asset features cannot be constructed."""


@dataclass(frozen=True)
class CrossAssetFeatureResult:
    """Metadata describing a cross-asset feature dataset."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int
    feature_columns: tuple[str, ...]


def create_cross_asset_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create cross-asset market features.

    Features:
    - cross-sectional mean return excluding the target asset
    - cross-sectional return dispersion excluding the target asset
    - cross-sectional mean absolute return excluding the target asset

    For an observation belonging to asset A at time t, the features
    use information from the other assets at the same timestamp t only.

    No future observations are used.
    """

    _validate_input(dataframe)

    df = dataframe.loc[
        :,
        REQUIRED_COLUMNS,
    ].copy()

    # ---------------------------------------------------------
    # Timestamp normalization
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise CrossAssetFeatureError(
            "timestamp contains invalid or missing values."
        )

    # ---------------------------------------------------------
    # Asset identifier normalization
    # ---------------------------------------------------------

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["asset_id"].isna().any():
        raise CrossAssetFeatureError(
            "asset_id contains missing values."
        )

    if (df["asset_id"] == "").any():
        raise CrossAssetFeatureError(
            "asset_id contains empty values."
        )

    # ---------------------------------------------------------
    # Close-price normalization
    # ---------------------------------------------------------

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if df["close"].isna().any():
        raise CrossAssetFeatureError(
            "close contains invalid or missing values."
        )

    if not np.isfinite(
        df["close"].to_numpy()
    ).all():
        raise CrossAssetFeatureError(
            "close contains non-finite values."
        )

    if (df["close"] <= 0).any():
        raise CrossAssetFeatureError(
            "close prices must be greater than zero."
        )

    # ---------------------------------------------------------
    # Duplicate observation detection
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "timestamp",
            "asset_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise CrossAssetFeatureError(
            "Duplicate asset/timestamp observations "
            f"detected: {duplicate_count} rows."
        )

    # ---------------------------------------------------------
    # Deterministic ordering
    # ---------------------------------------------------------

    df = df.sort_values(
        by=[
            "asset_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Asset-wise log returns
    #
    # r_(i,t) = log(C_(i,t) / C_(i,t-1))
    #
    # Returns are calculated independently for each asset.
    # ---------------------------------------------------------

    df["log_return"] = (
        df.groupby(
            "asset_id",
            sort=False,
        )["close"]
        .transform(
            lambda series: np.log(
                series / series.shift(1)
            )
        )
    )

    # ---------------------------------------------------------
    # Cross-sectional statistics
    #
    # At timestamp t:
    #
    #   mean return
    #   return dispersion
    #   mean absolute return
    #
    # are calculated from OTHER assets only.
    # ---------------------------------------------------------

    grouped = df.groupby(
        "timestamp",
        sort=False,
    )["log_return"]

    cross_asset_sum = grouped.transform("sum")
    cross_asset_count = grouped.transform("count")

    own_return = df["log_return"]

    own_return_available = (
        own_return.notna().astype(int)
    )

    other_asset_count = (
        cross_asset_count
        - own_return_available
    )

    other_asset_sum = (
        cross_asset_sum
        - own_return.fillna(0.0)
    )

    # ---------------------------------------------------------
    # Cross-asset mean return
    # ---------------------------------------------------------

    df["cross_asset_mean_return"] = (
        other_asset_sum
        / other_asset_count.replace(
            0,
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # Cross-asset return dispersion
    #
    # sqrt(E[r²] - E[r]²)
    # ---------------------------------------------------------

    squared_returns = (
        df["log_return"] ** 2
    )

    cross_asset_squared_sum = (
        squared_returns.groupby(
            df["timestamp"],
            sort=False,
        ).transform("sum")
    )

    own_squared_return = (
        squared_returns.fillna(0.0)
    )

    other_squared_sum = (
        cross_asset_squared_sum
        - own_squared_return
    )

    cross_asset_mean_squared_return = (
        other_squared_sum
        / other_asset_count.replace(
            0,
            np.nan,
        )
    )

    variance = (
        cross_asset_mean_squared_return
        - df["cross_asset_mean_return"] ** 2
    )

    df["cross_asset_return_dispersion"] = (
        np.sqrt(
            np.maximum(
                variance,
                0.0,
            )
        )
    )

    # ---------------------------------------------------------
    # Cross-asset mean absolute return
    # ---------------------------------------------------------

    absolute_returns = (
        df["log_return"].abs()
    )

    cross_asset_abs_sum = (
        absolute_returns.groupby(
            df["timestamp"],
            sort=False,
        ).transform("sum")
    )

    own_abs_return = (
        absolute_returns.fillna(0.0)
    )

    other_abs_sum = (
        cross_asset_abs_sum
        - own_abs_return
    )

    df["cross_asset_mean_abs_return"] = (
        other_abs_sum
        / other_asset_count.replace(
            0,
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # Locked feature set
    # ---------------------------------------------------------

    feature_columns = [
        "cross_asset_mean_return",
        "cross_asset_return_dispersion",
        "cross_asset_mean_abs_return",
    ]

    result = df.loc[
        :,
        [
            "timestamp",
            "asset_id",
            *feature_columns,
        ],
    ].copy()

    # ---------------------------------------------------------
    # Final deterministic ordering
    # ---------------------------------------------------------

    result = result.sort_values(
        by=[
            "timestamp",
            "asset_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return result


def build_cross_asset_feature_result(
    dataframe: pd.DataFrame,
) -> CrossAssetFeatureResult:
    """Build cross-asset features together with result metadata."""

    result = create_cross_asset_features(
        dataframe
    )

    feature_columns = tuple(
        column
        for column in result.columns
        if column not in {
            "timestamp",
            "asset_id",
        }
    )

    return CrossAssetFeatureResult(
        dataframe=result,
        asset_count=int(
            result["asset_id"].nunique()
        ),
        observation_count=len(result),
        feature_columns=feature_columns,
    )


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise CrossAssetFeatureError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise CrossAssetFeatureError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise CrossAssetFeatureError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise CrossAssetFeatureError(
            "asset_id contains missing values."
        )

    asset_ids = (
        dataframe["asset_id"]
        .astype("string")
        .str.strip()
    )

    if asset_ids.eq("").any():
        raise CrossAssetFeatureError(
            "asset_id contains empty values."
        )

    close = pd.to_numeric(
        dataframe["close"],
        errors="coerce",
    )

    if close.isna().any():
        raise CrossAssetFeatureError(
            "close contains invalid or missing values."
        )

    if not np.isfinite(
        close.to_numpy()
    ).all():
        raise CrossAssetFeatureError(
            "close contains non-finite values."
        )

    if (close <= 0).any():
        raise CrossAssetFeatureError(
            "close prices must be greater than zero."
        )