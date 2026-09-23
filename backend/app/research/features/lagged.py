from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "close",
)


class LaggedFeatureError(ValueError):
    """Raised when lagged features cannot be constructed."""


@dataclass(frozen=True)
class LaggedFeatureResult:
    """Metadata describing a lagged-feature dataset."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int
    feature_columns: tuple[str, ...]


def create_lagged_returns(
    dataframe: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    """
    Create lagged daily log-return features.

    Features:

        lagged_return_1
        lagged_return_2
        lagged_return_3
        lagged_return_5
        lagged_return_10

    Each feature uses information available strictly before
    the current observation.

    No future observations are used.
    """

    _validate_input(dataframe)
    _validate_lags(lags)

    df = dataframe[
        [
            "timestamp",
            "asset_id",
            "close",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Normalize timestamp
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise LaggedFeatureError(
            "timestamp contains invalid values."
        )

    # ---------------------------------------------------------
    # Normalize asset identifier
    # ---------------------------------------------------------

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["asset_id"].isna().any():
        raise LaggedFeatureError(
            "asset_id contains missing values."
        )

    if (df["asset_id"] == "").any():
        raise LaggedFeatureError(
            "asset_id contains empty values."
        )

    # ---------------------------------------------------------
    # Normalize close prices
    # ---------------------------------------------------------

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if df["close"].isna().any():
        raise LaggedFeatureError(
            "close contains missing or non-numeric values."
        )

    if (~np.isfinite(df["close"])).any():
        raise LaggedFeatureError(
            "close contains non-finite values."
        )

    if (df["close"] <= 0).any():
        raise LaggedFeatureError(
            "close contains non-positive values."
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
    # Duplicate observation detection
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "asset_id",
            "timestamp",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise LaggedFeatureError(
            "Duplicate asset/timestamp observations "
            f"detected: {duplicate_count} rows."
        )

    # ---------------------------------------------------------
    # Daily log return
    #
    # r_t = log(C_t / C_(t-1))
    #
    # Calculated independently for each asset.
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
    # Lagged returns
    #
    # lagged_return_k at t contains the return from
    # t-k, therefore no future information is used.
    # ---------------------------------------------------------

    feature_columns: list[str] = []

    for lag in lags:
        column = f"lagged_return_{lag}"

        df[column] = (
            df.groupby(
                "asset_id",
                sort=False,
            )["log_return"]
            .shift(lag)
        )

        feature_columns.append(column)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    result_columns = [
        "timestamp",
        "asset_id",
        *feature_columns,
    ]

    return df[result_columns].copy()


def create_lagged_volatility(
    dataframe: pd.DataFrame,
    *,
    volatility_column: str = "realized_volatility",
    lags: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    """
    Create lagged realized-volatility features.

    Features:

        lagged_volatility_1
        lagged_volatility_2
        lagged_volatility_3
        lagged_volatility_5
        lagged_volatility_10

    Only historical volatility observations are used.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise LaggedFeatureError(
            "Input dataset is empty."
        )

    if not isinstance(
        volatility_column,
        str,
    ) or not volatility_column.strip():
        raise LaggedFeatureError(
            "volatility_column must be a non-empty string."
        )

    if volatility_column not in dataframe.columns:
        raise LaggedFeatureError(
            f"Missing volatility column: "
            f"{volatility_column}"
        )

    _validate_lags(lags)

    required = [
        "timestamp",
        "asset_id",
        volatility_column,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise LaggedFeatureError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = dataframe[required].copy()

    # ---------------------------------------------------------
    # Normalize timestamp
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise LaggedFeatureError(
            "timestamp contains invalid values."
        )

    # ---------------------------------------------------------
    # Normalize asset identifier
    # ---------------------------------------------------------

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["asset_id"].isna().any():
        raise LaggedFeatureError(
            "asset_id contains missing values."
        )

    if (df["asset_id"] == "").any():
        raise LaggedFeatureError(
            "asset_id contains empty values."
        )

    # ---------------------------------------------------------
    # Normalize volatility
    # ---------------------------------------------------------

    df[volatility_column] = pd.to_numeric(
        df[volatility_column],
        errors="coerce",
    )

    if df[volatility_column].isna().any():
        raise LaggedFeatureError(
            f"{volatility_column} contains missing "
            "or non-numeric values."
        )

    if (
        ~np.isfinite(
            df[volatility_column]
        )
    ).any():
        raise LaggedFeatureError(
            f"{volatility_column} contains "
            "non-finite values."
        )

    if (
        df[volatility_column] < 0
    ).any():
        raise LaggedFeatureError(
            f"{volatility_column} cannot be negative."
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
    # Duplicate observation detection
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "asset_id",
            "timestamp",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise LaggedFeatureError(
            "Duplicate asset/timestamp observations "
            f"detected: {duplicate_count} rows."
        )

    # ---------------------------------------------------------
    # Lagged volatility
    # ---------------------------------------------------------

    feature_columns: list[str] = []

    for lag in lags:
        column = f"lagged_volatility_{lag}"

        df[column] = (
            df.groupby(
                "asset_id",
                sort=False,
            )[volatility_column]
            .shift(lag)
        )

        feature_columns.append(column)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    return df[
        [
            "timestamp",
            "asset_id",
            *feature_columns,
        ]
    ].copy()


def create_lagged_features(
    dataframe: pd.DataFrame,
    *,
    return_lags: tuple[int, ...] = (
        1,
        2,
        3,
        5,
        10,
    ),
    volatility_column: str | None = None,
    volatility_lags: tuple[int, ...] = (
        1,
        2,
        3,
        5,
        10,
    ),
) -> pd.DataFrame:
    """
    Create the combined lagged feature set.

    Return features are always created.

    Volatility features are created only when
    volatility_column is supplied.
    """

    lagged_returns = create_lagged_returns(
        dataframe,
        lags=return_lags,
    )

    result = lagged_returns

    if volatility_column is not None:
        volatility_features = create_lagged_volatility(
            dataframe,
            volatility_column=volatility_column,
            lags=volatility_lags,
        )

        result = result.merge(
            volatility_features,
            on=[
                "timestamp",
                "asset_id",
            ],
            how="inner",
            validate="one_to_one",
        )

    return result.sort_values(
        by=[
            "asset_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def build_lagged_feature_result(
    dataframe: pd.DataFrame,
    *,
    return_lags: tuple[int, ...] = (
        1,
        2,
        3,
        5,
        10,
    ),
    volatility_column: str | None = None,
    volatility_lags: tuple[int, ...] = (
        1,
        2,
        3,
        5,
        10,
    ),
) -> LaggedFeatureResult:
    """
    Build lagged features and return them with metadata.
    """

    result = create_lagged_features(
        dataframe,
        return_lags=return_lags,
        volatility_column=volatility_column,
        volatility_lags=volatility_lags,
    )

    feature_columns = tuple(
        column
        for column in result.columns
        if column not in {
            "timestamp",
            "asset_id",
        }
    )

    if not feature_columns:
        raise LaggedFeatureError(
            "No lagged features were generated."
        )

    return LaggedFeatureResult(
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
        raise LaggedFeatureError(
            "Input dataset is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise LaggedFeatureError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def _validate_lags(
    lags: tuple[int, ...],
) -> None:
    if not isinstance(
        lags,
        tuple,
    ):
        raise TypeError(
            "lags must be a tuple of positive integers."
        )

    if not lags:
        raise LaggedFeatureError(
            "At least one lag must be specified."
        )

    if any(
        not isinstance(lag, int)
        or isinstance(lag, bool)
        or lag <= 0
        for lag in lags
    ):
        raise LaggedFeatureError(
            "All lags must be positive integers."
        )

    if len(set(lags)) != len(lags):
        raise LaggedFeatureError(
            "Duplicate lag values are not allowed."
        )