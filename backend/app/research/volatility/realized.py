from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "close",
)


class RealizedVolatilityError(ValueError):
    """Raised when realized volatility cannot be calculated."""


@dataclass(frozen=True)
class RealizedVolatilityResult:
    """Metadata describing a realized-volatility dataset."""

    dataframe: pd.DataFrame
    asset_count: int
    day_count: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def calculate_intraday_log_returns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate intraday log returns from OHLCV observations.

    Returns a copy of the input with an additional
    'log_return' column.

    Returns are calculated independently within each
    asset and trading day.

    This prevents the overnight return between two trading
    sessions from entering the daily realized-volatility
    calculation.
    """

    _validate_input(dataframe)

    df = dataframe[
        [
            "timestamp",
            "asset_id",
            "close",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise RealizedVolatilityError(
            "timestamp contains invalid values."
        )

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["asset_id"].isna().any():
        raise RealizedVolatilityError(
            "asset_id contains missing values."
        )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if df["close"].isna().any():
        raise RealizedVolatilityError(
            "close contains missing or non-numeric values."
        )

    if (~np.isfinite(df["close"])).any():
        raise RealizedVolatilityError(
            "close contains non-finite values."
        )

    if (df["close"] <= 0).any():
        raise RealizedVolatilityError(
            "close contains non-positive values."
        )

    df = df.sort_values(
        by=[
            "asset_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Calendar date is derived from the exchange-local
    # timestamp supplied by the data source.
    df["trading_date"] = df["timestamp"].dt.normalize()

    # Calculate log returns only within each asset/day.
    previous_close = (
        df.groupby(
            [
                "asset_id",
                "trading_date",
            ],
            sort=False,
        )["close"]
        .shift(1)
    )

    df["log_return"] = np.log(
        df["close"] / previous_close
    )

    return df


def compute_realized_variance(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily realized variance from intraday log returns.

    RV_t = sum_j(r_t,j^2)

    The first observation of every trading day has no
    previous intraday observation and therefore does not
    contribute to realized variance.
    """

    returns = calculate_intraday_log_returns(
        dataframe
    )

    valid_returns = returns[
        returns["log_return"].notna()
    ].copy()

    if valid_returns.empty:
        raise RealizedVolatilityError(
            "No valid intraday returns are available "
            "for realized-variance calculation."
        )

    valid_returns["squared_return"] = (
        valid_returns["log_return"] ** 2
    )

    realized = (
        valid_returns.groupby(
            [
                "asset_id",
                "trading_date",
            ],
            as_index=False,
        )
        .agg(
            realized_variance=(
                "squared_return",
                "sum",
            ),
            observation_count=(
                "log_return",
                "count",
            ),
        )
    )

    realized["realized_variance"] = (
        realized["realized_variance"].astype(float)
    )

    realized["observation_count"] = (
        realized["observation_count"].astype(int)
    )

    realized = realized.sort_values(
        by=[
            "asset_id",
            "trading_date",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return realized


def compute_realized_volatility(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily realized volatility.

    RVOL_t = sqrt(RV_t)

    Returns one observation per asset and trading day.
    """

    realized = compute_realized_variance(
        dataframe
    )

    realized["realized_volatility"] = np.sqrt(
        realized["realized_variance"]
    )

    realized = realized[
        [
            "asset_id",
            "trading_date",
            "realized_variance",
            "realized_volatility",
            "observation_count",
        ]
    ]

    return realized


def save_realized_volatility(
    dataframe: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Compute realized volatility and save it as Parquet.
    """

    realized = compute_realized_volatility(
        dataframe
    )

    path = str(output_path)

    if not path.lower().endswith(".parquet"):
        raise RealizedVolatilityError(
            "Realized-volatility output must use "
            "the '.parquet' extension."
        )

    from pathlib import Path

    output_file = Path(path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    realized.to_parquet(
        output_file,
        index=False,
    )


def build_realized_volatility_result(
    dataframe: pd.DataFrame,
) -> RealizedVolatilityResult:
    """
    Compute realized volatility and return it together
    with dataset metadata.
    """

    realized = compute_realized_volatility(
        dataframe
    )

    if realized.empty:
        raise RealizedVolatilityError(
            "Realized-volatility dataset is empty."
        )

    assets = realized[
        "asset_id"
    ].unique()

    return RealizedVolatilityResult(
        dataframe=realized,
        asset_count=len(assets),
        day_count=len(realized),
        start_date=realized[
            "trading_date"
        ].min(),
        end_date=realized[
            "trading_date"
        ].max(),
    )


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the minimum input contract required by the
    realized-volatility engine.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise RealizedVolatilityError(
            "Intraday dataset is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise RealizedVolatilityError(
            "Missing required columns: "
            + ", ".join(missing)
        )