from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "asset_id",
    "trading_date",
    "realized_volatility",
)


class TargetConstructionError(ValueError):
    """Raised when the volatility target cannot be constructed."""


@dataclass(frozen=True)
class VolatilityTargetResult:
    """Metadata describing the one-step-ahead volatility target."""

    dataframe: pd.DataFrame
    asset_count: int
    observation_count: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def create_next_day_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the one-step-ahead realized-volatility target.

    Target definition:

        y_(t+1) = RVOL_(t+1)

    The target is aligned with the information date t.

    Example:

        trading_date     realized_volatility     target
        2026-09-15       0.0111                  0.0097
        2026-09-16       0.0097                  NaN

    The final observation of each asset has no future target and
    is therefore removed from the returned supervised dataset.
    """

    _validate_input(dataframe)

    target = dataframe[
        [
            "asset_id",
            "trading_date",
            "realized_volatility",
        ]
    ].copy()

    target["trading_date"] = pd.to_datetime(
        target["trading_date"],
        errors="coerce",
    )

    if target["trading_date"].isna().any():
        raise TargetConstructionError(
            "trading_date contains invalid values."
        )

    target["asset_id"] = (
        target["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    target["realized_volatility"] = pd.to_numeric(
        target["realized_volatility"],
        errors="coerce",
    )

    if target["realized_volatility"].isna().any():
        raise TargetConstructionError(
            "realized_volatility contains missing or "
            "non-numeric values."
        )

    if (
        ~np.isfinite(
            target["realized_volatility"]
        )
    ).any():
        raise TargetConstructionError(
            "realized_volatility contains non-finite values."
        )

    if (
        target["realized_volatility"] < 0
    ).any():
        raise TargetConstructionError(
            "realized_volatility cannot be negative."
        )

    target = target.sort_values(
        by=[
            "asset_id",
            "trading_date",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_mask = target.duplicated(
        subset=[
            "asset_id",
            "trading_date",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(
            duplicate_mask.sum()
        )

        raise TargetConstructionError(
            "Duplicate asset/trading_date observations "
            f"detected: {duplicate_count} rows."
        )

    # Shift future realized volatility backward so that
    # today's row contains tomorrow's realized volatility.
    target["target"] = (
        target.groupby(
            "asset_id",
            sort=False,
        )["realized_volatility"]
        .shift(-1)
    )

    # The final date for every asset has no t+1 observation.
    target = target[
        target["target"].notna()
    ].copy()

    target = target[
        [
            "asset_id",
            "trading_date",
            "realized_volatility",
            "target",
        ]
    ]

    target["target"] = target[
        "target"
    ].astype(float)

    target = target.reset_index(
        drop=True
    )

    return target


def build_target_result(
    dataframe: pd.DataFrame,
) -> VolatilityTargetResult:
    """
    Construct the next-day target and return it with metadata.
    """

    target = create_next_day_target(
        dataframe
    )

    if target.empty:
        raise TargetConstructionError(
            "No one-step-ahead target observations "
            "could be constructed."
        )

    assets = target[
        "asset_id"
    ].unique()

    return VolatilityTargetResult(
        dataframe=target,
        asset_count=len(assets),
        observation_count=len(target),
        start_date=target[
            "trading_date"
        ].min(),
        end_date=target[
            "trading_date"
        ].max(),
    )


def save_target(
    dataframe: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Construct the one-step-ahead target and save it as Parquet.
    """

    target = create_next_day_target(
        dataframe
    )

    if target.empty:
        raise TargetConstructionError(
            "Cannot save an empty target dataset."
        )

    from pathlib import Path

    path = Path(output_path)

    if path.suffix.lower() != ".parquet":
        raise TargetConstructionError(
            "Target output must use the '.parquet' extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.to_parquet(
        path,
        index=False,
    )


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the minimum realized-volatility dataset
    required for target construction.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise TargetConstructionError(
            "Realized-volatility dataset is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise TargetConstructionError(
            "Missing required columns: "
            + ", ".join(missing)
        )