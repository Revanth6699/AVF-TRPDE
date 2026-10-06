from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.data.loaders.csv import load_csv
from backend.app.data.loaders.parquet import load_parquet
from backend.app.data.loaders.twelve_data import load_twelve_data
from backend.app.research.volatility.realized import (
    compute_realized_volatility,
)
from backend.app.research.volatility.target import (
    create_next_day_target,
)


class DatasetResolutionError(RuntimeError):
    """Raised when a registered dataset cannot be resolved."""


def _parse_datetime(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        raise DatasetResolutionError(
            f"Invalid dataset timestamp: {value!r}"
        )

    return timestamp


def _load_raw_dataset(
    dataset: dict[str, Any],
    *,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    source = str(
        dataset.get("source", "")
    ).strip().lower()

    assets = tuple(
        str(asset).strip().upper()
        for asset in dataset.get("assets", [])
        if str(asset).strip()
    )

    if not assets:
        raise DatasetResolutionError(
            "Registered dataset contains no assets."
        )

    start_timestamp = _parse_datetime(
        dataset["start_timestamp"]
    )
    end_timestamp = _parse_datetime(
        dataset["end_timestamp"]
    )

    if start_timestamp > end_timestamp:
        raise DatasetResolutionError(
            "Dataset start timestamp is after end timestamp."
        )

    if source in {
        "csv",
        "parquet",
    }:
        path_value = (
            parameters.get("path")
            or parameters.get("data_path")
            or parameters.get("file_path")
        )

        if not path_value:
            raise DatasetResolutionError(
                f"Dataset source '{source}' requires "
                "parameters.path."
            )

        path = Path(str(path_value)).expanduser()

        if not path.exists():
            raise DatasetResolutionError(
                f"Dataset file does not exist: {path}"
            )

        if source == "csv":
            dataframe = load_csv(path)

        else:
            dataframe = load_parquet(path)

    elif source in {
        "twelve_data",
        "twelvedata",
        "twelve-data",
    }:
        interval = str(
            parameters.get(
                "interval",
                dataset.get("frequency", "daily"),
            )
        ).strip()

        # Dataset metadata uses project-level frequency names, while
        # Twelve Data uses provider-specific interval names.
        twelve_data_intervals = {
            "daily": "5min",
            "day": "5min",
            "1d": "5min",
            "intraday": "5min",
            "5min": "5min",
            "5m": "5min",
            "weekly": "1week",
            "week": "1week",
            "monthly": "1month",
            "month": "1month",
        }
        interval = twelve_data_intervals.get(
            interval.lower(),
            interval,
        )

        frames: list[pd.DataFrame] = []

        for asset in assets:
            frame = load_twelve_data(
                symbol=asset,
                interval=interval,
                start_date=start_timestamp.date(),
                end_date=end_timestamp.date(),
            )

            if frame.empty:
                raise DatasetResolutionError(
                    f"Twelve Data returned no rows for "
                    f"asset '{asset}'."
                )

            frames.append(frame)

        dataframe = pd.concat(
            frames,
            ignore_index=True,
        )

    else:
        raise DatasetResolutionError(
            "Unsupported dataset source "
            f"'{dataset.get('source')}'. "
            "Supported sources are: "
            "twelve_data, csv, parquet."
        )

    if dataframe.empty:
        raise DatasetResolutionError(
            "Resolved dataset contains no observations."
        )

    required_columns = {
        "timestamp",
        "asset_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = sorted(
        required_columns.difference(dataframe.columns)
    )

    if missing_columns:
        raise DatasetResolutionError(
            "Resolved dataset is missing required OHLCV "
            f"columns: {missing_columns}"
        )

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
        utc=True,
    )

    dataframe["asset_id"] = (
        dataframe["asset_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp",
            "asset_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    dataframe = dataframe.loc[
        dataframe["asset_id"].isin(assets)
    ].copy()

    dataframe = dataframe.loc[
        (dataframe["timestamp"] >= start_timestamp)
        & (dataframe["timestamp"] <= end_timestamp)
    ].copy()

    if dataframe.empty:
        raise DatasetResolutionError(
            "No observations remain after applying the "
            "registered dataset date and asset constraints."
        )

    dataframe = (
        dataframe
        .sort_values(
            ["asset_id", "timestamp"]
        )
        .drop_duplicates(
            ["asset_id", "timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataframe


def _prepare_returns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["close"] = pd.to_numeric(
        result["close"],
        errors="coerce",
    )

    result = result.loc[
        result["close"] > 0
    ].copy()

    result["return"] = (
        result
        .groupby("asset_id", sort=False)["close"]
        .transform(
            lambda series: np.log(series).diff()
        )
    )

    result = result.dropna(
        subset=["return"]
    ).copy()

    return result


def _merge_realized_volatility(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    realized = compute_realized_volatility(
        dataframe
    )

    if realized.empty:
        raise DatasetResolutionError(
            "Realized-volatility calculation returned "
            "no observations."
        )

    realized = realized.copy()

    realized["trading_date"] = pd.to_datetime(
        realized["trading_date"],
        errors="coerce",
    ).dt.normalize()

    dataframe = dataframe.copy()

    dataframe["trading_date"] = (
        dataframe["timestamp"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    result = dataframe.merge(
        realized[
            [
                "asset_id",
                "trading_date",
                "realized_variance",
                "realized_volatility",
                "observation_count",
            ]
        ],
        on=[
            "asset_id",
            "trading_date",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.drop(
        columns=["trading_date"]
    )

    return result


def _prepare_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    volatility = (
        dataframe[
            [
                "asset_id",
                "timestamp",
                "realized_volatility",
            ]
        ]
        .copy()
    )

    volatility["trading_date"] = (
        volatility["timestamp"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    daily = (
        volatility
        .groupby(
            [
                "asset_id",
                "trading_date",
            ],
            as_index=False,
        )[
            "realized_volatility"
        ]
        .last()
        .sort_values(
            [
                "asset_id",
                "trading_date",
            ]
        )
        .reset_index(drop=True)
    )

    targeted = create_next_day_target(
        daily[
            [
                "asset_id",
                "trading_date",
                "realized_volatility",
            ]
        ]
    )

    if targeted.empty:
        raise DatasetResolutionError(
            "Next-day target generation returned "
            "no observations."
        )

    targeted["trading_date"] = pd.to_datetime(
        targeted["trading_date"],
        errors="coerce",
    ).dt.normalize()

    dataframe["trading_date"] = (
        dataframe["timestamp"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    result = dataframe.merge(
        targeted[
            [
                "asset_id",
                "trading_date",
                "target",
            ]
        ],
        on=[
            "asset_id",
            "trading_date",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.drop(
        columns=["trading_date"]
    )

    return result


def prepare_research_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert canonical OHLCV data into the research-ready
    DataFrame required by WalkForwardRunner.
    """

    result = _prepare_returns(
        dataframe
    )

    result = _merge_realized_volatility(
        result
    )

    result = _prepare_target(
        result
    )

    result = result.dropna(
        subset=[
            "return",
            "realized_volatility",
            "target",
        ]
    ).copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "realized_variance",
        "realized_volatility",
        "target",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = result.dropna(
        subset=numeric_columns
    ).copy()

    result = result.loc[
        np.isfinite(
            result[
                numeric_columns
            ].to_numpy(
                dtype=float
            )
        ).all(axis=1)
    ].copy()

    result = result.loc[
        result["realized_volatility"] >= 0
    ].copy()

    result = result.loc[
        result["target"] >= 0
    ].copy()

    if result.empty:
        raise DatasetResolutionError(
            "Research preparation produced no valid "
            "observations."
        )

    result = (
        result
        .sort_values(
            [
                "timestamp",
                "asset_id",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def resolve_research_dataset(
    dataset: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Resolve a registered dataset and prepare it for
    AVF-TRPDE walk-forward research.
    """

    if not isinstance(dataset, dict):
        raise TypeError(
            "dataset must be a dictionary."
        )

    effective_parameters = (
        dict(parameters)
        if parameters is not None
        else {}
    )

    raw = _load_raw_dataset(
        dataset,
        parameters=effective_parameters,
    )

    return prepare_research_dataset(
        raw
    )