from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.data.loaders.csv import load_csv
from backend.app.data.loaders.parquet import load_parquet
from backend.app.data.loaders.twelve_data import load_twelve_data
from backend.app.research.volatility.realized import compute_realized_volatility


class DatasetResolutionError(RuntimeError):
    """Raised when a registered dataset cannot be resolved."""


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
DAILY_INTERVALS = {"daily": "1day", "day": "1day", "1d": "1day", "1day": "1day"}
INTRADAY_INTERVALS = {"intraday": "5min", "5m": "5min", "5min": "5min"}


def _parse_datetime(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise DatasetResolutionError(f"Invalid dataset timestamp: {value!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _inclusive_end(value: pd.Timestamp) -> pd.Timestamp:
    """Treat date-only end boundaries as inclusive for the entire exchange-local day."""
    if value.time() == time.min:
        return value + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return value


def _normalize_timestamps(values: pd.Series) -> pd.Series:
    def normalize(value: Any) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp

    return values.map(normalize)


def _load_file(path_value: str | Path) -> pd.DataFrame:
    path = Path(str(path_value)).expanduser()
    if not path.exists() or not path.is_file():
        raise DatasetResolutionError(f"Dataset file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".parquet", ".pq"}:
        return load_parquet(path)
    raise DatasetResolutionError(
        f"Unsupported dataset file extension '{suffix}'. Use CSV or Parquet."
    )


def _normalize_interval(frequency: str, parameters: dict[str, Any]) -> str:
    normalized = frequency.strip().lower()
    if normalized in DAILY_INTERVALS:
        # Daily source data must remain daily. Intraday sampling is a
        # separate input and is configured with intraday_interval.
        return DAILY_INTERVALS[normalized]
    if normalized in INTRADAY_INTERVALS:
        selected = str(parameters.get("intraday_interval", INTRADAY_INTERVALS[normalized])).lower()
        return INTRADAY_INTERVALS.get(selected, selected)
    selected = str(parameters.get("interval", normalized)).lower()
    return INTRADAY_INTERVALS.get(selected, selected)


def _validate_ohlcv_frame(dataframe: pd.DataFrame, assets: tuple[str, ...]) -> pd.DataFrame:
    required = {"timestamp", "asset_id", *OHLCV_COLUMNS}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise DatasetResolutionError(f"Resolved dataset is missing required OHLCV columns: {missing}")

    result = dataframe.copy()
    result["timestamp"] = _normalize_timestamps(result["timestamp"])
    result["asset_id"] = result["asset_id"].astype("string").str.strip().str.upper()

    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    invalid = result[list(OHLCV_COLUMNS)].isna().any(axis=1)
    invalid |= ~np.isfinite(result[list(OHLCV_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    invalid |= result["timestamp"].isna() | result["asset_id"].isna() | result["asset_id"].eq("")
    if invalid.any():
        raise DatasetResolutionError(
            f"Resolved OHLCV data contains {int(invalid.sum())} invalid row(s); "
            "invalid observations are not silently dropped."
        )

    result = result.loc[result["asset_id"].isin(assets)].copy()
    if result.empty:
        raise DatasetResolutionError("No observations remain for the registered assets.")

    duplicate = result.duplicated(["asset_id", "timestamp"], keep=False)
    if duplicate.any():
        raise DatasetResolutionError(
            f"Resolved OHLCV data contains {int(duplicate.sum())} duplicate asset/timestamp rows."
        )

    return result.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(drop=True)


def _load_raw_dataset(
    dataset: dict[str, Any],
    *,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    source = str(dataset.get("source", "")).strip().lower()
    assets = tuple(
        str(asset).strip().upper()
        for asset in dataset.get("assets", [])
        if str(asset).strip()
    )
    if not assets:
        raise DatasetResolutionError("Registered dataset contains no assets.")

    start = _parse_datetime(dataset["start_timestamp"])
    end = _inclusive_end(_parse_datetime(dataset["end_timestamp"]))
    if start > end:
        raise DatasetResolutionError("Dataset start timestamp is after end timestamp.")

    if source in {"csv", "parquet"}:
        path_value = parameters.get("path") or parameters.get("data_path") or parameters.get("file_path")
        if not path_value:
            raise DatasetResolutionError(f"Dataset source '{source}' requires parameters.path.")
        dataframe = _load_file(path_value)
    elif source in {"twelve_data", "twelvedata", "twelve-data"}:
        interval = _normalize_interval(str(dataset.get("frequency", "daily")), parameters)
        frames = []
        for asset in assets:
            frame = load_twelve_data(
                symbol=asset,
                interval=interval,
                start_date=start.to_pydatetime(),
                end_date=end.to_pydatetime(),
            )
            if frame.empty:
                raise DatasetResolutionError(
                    f"Twelve Data returned no rows for asset '{asset}' at interval '{interval}'."
                )
            frames.append(frame)
        dataframe = pd.concat(frames, ignore_index=True)
    else:
        raise DatasetResolutionError(
            f"Unsupported dataset source '{dataset.get('source')}'. "
            "Supported sources are: twelve_data, csv, parquet."
        )

    result = _validate_ohlcv_frame(dataframe, assets)
    result = result.loc[(result["timestamp"] >= start) & (result["timestamp"] <= end)].copy()
    if result.empty:
        raise DatasetResolutionError(
            "No observations remain after applying the registered dataset date and asset constraints."
        )
    return result.reset_index(drop=True)


def _load_intraday_dataset(
    dataset: dict[str, Any],
    *,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    assets = tuple(str(asset).strip().upper() for asset in dataset.get("assets", []) if str(asset).strip())
    path_value = parameters.get("intraday_path")
    if path_value:
        frame = _load_file(path_value)
    elif str(dataset.get("source", "")).strip().lower() in {"twelve_data", "twelvedata", "twelve-data"}:
        if not parameters.get("intraday_start_timestamp"):
            raise DatasetResolutionError(
                "Daily Twelve Data input requires parameters.intraday_start_timestamp "
                "to define the separate intraday realized-volatility history."
            )
        start = _parse_datetime(parameters["intraday_start_timestamp"])
        end = _inclusive_end(_parse_datetime(parameters.get("intraday_end_timestamp", dataset["end_timestamp"])))
        interval = _normalize_interval("intraday", parameters)
        if start > end:
            raise DatasetResolutionError("Intraday start timestamp is after intraday end timestamp.")
        frames = []
        for asset in assets:
            frame = load_twelve_data(
                symbol=asset,
                interval=interval,
                start_date=start.to_pydatetime(),
                end_date=end.to_pydatetime(),
            )
            if frame.empty:
                raise DatasetResolutionError(
                    f"Twelve Data returned no intraday rows for asset '{asset}'."
                )
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
    else:
        raise DatasetResolutionError(
            "Daily OHLCV input needs a separate intraday dataset. "
            "Supply parameters.intraday_path (CSV/Parquet), or use Twelve Data "
            "with parameters.intraday_start_timestamp and parameters.intraday_interval."
        )

    result = _validate_ohlcv_frame(frame, assets)
    start = _parse_datetime(parameters.get("intraday_start_timestamp", dataset["start_timestamp"]))
    end = _inclusive_end(_parse_datetime(parameters.get("intraday_end_timestamp", dataset["end_timestamp"])))
    result = result.loc[(result["timestamp"] >= start) & (result["timestamp"] <= end)].copy()
    if result.empty:
        raise DatasetResolutionError("Intraday dataset has no observations in the configured date range.")
    return result.reset_index(drop=True)


def _daily_ohlcv(intraday: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars into one exchange-local OHLCV row per asset/day."""
    frame = intraday.copy()
    frame["trading_date"] = frame["timestamp"].dt.normalize()
    daily = (
        frame.sort_values(["asset_id", "timestamp"], kind="mergesort")
        .groupby(["asset_id", "trading_date"], as_index=False, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"trading_date": "timestamp"})
    )
    return daily[["timestamp", "asset_id", *OHLCV_COLUMNS]]


def _prepare_returns(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result = result.sort_values(["asset_id", "timestamp"], kind="mergesort")
    result["return"] = result.groupby("asset_id", sort=False)["close"].transform(
        lambda values: np.log(values).diff()
    )
    return result.dropna(subset=["return"]).reset_index(drop=True)


def _merge_realized_volatility(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
) -> pd.DataFrame:
    try:
        realized = compute_realized_volatility(intraday)
    except (TypeError, ValueError) as exc:
        raise DatasetResolutionError(
            f"Intraday realized-volatility calculation failed: {exc}"
        ) from exc

    if realized.empty:
        raise DatasetResolutionError("Realized-volatility calculation returned no observations.")

    realized = realized.copy()
    realized["trading_date"] = _normalize_timestamps(realized["trading_date"]).dt.normalize()
    result = daily.copy()
    result["trading_date"] = result["timestamp"].dt.normalize()
    merged = result.merge(
        realized[["asset_id", "trading_date", "realized_variance", "realized_volatility", "observation_count"]],
        on=["asset_id", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    return merged.drop(columns=["trading_date"])


def prepare_research_dataset(
    dataframe: pd.DataFrame,
    *,
    intraday_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build one supervised observation per asset/trading day.

    Intraday bars are used only to calculate that day's realized variance.
    Daily OHLCV features are observed at the close of day t and the target
    is realized volatility on the next available trading day t+1.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise DatasetResolutionError("Input OHLCV dataset must be a non-empty DataFrame.")

    assets = tuple(sorted(dataframe["asset_id"].astype(str).str.upper().unique()))
    main = _validate_ohlcv_frame(dataframe, assets)
    main["trading_date"] = main["timestamp"].dt.normalize()
    is_intraday = main.duplicated(["asset_id", "trading_date"], keep=False).any()

    if is_intraday:
        intraday = main
        daily = _daily_ohlcv(intraday)
    else:
        daily = main[["timestamp", "asset_id", *OHLCV_COLUMNS]].copy()
        if intraday_dataframe is None:
            raise DatasetResolutionError(
                "Daily OHLCV data cannot produce intraday realized variance. "
                "Provide intraday_dataframe or parameters.intraday_path."
            )
        intraday = _validate_ohlcv_frame(intraday_dataframe, assets)

    daily = daily.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    daily = _prepare_returns(daily)
    if daily.empty:
        raise DatasetResolutionError("Daily return preparation produced no observations.")

    daily = _merge_realized_volatility(daily, intraday)
    daily = daily.sort_values(["asset_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Shift across the complete daily calendar. Missing next-day intraday
    # observations remain missing targets; they are never skipped over.
    grouped = daily.groupby("asset_id", sort=False)
    daily["target"] = grouped["realized_volatility"].shift(-1)
    daily["target_timestamp"] = grouped["timestamp"].shift(-1)

    numeric_columns = [
        "open", "high", "low", "close", "volume", "return",
        "realized_variance", "realized_volatility", "target",
    ]
    result = daily.dropna(subset=numeric_columns + ["target_timestamp"]).copy()
    if result.empty:
        raise DatasetResolutionError(
            "Research preparation produced no complete next-day target observations. "
            "Check the overlap between daily and intraday datasets."
        )

    values = result[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DatasetResolutionError("Research dataset contains non-finite numeric values.")
    if (result[["realized_variance", "realized_volatility", "target"]] < 0).any().any():
        raise DatasetResolutionError("Realized variance and volatility targets cannot be negative.")

    result = result.sort_values(["timestamp", "asset_id"], kind="mergesort").reset_index(drop=True)
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise DatasetResolutionError("Research dataset must contain one row per asset/trading day.")
    return result


def resolve_research_dataset(
    dataset: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Resolve a registered dataset and prepare it for daily walk-forward research."""
    if not isinstance(dataset, dict):
        raise TypeError("dataset must be a dictionary.")

    effective_parameters = dict(parameters or {})
    raw = _load_raw_dataset(dataset, parameters=effective_parameters)

    frequency = str(dataset.get("frequency", "daily")).strip().lower()
    if frequency in INTRADAY_INTERVALS or frequency in {"1min", "15min", "30min", "45min", "1h", "2h", "4h", "8h"}:
        return prepare_research_dataset(raw)

    intraday = _load_intraday_dataset(dataset, parameters=effective_parameters)
    return prepare_research_dataset(raw, intraday_dataframe=intraday)
