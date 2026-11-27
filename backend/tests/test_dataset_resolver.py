from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.research.integration.dataset_resolver import (
    DatasetResolutionError,
    _normalize_interval,
    prepare_research_dataset,
)


def _intraday_frame(days: int = 4) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    for day_index, day in enumerate(dates):
        base = 100.0 + day_index
        for bar_index, close in enumerate((base, base * 1.01, base * 1.02)):
            rows.append(
                {
                    "timestamp": day + pd.Timedelta(hours=10 + bar_index),
                    "asset_id": "AAPL",
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _daily_frame(days: int = 4) -> pd.DataFrame:
    intraday = _intraday_frame(days)
    intraday["trading_date"] = intraday["timestamp"].dt.normalize()
    return (
        intraday.sort_values("timestamp")
        .groupby(["asset_id", "trading_date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"trading_date": "timestamp"})
    )


def test_daily_frequency_maps_to_daily_provider_interval() -> None:
    assert _normalize_interval("daily", {}) == "1day"
    assert _normalize_interval("day", {}) == "1day"
    assert _normalize_interval("1d", {}) == "1day"
    assert _normalize_interval("intraday", {}) == "5min"
    assert _normalize_interval("daily", {"interval": "5min"}) == "1day"
    assert _normalize_interval("intraday", {"intraday_interval": "5m"}) == "5min"


def test_intraday_input_becomes_one_daily_observation_per_asset() -> None:
    intraday = _intraday_frame()
    result = prepare_research_dataset(intraday)

    assert not result.empty
    assert not result.duplicated(["asset_id", "timestamp"]).any()
    assert result["timestamp"].dt.normalize().equals(result["timestamp"])
    assert result["target_timestamp"].gt(result["timestamp"]).all()
    assert np.isfinite(result[["realized_variance", "realized_volatility", "target"]]).all().all()

    # Four source trading days: the first has no lagged daily return and
    # the final has no next-day target, leaving two supervised observations.
    assert len(result) == 2
    assert result["timestamp"].is_monotonic_increasing


def test_daily_input_requires_separate_intraday_observations() -> None:
    with pytest.raises(DatasetResolutionError, match="cannot produce intraday realized variance"):
        prepare_research_dataset(_daily_frame())


def test_missing_intraday_day_does_not_skip_forward_to_a_later_target() -> None:
    daily = _daily_frame()
    intraday = _intraday_frame()
    missing_day = pd.Timestamp("2024-01-04")
    intraday = intraday.loc[intraday["timestamp"].dt.normalize() != missing_day]

    # The full daily calendar is retained during target shifting. Because
    # day 3 has no RV, day 2 cannot silently be labelled with day 4's RV.
    with pytest.raises(DatasetResolutionError, match="no complete next-day target"):
        prepare_research_dataset(daily, intraday_dataframe=intraday)
