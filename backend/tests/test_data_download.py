from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Allow the test to be executed directly with:
# python backend\tests\test_data_download.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.data.canonical import (
    assert_canonical_schema,
    to_canonical,
)
from backend.app.data.fingerprint import (
    fingerprint_dataframe,
    fingerprint_parquet,
)
from backend.app.data.loaders.twelve_data import (
    TwelveDataClient,
)


TEST_SYMBOL = "AAPL"

# Small ranges are intentional.
# This test verifies the pipeline, not full historical acquisition.
DAILY_START = "2026-09-01"
DAILY_END = "2026-09-16"

INTRADAY_START = "2026-09-15 09:30:00"
INTRADAY_END = "2026-09-15 16:00:00"


def _require_api_key() -> None:
    """
    Skip integration tests when the Twelve Data API key
    is not configured.
    """

    if not os.getenv("TWELVE_DATA_API_KEY"):
        pytest.skip(
            "TWELVE_DATA_API_KEY is not configured."
        )


def test_daily_download() -> None:
    """
    Verify that daily OHLCV data can be downloaded successfully.
    """

    _require_api_key()

    client = TwelveDataClient()

    dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert not dataframe.empty

    expected_columns = [
        "timestamp",
        "asset_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert list(dataframe.columns) == expected_columns

    assert set(dataframe["asset_id"]) == {
        TEST_SYMBOL
    }

    assert dataframe["timestamp"].notna().all()

    assert dataframe[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].notna().all().all()


def test_intraday_download() -> None:
    """
    Verify that intraday OHLCV data can be downloaded successfully.
    """

    _require_api_key()

    client = TwelveDataClient()

    dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="5min",
        start_date=INTRADAY_START,
        end_date=INTRADAY_END,
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert not dataframe.empty

    expected_columns = [
        "timestamp",
        "asset_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert list(dataframe.columns) == expected_columns

    assert set(dataframe["asset_id"]) == {
        TEST_SYMBOL
    }

    assert dataframe["timestamp"].notna().all()

    assert dataframe[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].notna().all().all()


def test_daily_canonicalization() -> None:
    """
    Verify that downloaded daily data can pass through
    the canonical data contract.
    """

    _require_api_key()

    client = TwelveDataClient()

    raw_dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    canonical = to_canonical(
        raw_dataframe,
        validate=True,
    )

    assert not canonical.empty

    assert_canonical_schema(
        canonical
    )

    assert canonical["asset_id"].eq(
        TEST_SYMBOL
    ).all()

    assert canonical[
        "timestamp"
    ].is_monotonic_increasing


def test_daily_fingerprint_is_deterministic() -> None:
    """
    Verify that the same downloaded dataset produces
    the same fingerprint.
    """

    _require_api_key()

    client = TwelveDataClient()

    dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    first = fingerprint_dataframe(
        dataframe
    )

    second = fingerprint_dataframe(
        dataframe
    )

    assert first.fingerprint == second.fingerprint

    assert first.algorithm == "sha256"

    assert first.version == "1"

    assert first.row_count == len(
        dataframe
    )

    assert first.asset_count == 1

    assert first.assets == (
        TEST_SYMBOL,
    )


def test_daily_parquet_round_trip(
    tmp_path: Path,
) -> None:
    """
    Verify:

        API data
            ↓
        canonical data
            ↓
        Parquet
            ↓
        reload
            ↓
        same fingerprint
    """

    _require_api_key()

    client = TwelveDataClient()

    dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    canonical = to_canonical(
        dataframe,
        validate=True,
    )

    original_fingerprint = fingerprint_dataframe(
        canonical
    )

    output_path = (
        tmp_path
        / "AAPL_daily.parquet"
    )

    canonical.to_parquet(
        output_path,
        index=False,
    )

    assert output_path.exists()

    reloaded = pd.read_parquet(
        output_path
    )

    assert_canonical_schema(
        reloaded
    )

    reloaded_fingerprint = fingerprint_dataframe(
        reloaded
    )

    assert (
        original_fingerprint.fingerprint
        == reloaded_fingerprint.fingerprint
    )


def test_parquet_fingerprint_matches_dataframe(
    tmp_path: Path,
) -> None:
    """
    Verify that fingerprint_parquet() produces the
    same dataset fingerprint as fingerprint_dataframe().
    """

    _require_api_key()

    client = TwelveDataClient()

    dataframe = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    canonical = to_canonical(
        dataframe,
        validate=True,
    )

    dataframe_fingerprint = fingerprint_dataframe(
        canonical
    )

    output_path = (
        tmp_path
        / "AAPL_daily.parquet"
    )

    canonical.to_parquet(
        output_path,
        index=False,
    )

    parquet_fingerprint = fingerprint_parquet(
        output_path
    )

    assert (
        dataframe_fingerprint.fingerprint
        == parquet_fingerprint.fingerprint
    )


def run_direct_smoke_test() -> None:
    """
    Direct execution helper.

    This allows:

        python backend\\tests\\test_data_download.py

    without requiring pytest.
    """

    _require_api_key()

    client = TwelveDataClient()

    print("=" * 60)
    print("AVF-TRPDE DATA DOWNLOAD SMOKE TEST")
    print("=" * 60)

    print("\n[1] Downloading daily data...")

    daily = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="1day",
        start_date=DAILY_START,
        end_date=DAILY_END,
    )

    print(
        f"Daily rows: {len(daily)}"
    )

    print(
        f"Daily range: "
        f"{daily['timestamp'].min()} "
        f"→ "
        f"{daily['timestamp'].max()}"
    )

    print("\n[2] Downloading intraday data...")

    intraday = client.get_time_series(
        symbol=TEST_SYMBOL,
        interval="5min",
        start_date=INTRADAY_START,
        end_date=INTRADAY_END,
    )

    print(
        f"Intraday rows: {len(intraday)}"
    )

    print(
        f"Intraday range: "
        f"{intraday['timestamp'].min()} "
        f"→ "
        f"{intraday['timestamp'].max()}"
    )

    print("\n[3] Canonicalizing daily data...")

    canonical = to_canonical(
        daily,
        validate=True,
    )

    assert_canonical_schema(
        canonical
    )

    print(
        "Canonical schema: OK"
    )

    print("\n[4] Generating fingerprint...")

    fingerprint = fingerprint_dataframe(
        canonical
    )

    print(
        f"Fingerprint: "
        f"{fingerprint.fingerprint}"
    )

    print(
        f"Rows: {fingerprint.row_count}"
    )

    print(
        f"Assets: {fingerprint.assets}"
    )

    print("\n" + "=" * 60)
    print("DATA DOWNLOAD SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_direct_smoke_test()