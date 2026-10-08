from __future__ import annotations

from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("data/local/AVF_LOCAL_INTRADAY.csv")
ASSET = "AAPL_LOCAL"
TRADING_DAYS = 130
BARS_PER_DAY = 4


def build_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 2)
    day = start
    trading_days = 0
    previous_close = 100.0
    global_bar = 0

    while trading_days < TRADING_DAYS:
        if day.weekday() < 5:
            session_start = datetime.combine(day.date(), time(9, 30))

            for bar in range(BARS_PER_DAY):
                timestamp = session_start + timedelta(minutes=60 * bar)

                # Deterministic synthetic intraday return.
                # This file is an integration fixture only, not research data.
                seasonal = 0.0007 * np.sin(global_bar / 7.0)
                cycle = 0.00035 * np.cos(global_bar / 3.0)
                drift = 0.00008
                shock = 0.0012 * ((global_bar % 11) - 5) / 5.0
                log_return = drift + seasonal + cycle + shock

                close = previous_close * np.exp(log_return)
                open_price = previous_close
                high = max(open_price, close) * (
                    1.0 + 0.0004 + 0.00015 * ((global_bar % 5) / 5.0)
                )
                low = min(open_price, close) * (
                    1.0 - 0.0004 - 0.00015 * ((global_bar % 4) / 4.0)
                )
                volume = 100_000 + 5_000 * (global_bar % 17)

                rows.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "asset_id": ASSET,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )

                previous_close = close
                global_bar += 1

            trading_days += 1

        day += timedelta(days=1)

    return pd.DataFrame(rows)


def main() -> None:
    dataframe = build_fixture()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT, index=False)

    print(f"created={OUTPUT}")
    print(f"rows={len(dataframe)}")
    print(f"asset={dataframe['asset_id'].nunique()}")
    print(f"start={dataframe['timestamp'].iloc[0]}")
    print(f"end={dataframe['timestamp'].iloc[-1]}")
    print(f"columns={','.join(dataframe.columns)}")


if __name__ == "__main__":
    main()
