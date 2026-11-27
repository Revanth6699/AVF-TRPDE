from __future__ import annotations

import os

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.twelvedata.com/earliest_timestamp"
SYMBOL = "AAPL"
INTERVALS = ("1day", "5min")


def check_earliest_timestamp(symbol: str, interval: str, api_key: str) -> dict:
    """Query Twelve Data for the earliest available observation."""
    response = requests.get(
        BASE_URL,
        params={"symbol": symbol, "interval": interval, "apikey": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload)
    return payload


def main() -> None:
    """Run the optional live-provider availability check."""
    load_dotenv()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "TWELVE_DATA_API_KEY is not configured. Add it to the local .env "
            "before running this optional integration script."
        )

    for interval in INTERVALS:
        result = check_earliest_timestamp(SYMBOL, interval, api_key)
        print("=" * 60)
        print(f"Symbol   : {SYMBOL}")
        print(f"Interval : {interval}")
        print(f"Response : {result}")
        print("=" * 60)


if __name__ == "__main__":
    main()
