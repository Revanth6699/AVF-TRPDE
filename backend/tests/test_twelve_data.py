from __future__ import annotations

import os

import requests
from dotenv import load_dotenv


TIME_SERIES_URL = "https://api.twelvedata.com/time_series"


def main() -> None:
    """Run an optional live Twelve Data API smoke check."""
    load_dotenv()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "TWELVE_DATA_API_KEY is not configured. Add it to the local .env "
            "before running this optional integration script."
        )

    response = requests.get(
        TIME_SERIES_URL,
        params={
            "symbol": "AAPL",
            "interval": "1day",
            "outputsize": 5,
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(payload)
    print(payload)


if __name__ == "__main__":
    main()
