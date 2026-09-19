import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is not configured.")

BASE_URL = "https://api.twelvedata.com/earliest_timestamp"

SYMBOL = "AAPL"
INTERVALS = ["1day", "5min"]


def check_earliest_timestamp(symbol: str, interval: str) -> None:
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    print("=" * 60)
    print(f"Symbol   : {symbol}")
    print(f"Interval : {interval}")
    print(f"Response : {data}")
    print("=" * 60)


if __name__ == "__main__":
    for interval in INTERVALS:
        check_earliest_timestamp(SYMBOL, interval)