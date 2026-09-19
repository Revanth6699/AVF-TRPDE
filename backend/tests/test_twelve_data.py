import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is not configured.")

url = "https://api.twelvedata.com/time_series"

params = {
    "symbol": "AAPL",
    "interval": "1day",
    "outputsize": 5,
    "apikey": API_KEY,
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

if data.get("status") != "ok":
    raise RuntimeError(data)

print(data)