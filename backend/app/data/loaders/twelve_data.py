from __future__ import annotations

import json
import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


load_dotenv()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
TIME_SERIES_ENDPOINT = f"{TWELVE_DATA_BASE_URL}/time_series"

MAX_POINTS_PER_REQUEST = 5_000
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0

SUPPORTED_INTERVALS = {
    "1min",
    "5min",
    "15min",
    "30min",
    "45min",
    "1h",
    "2h",
    "4h",
    "8h",
    "1day",
    "1week",
    "1month",
}

REQUIRED_COLUMNS = [
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class TwelveDataError(RuntimeError):
    """Base exception for Twelve Data acquisition failures."""


class TwelveDataConfigurationError(TwelveDataError):
    """Raised when the Twelve Data configuration is invalid."""


class TwelveDataRequestError(TwelveDataError):
    """Raised when a Twelve Data request fails."""


@dataclass(frozen=True)
class TwelveDataConfig:
    """Configuration for the Twelve Data API client."""

    api_key: str
    base_url: str = TWELVE_DATA_BASE_URL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS

    @classmethod
    def from_environment(cls) -> "TwelveDataConfig":
        api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not api_key:
            raise TwelveDataConfigurationError(
                "TWELVE_DATA_API_KEY is not configured."
            )

        return cls(api_key=api_key.strip())


class TwelveDataClient:
    """
    Client for retrieving historical OHLCV data from Twelve Data.

    The client is intentionally focused on data acquisition.
    Validation, canonicalization, and fingerprinting are handled
    by separate modules in the data layer.
    """

    def __init__(self, config: TwelveDataConfig | None = None) -> None:
        self.config = config or TwelveDataConfig.from_environment()

    def get_time_series(
        self,
        symbol: str,
        interval: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        """
        Retrieve a complete historical time series between two boundaries.

        Large date ranges are recursively split so that no individual
        API request exceeds Twelve Data's 5,000-point response limit.

        Parameters
        ----------
        symbol:
            Twelve Data symbol, e.g. "AAPL".
        interval:
            Twelve Data interval, e.g. "1day" or "5min".
        start_date:
            Inclusive lower date/time boundary.
        end_date:
            Inclusive upper date/time boundary.

        Returns
        -------
        pandas.DataFrame
            Canonical acquisition DataFrame containing:

            timestamp
            asset_id
            open
            high
            low
            close
            volume
        """

        symbol = self._normalize_symbol(symbol)
        interval = self._normalize_interval(interval)

        start = self._parse_datetime(start_date)
        end = self._parse_datetime(end_date)

        if start > end:
            raise ValueError("start_date must be earlier than or equal to end_date.")

        chunks = self._fetch_range_recursive(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )

        if not chunks:
            return self._empty_dataframe()

        dataframe = pd.concat(chunks, ignore_index=True)

        return self._normalize_dataframe(
            dataframe=dataframe,
            symbol=symbol,
        )

    def _fetch_range_recursive(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[pd.DataFrame]:
        """
        Fetch a date range.

        If the API returns the maximum number of points, split the
        interval into two smaller ranges and fetch each recursively.
        """

        response = self._request_time_series(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )

        values = response.get("values", [])

        if not values:
            return []

        dataframe = self._values_to_dataframe(
            values=values,
            symbol=symbol,
        )

        if len(dataframe) < MAX_POINTS_PER_REQUEST:
            return [dataframe]

        if start == end:
            raise TwelveDataRequestError(
                "A single timestamp range returned the maximum number "
                "of observations. Unable to split the requested range further."
            )

        midpoint = start + (end - start) / 2

        left_end = midpoint
        right_start = midpoint + timedelta(microseconds=1)

        left_chunks = self._fetch_range_recursive(
            symbol=symbol,
            interval=interval,
            start=start,
            end=left_end,
        )

        right_chunks = self._fetch_range_recursive(
            symbol=symbol,
            interval=interval,
            start=right_start,
            end=end,
        )

        return left_chunks + right_chunks

    def _request_time_series(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Execute one Twelve Data time-series request."""

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": self._format_datetime(start, interval),
            "end_date": self._format_datetime(end, interval),
            "apikey": self.config.api_key,
        }

        url = f"{TIME_SERIES_ENDPOINT}?{urlencode(params)}"

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                request = Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "AVF-TRPDE/1.0",
                    },
                    method="GET",
                )

                with urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    status_code = response.status
                    raw_body = response.read().decode("utf-8")

                payload = self._parse_json_response(raw_body)

                if status_code != 200:
                    raise TwelveDataRequestError(
                        self._format_api_error(
                            status_code=status_code,
                            payload=payload,
                        )
                    )

                if payload.get("status") == "error":
                    raise TwelveDataRequestError(
                        self._format_api_error(
                            status_code=status_code,
                            payload=payload,
                        )
                    )

                return payload

            except HTTPError as exc:
                last_error = self._handle_http_error(exc)

                if not self._is_retryable_status(exc.code):
                    raise last_error from exc

            except URLError as exc:
                last_error = TwelveDataRequestError(
                    f"Network error while requesting Twelve Data: {exc.reason}"
                )

            except TimeoutError as exc:
                last_error = TwelveDataRequestError(
                    "Twelve Data request timed out."
                )

            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2**attempt)
                time.sleep(delay)

        if last_error is not None:
            raise last_error

        raise TwelveDataRequestError(
            "Twelve Data request failed without a specific error."
        )

    @staticmethod
    def _parse_json_response(raw_body: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise TwelveDataRequestError(
                "Twelve Data returned an invalid JSON response."
            ) from exc

        if not isinstance(payload, dict):
            raise TwelveDataRequestError(
                "Twelve Data returned an unexpected response structure."
            )

        return payload

    @staticmethod
    def _handle_http_error(exc: HTTPError) -> TwelveDataRequestError:
        try:
            raw_body = exc.read().decode("utf-8")
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}

        message = payload.get("message", str(exc))
        code = payload.get("code")

        details = f"HTTP {exc.code}: {message}"

        if code is not None:
            details = f"{details} (code={code})"

        return TwelveDataRequestError(details)

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _format_api_error(
        status_code: int,
        payload: dict[str, Any],
    ) -> str:
        message = payload.get(
            "message",
            "Unknown Twelve Data API error.",
        )

        code = payload.get("code")

        if code is not None:
            return (
                f"Twelve Data API error "
                f"(HTTP {status_code}, code={code}): {message}"
            )

        return f"Twelve Data API error (HTTP {status_code}): {message}"

    @staticmethod
    def _values_to_dataframe(
        values: list[dict[str, Any]],
        symbol: str,
    ) -> pd.DataFrame:
        dataframe = pd.DataFrame(values)

        if dataframe.empty:
            return TwelveDataClient._empty_dataframe()

        required_api_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            column
            for column in required_api_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise TwelveDataRequestError(
                "Twelve Data response is missing required fields: "
                + ", ".join(missing)
            )

        dataframe = dataframe.rename(
            columns={
                "datetime": "timestamp",
            }
        )

        dataframe["asset_id"] = symbol

        return dataframe[
            [
                "timestamp",
                "asset_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

    @staticmethod
    def _normalize_dataframe(
        dataframe: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        dataframe["asset_id"] = symbol

        dataframe = dataframe.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        dataframe = dataframe.drop_duplicates(
            subset=["timestamp", "asset_id"],
            keep="last",
        )

        dataframe = dataframe.sort_values(
            by=["timestamp", "asset_id"],
            ascending=True,
        )

        dataframe = dataframe.reset_index(drop=True)

        return dataframe[REQUIRED_COLUMNS]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()

        if not normalized:
            raise ValueError("symbol must not be empty.")

        return normalized

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        normalized = interval.strip().lower()

        if normalized not in SUPPORTED_INTERVALS:
            supported = ", ".join(sorted(SUPPORTED_INTERVALS))
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported intervals: {supported}"
            )

        return normalized

    @staticmethod
    def _parse_datetime(
        value: str | date | datetime,
    ) -> datetime:
        if isinstance(value, datetime):
            return value

        if isinstance(value, date):
            return datetime.combine(
                value,
                datetime.min.time(),
            )

        if isinstance(value, str):
            normalized = value.strip()

            if not normalized:
                raise ValueError("Date value must not be empty.")

            try:
                parsed = datetime.fromisoformat(
                    normalized.replace("Z", "+00:00")
                )
            except ValueError:
                try:
                    parsed_date = date.fromisoformat(normalized)
                    parsed = datetime.combine(
                        parsed_date,
                        datetime.min.time(),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid date/datetime value: {value}"
                    ) from exc

            return parsed

        raise TypeError(
            "Date values must be str, date, or datetime."
        )

    @staticmethod
    def _format_datetime(
        value: datetime,
        interval: str,
    ) -> str:
        if interval in {"1day", "1week", "1month"}:
            return value.strftime("%Y-%m-%d")

        return value.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "asset_id": pd.Series(dtype="string"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="float64"),
            }
        )


def load_twelve_data(
    symbol: str,
    interval: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> pd.DataFrame:
    """
    Convenience function for historical Twelve Data acquisition.
    """

    client = TwelveDataClient()

    return client.get_time_series(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )