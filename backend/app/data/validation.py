from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
)

NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class DataValidationError(ValueError):
    """Raised when a dataset violates the AVF-TRPDE data contract."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)

        message = "Dataset validation failed:\n" + "\n".join(
            f"- {error}" for error in self.errors
        )

        super().__init__(message)


@dataclass
class ValidationReport:
    """Structured result of dataset validation."""

    valid: bool
    row_count: int
    asset_count: int
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None
    assets: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise DataValidationError when validation failed."""

        if not self.valid:
            raise DataValidationError(self.errors)


def validate_ohlcv(
    dataframe: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> ValidationReport:
    """
    Validate an AVF-TRPDE OHLCV dataset.

    Expected schema:

        timestamp
        asset_id
        open
        high
        low
        close
        volume

    Validation covers:

    - DataFrame structure
    - required columns
    - empty dataset
    - missing values
    - timestamp validity
    - asset identifiers
    - numeric OHLCV fields
    - positive prices
    - non-negative volume
    - OHLC price relationships
    - duplicate observations
    - chronological ordering
    - invalid infinite numeric values

    The function does not mutate the supplied DataFrame.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    row_count = len(dataframe)

    if row_count == 0:
        errors.append("Dataset is empty.")

        report = ValidationReport(
            valid=False,
            row_count=0,
            asset_count=0,
            start_timestamp=None,
            end_timestamp=None,
            assets=[],
            errors=errors,
            warnings=warnings,
        )

        if raise_on_error:
            report.raise_if_invalid()

        return report

    # ---------------------------------------------------------
    # 1. REQUIRED COLUMNS
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # If the schema is incomplete, several later checks cannot
    # be performed safely.
    if missing_columns:
        report = ValidationReport(
            valid=False,
            row_count=row_count,
            asset_count=0,
            start_timestamp=None,
            end_timestamp=None,
            assets=[],
            errors=errors,
            warnings=warnings,
        )

        if raise_on_error:
            report.raise_if_invalid()

        return report

    # ---------------------------------------------------------
    # 2. COPY
    # ---------------------------------------------------------

    df = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    # ---------------------------------------------------------
    # 3. TIMESTAMP VALIDATION
    # ---------------------------------------------------------

    timestamp_series = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        timestamp_series.isna().sum()
    )

    if invalid_timestamp_count > 0:
        errors.append(
            "timestamp contains "
            f"{invalid_timestamp_count} invalid or missing values."
        )

    start_timestamp: pd.Timestamp | None = None
    end_timestamp: pd.Timestamp | None = None

    valid_timestamps = timestamp_series.dropna()

    if not valid_timestamps.empty:
        start_timestamp = valid_timestamps.min()
        end_timestamp = valid_timestamps.max()

    # ---------------------------------------------------------
    # 4. ASSET ID VALIDATION
    # ---------------------------------------------------------

    asset_series = df["asset_id"]

    missing_asset_count = int(
        asset_series.isna().sum()
    )

    if missing_asset_count > 0:
        errors.append(
            "asset_id contains "
            f"{missing_asset_count} missing values."
        )

    empty_asset_count = int(
        asset_series.astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_asset_count > 0:
        errors.append(
            "asset_id contains "
            f"{empty_asset_count} empty values."
        )

    assets = sorted(
        {
            str(asset).strip()
            for asset in asset_series.dropna()
            if str(asset).strip()
        }
    )

    # ---------------------------------------------------------
    # 5. NUMERIC TYPE / CONVERSION VALIDATION
    # ---------------------------------------------------------

    numeric_data: dict[str, pd.Series] = {}

    for column in NUMERIC_COLUMNS:
        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        numeric_data[column] = numeric_series

        invalid_count = int(
            numeric_series.isna().sum()
        )

        original_missing_count = int(
            df[column].isna().sum()
        )

        conversion_failure_count = (
            invalid_count - original_missing_count
        )

        if conversion_failure_count > 0:
            errors.append(
                f"{column} contains "
                f"{conversion_failure_count} non-numeric values."
            )

        if original_missing_count > 0:
            errors.append(
                f"{column} contains "
                f"{original_missing_count} missing values."
            )

    # ---------------------------------------------------------
    # 6. INFINITE VALUE VALIDATION
    # ---------------------------------------------------------

    for column, series in numeric_data.items():
        finite_mask = series.notna() & ~series.isin(
            [float("inf"), float("-inf")]
        )

        infinite_count = int(
            (~finite_mask & series.notna()).sum()
        )

        if infinite_count > 0:
            errors.append(
                f"{column} contains "
                f"{infinite_count} infinite values."
            )

    # ---------------------------------------------------------
    # 7. POSITIVE PRICE VALIDATION
    # ---------------------------------------------------------

    for column in PRICE_COLUMNS:
        series = numeric_data[column]

        invalid_price_count = int(
            (series <= 0).sum()
        )

        if invalid_price_count > 0:
            errors.append(
                f"{column} contains "
                f"{invalid_price_count} values that are "
                "less than or equal to zero."
            )

    # ---------------------------------------------------------
    # 8. VOLUME VALIDATION
    # ---------------------------------------------------------

    volume = numeric_data["volume"]

    negative_volume_count = int(
        (volume < 0).sum()
    )

    if negative_volume_count > 0:
        errors.append(
            "volume contains "
            f"{negative_volume_count} negative values."
        )

    # ---------------------------------------------------------
    # 9. OHLC RELATIONSHIP VALIDATION
    # ---------------------------------------------------------

    open_price = numeric_data["open"]
    high_price = numeric_data["high"]
    low_price = numeric_data["low"]
    close_price = numeric_data["close"]

    valid_price_rows = (
        open_price.notna()
        & high_price.notna()
        & low_price.notna()
        & close_price.notna()
    )

    invalid_high_low = (
        valid_price_rows
        & (high_price < low_price)
    )

    high_low_count = int(
        invalid_high_low.sum()
    )

    if high_low_count > 0:
        errors.append(
            "OHLC relationship violation: "
            f"high < low in {high_low_count} rows."
        )

    invalid_open_range = (
        valid_price_rows
        & (
            (open_price > high_price)
            | (open_price < low_price)
        )
    )

    open_range_count = int(
        invalid_open_range.sum()
    )

    if open_range_count > 0:
        errors.append(
            "OHLC relationship violation: "
            "open lies outside the high-low range in "
            f"{open_range_count} rows."
        )

    invalid_close_range = (
        valid_price_rows
        & (
            (close_price > high_price)
            | (close_price < low_price)
        )
    )

    close_range_count = int(
        invalid_close_range.sum()
    )

    if close_range_count > 0:
        errors.append(
            "OHLC relationship violation: "
            "close lies outside the high-low range in "
            f"{close_range_count} rows."
        )

    # ---------------------------------------------------------
    # 10. DUPLICATE OBSERVATION VALIDATION
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["timestamp", "asset_id"],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        errors.append(
            "Duplicate timestamp/asset observations detected: "
            f"{duplicate_count} rows."
        )

    # ---------------------------------------------------------
    # 11. CHRONOLOGICAL ORDER VALIDATION
    # ---------------------------------------------------------

    timestamp_check = timestamp_series

    if timestamp_check.notna().all():
        ordering_frame = pd.DataFrame(
            {
                "timestamp": timestamp_check,
                "asset_id": df["asset_id"].astype("string"),
            }
        )

        for asset_id, asset_frame in ordering_frame.groupby(
            "asset_id",
            dropna=False,
        ):
            if not asset_frame["timestamp"].is_monotonic_increasing:
                errors.append(
                    "Timestamps are not chronologically ordered "
                    f"for asset '{asset_id}'."
                )

    # ---------------------------------------------------------
    # 12. TIMESTAMP DUPLICATES WITHIN EACH ASSET
    # ---------------------------------------------------------

    duplicate_timestamps = (
        df.groupby("asset_id", dropna=False)["timestamp"]
        .apply(lambda values: values.duplicated().sum())
    )

    duplicate_timestamp_total = int(
        duplicate_timestamps.sum()
    )

    if duplicate_timestamp_total > 0:
        errors.append(
            "Repeated timestamps detected within asset series: "
            f"{duplicate_timestamp_total} observations."
        )

    # ---------------------------------------------------------
    # 13. DATASET-LEVEL WARNINGS
    # ---------------------------------------------------------

    if len(assets) == 1:
        warnings.append(
            "Dataset contains one asset."
        )

    if "volume" in df.columns:
        zero_volume_count = int(
            (numeric_data["volume"] == 0).sum()
        )

        if zero_volume_count > 0:
            warnings.append(
                f"Dataset contains {zero_volume_count} "
                "zero-volume observations."
            )

    # ---------------------------------------------------------
    # 14. REPORT
    # ---------------------------------------------------------

    report = ValidationReport(
        valid=len(errors) == 0,
        row_count=row_count,
        asset_count=len(assets),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        assets=assets,
        errors=errors,
        warnings=warnings,
    )

    if raise_on_error:
        report.raise_if_invalid()

    return report


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that all AVF-TRPDE required columns exist.

    Raises
    ------
    DataValidationError
        If one or more required columns are missing.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise DataValidationError(
            [
                "Missing required columns: "
                + ", ".join(missing_columns)
            ]
        )


def validate_no_duplicates(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate uniqueness of timestamp + asset_id observations.
    """

    validate_required_columns(dataframe)

    duplicate_mask = dataframe.duplicated(
        subset=["timestamp", "asset_id"],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        raise DataValidationError(
            [
                "Duplicate timestamp/asset observations detected: "
                f"{duplicate_count} rows."
            ]
        )


def validate_ohlc_relationships(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate basic OHLC price relationships.

    Required relationships:

        high >= low
        high >= open >= low
        high >= close >= low
    """

    validate_required_columns(dataframe)

    numeric = dataframe.copy()

    for column in PRICE_COLUMNS:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

    valid_rows = numeric[list(PRICE_COLUMNS)].notna().all(axis=1)

    high = numeric["high"]
    low = numeric["low"]
    open_price = numeric["open"]
    close = numeric["close"]

    invalid = valid_rows & (
        (high < low)
        | (open_price > high)
        | (open_price < low)
        | (close > high)
        | (close < low)
    )

    invalid_count = int(
        invalid.sum()
    )

    if invalid_count > 0:
        raise DataValidationError(
            [
                "Invalid OHLC relationships detected in "
                f"{invalid_count} rows."
            ]
        )


def validate_positive_prices(
    dataframe: pd.DataFrame,
) -> None:
    """Validate that OHLC prices are strictly positive."""

    validate_required_columns(dataframe)

    errors: list[str] = []

    for column in PRICE_COLUMNS:
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        invalid_count = int(
            (values <= 0).sum()
        )

        if invalid_count > 0:
            errors.append(
                f"{column} contains {invalid_count} "
                "non-positive values."
            )

    if errors:
        raise DataValidationError(errors)


def validate_non_negative_volume(
    dataframe: pd.DataFrame,
) -> None:
    """Validate that volume is not negative."""

    validate_required_columns(dataframe)

    volume = pd.to_numeric(
        dataframe["volume"],
        errors="coerce",
    )

    invalid_count = int(
        (volume < 0).sum()
    )

    if invalid_count > 0:
        raise DataValidationError(
            [
                "volume contains "
                f"{invalid_count} negative values."
            ]
        )


def validate_chronological_order(
    dataframe: pd.DataFrame,
) -> None:
    """Validate timestamp ordering independently for each asset."""

    validate_required_columns(dataframe)

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise DataValidationError(
            [
                "Cannot validate chronological ordering "
                "because timestamp contains invalid values."
            ]
        )

    check_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": dataframe["asset_id"],
        }
    )

    errors: list[str] = []

    for asset_id, asset_frame in check_frame.groupby(
        "asset_id",
        dropna=False,
    ):
        if not asset_frame["timestamp"].is_monotonic_increasing:
            errors.append(
                "Timestamps are not chronologically ordered "
                f"for asset '{asset_id}'."
            )

    if errors:
        raise DataValidationError(errors)