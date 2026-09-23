from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.app.data.validation import (
    REQUIRED_COLUMNS,
    validate_ohlcv,
)


CANONICAL_COLUMNS = [
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class CanonicalDataError(ValueError):
    """Raised when data cannot be converted to canonical format."""


@dataclass(frozen=True)
class CanonicalDataset:
    """Metadata describing a canonical dataset."""

    dataframe: pd.DataFrame
    asset_ids: tuple[str, ...]
    row_count: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


def to_canonical(
    dataframe: pd.DataFrame,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Convert an OHLCV DataFrame into the AVF-TRPDE canonical schema.

    Canonical schema:

        timestamp
        asset_id
        open
        high
        low
        close
        volume

    Processing order:

    1. check source schema
    2. copy the input
    3. normalize timestamps
    4. normalize asset identifiers
    5. convert numeric fields
    6. detect duplicate observations
    7. sort deterministically
    8. validate the canonical result
    9. return a new DataFrame

    The input DataFrame is never modified in place.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise CanonicalDataError(
            "Cannot canonicalize an empty DataFrame."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise CanonicalDataError(
            "Cannot canonicalize dataset. Missing columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------
    # Select canonical columns and copy
    # ---------------------------------------------------------

    canonical = dataframe.loc[
        :,
        CANONICAL_COLUMNS,
    ].copy()

    # ---------------------------------------------------------
    # Timestamp normalization
    # ---------------------------------------------------------

    canonical["timestamp"] = pd.to_datetime(
        canonical["timestamp"],
        errors="coerce",
    )

    if canonical["timestamp"].isna().any():
        invalid_count = int(
            canonical["timestamp"].isna().sum()
        )

        raise CanonicalDataError(
            "timestamp contains "
            f"{invalid_count} invalid or missing value(s)."
        )

    # ---------------------------------------------------------
    # Asset identifier normalization
    # ---------------------------------------------------------

    canonical["asset_id"] = (
        canonical["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if canonical["asset_id"].isna().any():
        raise CanonicalDataError(
            "asset_id contains missing values."
        )

    if (canonical["asset_id"] == "").any():
        raise CanonicalDataError(
            "asset_id contains empty values."
        )

    # ---------------------------------------------------------
    # Numeric normalization
    # ---------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        canonical[column] = pd.to_numeric(
            canonical[column],
            errors="coerce",
        )

    numeric_missing = canonical[
        numeric_columns
    ].isna().any(axis=1)

    if numeric_missing.any():
        count = int(numeric_missing.sum())

        raise CanonicalDataError(
            f"{count} rows contain invalid numeric "
            "OHLCV values."
        )

    # ---------------------------------------------------------
    # Duplicate observation detection
    # ---------------------------------------------------------

    duplicate_mask = canonical.duplicated(
        subset=[
            "timestamp",
            "asset_id",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        raise CanonicalDataError(
            "Canonical dataset contains "
            f"{duplicate_count} duplicate "
            "timestamp/asset observations."
        )

    # ---------------------------------------------------------
    # Deterministic sorting
    # ---------------------------------------------------------

    canonical = canonical.sort_values(
        by=[
            "asset_id",
            "timestamp",
        ],
        ascending=[
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Stable column order
    # ---------------------------------------------------------

    canonical = canonical[
        CANONICAL_COLUMNS
    ].copy()

    # ---------------------------------------------------------
    # Final canonical validation
    #
    # Validation happens AFTER normalization and sorting.
    # This is critical because validation checks chronological
    # ordering and other canonical data-contract properties.
    # ---------------------------------------------------------

    if validate:
        validate_ohlcv(
            canonical,
            raise_on_error=True,
        )

    return canonical


def build_canonical_dataset(
    dataframe: pd.DataFrame,
    *,
    validate: bool = True,
) -> CanonicalDataset:
    """
    Convert a DataFrame into a canonical dataset object.

    Returns the canonical DataFrame together with basic
    dataset metadata.
    """

    canonical = to_canonical(
        dataframe,
        validate=validate,
    )

    if canonical.empty:
        raise CanonicalDataError(
            "Cannot build canonical dataset from an empty DataFrame."
        )

    assets = tuple(
        sorted(
            canonical["asset_id"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    return CanonicalDataset(
        dataframe=canonical,
        asset_ids=assets,
        row_count=len(canonical),
        start_timestamp=canonical["timestamp"].min(),
        end_timestamp=canonical["timestamp"].max(),
    )


def save_canonical_parquet(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """
    Canonicalize a DataFrame and save it as Parquet.

    The parent directory is created automatically.
    """

    canonical = to_canonical(
        dataframe,
        validate=validate,
    )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical.to_parquet(
        path,
        index=False,
    )

    return path


def load_canonical_parquet(
    input_path: str | Path,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a Parquet dataset and verify its canonical structure.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {path}"
        )

    if path.suffix.lower() != ".parquet":
        raise CanonicalDataError(
            "Canonical datasets must be stored as Parquet files."
        )

    dataframe = pd.read_parquet(path)

    return to_canonical(
        dataframe,
        validate=validate,
    )


def get_dataset_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    Return deterministic metadata for a canonical dataset.

    This metadata will later be consumed by fingerprint.py.
    """

    canonical = to_canonical(
        dataframe,
        validate=True,
    )

    assets = sorted(
        canonical["asset_id"]
        .unique()
        .tolist()
    )

    return {
        "row_count": int(len(canonical)),
        "asset_count": int(len(assets)),
        "assets": assets,
        "start_timestamp": (
            canonical["timestamp"]
            .min()
            .isoformat()
        ),
        "end_timestamp": (
            canonical["timestamp"]
            .max()
            .isoformat()
        ),
        "columns": CANONICAL_COLUMNS.copy(),
    }


def assert_canonical_schema(
    dataframe: pd.DataFrame,
) -> None:
    """
    Assert that a DataFrame exactly follows the canonical schema.

    Column order is checked deliberately because the canonical
    dataset is an explicit data contract.
    """

    actual_columns = list(dataframe.columns)

    if actual_columns != CANONICAL_COLUMNS:
        raise CanonicalDataError(
            "Invalid canonical schema.\n"
            f"Expected: {CANONICAL_COLUMNS}\n"
            f"Received: {actual_columns}"
        )

    if dataframe.empty:
        raise CanonicalDataError(
            "Canonical dataset is empty."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["timestamp"]
    ):
        raise CanonicalDataError(
            "Canonical timestamp column must be datetime-like."
        )

    if not (
        pd.api.types.is_string_dtype(
            dataframe["asset_id"]
        )
        or pd.api.types.is_object_dtype(
            dataframe["asset_id"]
        )
    ):
        raise CanonicalDataError(
            "Canonical asset_id column must contain strings."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            raise CanonicalDataError(
                f"Canonical column '{column}' "
                "must be numeric."
            )


def canonicalize_and_save(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> CanonicalDataset:
    """
    Canonicalize, validate, and persist a dataset.

    This is the main convenience function for the data pipeline.
    """

    dataset = build_canonical_dataset(
        dataframe,
        validate=True,
    )

    save_canonical_parquet(
        dataset.dataframe,
        output_path,
        validate=False,
    )

    return dataset