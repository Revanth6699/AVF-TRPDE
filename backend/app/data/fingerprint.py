from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from backend.app.data.canonical import (
    CANONICAL_COLUMNS,
    to_canonical,
)


FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_VERSION = "1"


class FingerprintError(ValueError):
    """Raised when a dataset fingerprint cannot be generated."""


@dataclass(frozen=True)
class DatasetFingerprint:
    """
    Deterministic identity for a canonical AVF-TRPDE dataset.
    """

    fingerprint: str
    algorithm: str
    version: str
    row_count: int
    asset_count: int
    assets: tuple[str, ...]
    start_timestamp: str
    end_timestamp: str
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """
        Convert the fingerprint metadata into a JSON-compatible dict.
        """

        return asdict(self)

    def to_json(self) -> str:
        """
        Serialize fingerprint metadata deterministically.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def fingerprint_dataframe(
    dataframe: pd.DataFrame,
    *,
    validate: bool = True,
) -> DatasetFingerprint:
    """
    Generate a deterministic SHA-256 fingerprint for a canonical dataset.

    The dataset is canonicalized before hashing so that equivalent
    source representations produce the same fingerprint.

    The fingerprint includes:

    - canonical schema
    - row ordering
    - timestamps
    - asset identifiers
    - OHLCV values
    - dataset metadata

    The original DataFrame is never modified.
    """

    canonical = to_canonical(
        dataframe,
        validate=validate,
    )

    if canonical.empty:
        raise FingerprintError(
            "Cannot fingerprint an empty dataset."
        )

    payload = _serialize_dataframe(canonical)

    digest = hashlib.sha256(
        payload
    ).hexdigest()

    assets = tuple(
        sorted(
            canonical["asset_id"]
            .unique()
            .tolist()
        )
    )

    return DatasetFingerprint(
        fingerprint=digest,
        algorithm=FINGERPRINT_ALGORITHM,
        version=FINGERPRINT_VERSION,
        row_count=int(len(canonical)),
        asset_count=int(len(assets)),
        assets=assets,
        start_timestamp=_timestamp_to_string(
            canonical["timestamp"].min()
        ),
        end_timestamp=_timestamp_to_string(
            canonical["timestamp"].max()
        ),
        columns=tuple(CANONICAL_COLUMNS),
    )


def fingerprint_parquet(
    input_path: str | Path,
    *,
    validate: bool = True,
) -> DatasetFingerprint:
    """
    Load a canonical Parquet dataset and generate its fingerprint.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    if path.suffix.lower() != ".parquet":
        raise FingerprintError(
            "Fingerprint input must be a Parquet file."
        )

    dataframe = pd.read_parquet(path)

    return fingerprint_dataframe(
        dataframe,
        validate=validate,
    )


def fingerprint_file(
    input_path: str | Path,
) -> str:
    """
    Generate a SHA-256 hash of the raw file bytes.

    This is different from fingerprint_dataframe():

    fingerprint_dataframe()
        -> identifies the canonical dataset content

    fingerprint_file()
        -> identifies the exact physical file bytes
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_fingerprint(
    dataframe: pd.DataFrame,
    expected_fingerprint: str,
    *,
    validate: bool = True,
) -> bool:
    """
    Verify that a DataFrame matches an expected dataset fingerprint.
    """

    if not expected_fingerprint:
        raise FingerprintError(
            "expected_fingerprint must not be empty."
        )

    actual = fingerprint_dataframe(
        dataframe,
        validate=validate,
    )

    return actual.fingerprint == expected_fingerprint.strip().lower()


def _serialize_dataframe(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Serialize the canonical dataset deterministically.

    Fixed serialization parameters prevent differences in:
    - index handling
    - column ordering
    - float formatting
    - timestamp formatting
    - line endings
    """

    if list(dataframe.columns) != CANONICAL_COLUMNS:
        raise FingerprintError(
            "DataFrame does not follow the canonical column order."
        )

    serialized = dataframe.to_csv(
        index=False,
        columns=CANONICAL_COLUMNS,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
        float_format="%.17g",
        lineterminator="\n",
    )

    return serialized.encode("utf-8")


def _timestamp_to_string(
    timestamp: pd.Timestamp,
) -> str:
    """
    Convert a timestamp into a deterministic representation.
    """

    if pd.isna(timestamp):
        raise FingerprintError(
            "Dataset contains an invalid timestamp."
        )

    return timestamp.isoformat()