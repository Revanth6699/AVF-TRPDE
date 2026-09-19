from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.data.canonical import (
    assert_canonical_schema,
    to_canonical,
)


class ParquetLoaderError(ValueError):
    """Raised when a Parquet dataset cannot be loaded."""


DEFAULT_ENGINE = "auto"

REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def load_parquet(
    input_path: str | Path,
    *,
    validate: bool = True,
    engine: str = DEFAULT_ENGINE,
) -> pd.DataFrame:
    """
    Load a Parquet OHLCV dataset into the AVF-TRPDE
    canonical schema.
    """

    path = Path(input_path)

    _validate_input_path(path)

    try:
        dataframe = pd.read_parquet(
            path,
            engine=engine,
        )
    except (
        OSError,
        ImportError,
        ValueError,
    ) as exc:
        raise ParquetLoaderError(
            f"Unable to read Parquet file '{path}': {exc}"
        ) from exc

    if dataframe.empty:
        raise ParquetLoaderError(
            f"Parquet file '{path}' contains no rows."
        )

    _validate_columns(dataframe)

    try:
        canonical = to_canonical(
            dataframe,
            validate=validate,
        )
    except Exception as exc:
        raise ParquetLoaderError(
            f"Parquet file '{path}' failed canonicalization: {exc}"
        ) from exc

    return canonical


def load_parquet_raw(
    input_path: str | Path,
    *,
    engine: str = DEFAULT_ENGINE,
) -> pd.DataFrame:
    """
    Load a Parquet file without canonicalization.

    This is intended for source inspection and debugging.
    """

    path = Path(input_path)

    _validate_input_path(path)

    try:
        dataframe = pd.read_parquet(
            path,
            engine=engine,
        )
    except (
        OSError,
        ImportError,
        ValueError,
    ) as exc:
        raise ParquetLoaderError(
            f"Unable to read Parquet file '{path}': {exc}"
        ) from exc

    return dataframe


def save_parquet(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    validate: bool = True,
    engine: str = DEFAULT_ENGINE,
    compression: str = "snappy",
) -> Path:
    """
    Canonicalize and save an OHLCV DataFrame as Parquet.
    """

    canonical = to_canonical(
        dataframe,
        validate=validate,
    )

    assert_canonical_schema(
        canonical
    )

    path = Path(output_path)

    if path.suffix.lower() != ".parquet":
        raise ParquetLoaderError(
            "Parquet output path must use the '.parquet' extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        canonical.to_parquet(
            path,
            engine=engine,
            compression=compression,
            index=False,
        )
    except (
        OSError,
        ImportError,
        ValueError,
    ) as exc:
        raise ParquetLoaderError(
            f"Unable to write Parquet file '{path}': {exc}"
        ) from exc

    return path


def save_canonical_parquet(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    engine: str = DEFAULT_ENGINE,
    compression: str = "snappy",
) -> Path:
    """
    Save a DataFrame that is already expected to be canonical.

    The canonical schema is checked before writing.
    """

    assert_canonical_schema(
        dataframe
    )

    path = Path(output_path)

    if path.suffix.lower() != ".parquet":
        raise ParquetLoaderError(
            "Parquet output path must use the '.parquet' extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        dataframe.to_parquet(
            path,
            engine=engine,
            compression=compression,
            index=False,
        )
    except (
        OSError,
        ImportError,
        ValueError,
    ) as exc:
        raise ParquetLoaderError(
            f"Unable to write Parquet file '{path}': {exc}"
        ) from exc

    return path


def load_and_verify_parquet(
    input_path: str | Path,
    *,
    validate: bool = True,
    engine: str = DEFAULT_ENGINE,
) -> pd.DataFrame:
    """
    Load a Parquet dataset and verify that it satisfies the
    canonical data contract.
    """

    dataframe = load_parquet(
        input_path,
        validate=validate,
        engine=engine,
    )

    assert_canonical_schema(
        dataframe
    )

    return dataframe


def _validate_input_path(
    path: Path,
) -> None:
    """
    Validate that the requested path is an existing Parquet file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}"
        )

    if not path.is_file():
        raise ParquetLoaderError(
            f"Parquet path is not a file: {path}"
        )

    if path.suffix.lower() != ".parquet":
        raise ParquetLoaderError(
            f"Expected a '.parquet' file, received: {path.name}"
        )


def _validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verify that the Parquet dataset contains the required
    AVF-TRPDE OHLCV columns.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ParquetLoaderError(
            "Parquet file is missing required columns: "
            + ", ".join(missing)
        )