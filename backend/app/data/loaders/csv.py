from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.data.canonical import to_canonical


class CSVLoaderError(ValueError):
    """Raised when a CSV dataset cannot be loaded."""


DEFAULT_ENCODING = "utf-8"

REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def load_csv(
    input_path: str | Path,
    *,
    validate: bool = True,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """
    Load an OHLCV CSV file into the AVF-TRPDE canonical schema.

    Parameters
    ----------
    input_path:
        Path to the CSV file.

    validate:
        Whether to validate the dataset during canonicalization.

    encoding:
        File encoding. Defaults to UTF-8.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV DataFrame.
    """

    path = Path(input_path)

    _validate_input_path(path)

    try:
        dataframe = pd.read_csv(
            path,
            encoding=encoding,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CSVLoaderError(
            f"Unable to read CSV file '{path}': {exc}"
        ) from exc

    if dataframe.empty:
        raise CSVLoaderError(
            f"CSV file '{path}' contains no rows."
        )

    _validate_columns(dataframe)

    try:
        canonical = to_canonical(
            dataframe,
            validate=validate,
        )
    except Exception as exc:
        raise CSVLoaderError(
            f"CSV file '{path}' failed canonicalization: {exc}"
        ) from exc

    return canonical


def load_csv_raw(
    input_path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """
    Load a CSV without canonicalization.

    This is useful when inspecting a source file before applying
    the canonical data contract.
    """

    path = Path(input_path)

    _validate_input_path(path)

    try:
        dataframe = pd.read_csv(
            path,
            encoding=encoding,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CSVLoaderError(
            f"Unable to read CSV file '{path}': {exc}"
        ) from exc

    return dataframe


def load_csv_chunks(
    input_path: str | Path,
    *,
    chunksize: int = 100_000,
    encoding: str = DEFAULT_ENCODING,
):
    """
    Read a large CSV incrementally.

    The returned iterator yields pandas DataFrames.

    Canonicalization is intentionally not performed here because
    chunk-level processing must be handled explicitly by the caller.
    """

    path = Path(input_path)

    _validate_input_path(path)

    if chunksize <= 0:
        raise ValueError(
            "chunksize must be greater than zero."
        )

    try:
        return pd.read_csv(
            path,
            encoding=encoding,
            chunksize=chunksize,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CSVLoaderError(
            f"Unable to read CSV file '{path}': {exc}"
        ) from exc


def save_csv(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """
    Canonicalize and save an OHLCV DataFrame as CSV.
    """

    canonical = to_canonical(
        dataframe,
        validate=validate,
    )

    path = Path(output_path)

    if path.suffix.lower() != ".csv":
        raise CSVLoaderError(
            "CSV output path must use the '.csv' extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical.to_csv(
        path,
        index=False,
    )

    return path


def _validate_input_path(
    path: Path,
) -> None:
    """
    Validate that the requested path is an existing CSV file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    if not path.is_file():
        raise CSVLoaderError(
            f"CSV path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise CSVLoaderError(
            f"Expected a '.csv' file, received: {path.name}"
        )


def _validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verify that the source CSV contains the required AVF-TRPDE
    OHLCV columns.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise CSVLoaderError(
            "CSV file is missing required columns: "
            + ", ".join(missing)
        )