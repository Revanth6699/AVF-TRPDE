from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class WalkForwardFoldError(ValueError):
    """Raised when a walk-forward fold is invalid."""


@dataclass(frozen=True)
class WalkForwardFoldData:
    """Data and metadata for one walk-forward fold."""

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)


def create_fold(
    dataframe: pd.DataFrame,
    *,
    fold_id: int,
    train_indices: list[int],
    test_indices: list[int],
    timestamp_column: str = "timestamp",
) -> WalkForwardFoldData:
    """
    Construct and validate one walk-forward fold.

    The supplied indices identify the training and testing observations.
    No model fitting or feature transformation occurs here.
    """

    _validate_input(
        dataframe,
        timestamp_column=timestamp_column,
    )

    if not isinstance(fold_id, int) or isinstance(fold_id, bool):
        raise WalkForwardFoldError(
            "fold_id must be an integer."
        )

    if fold_id < 1:
        raise WalkForwardFoldError(
            "fold_id must be greater than zero."
        )

    if not train_indices:
        raise WalkForwardFoldError(
            "train_indices must not be empty."
        )

    if not test_indices:
        raise WalkForwardFoldError(
            "test_indices must not be empty."
        )

    train = dataframe.loc[train_indices].copy()
    test = dataframe.loc[test_indices].copy()

    train = _sort_by_timestamp(
        train,
        timestamp_column=timestamp_column,
    )

    test = _sort_by_timestamp(
        test,
        timestamp_column=timestamp_column,
    )

    train_timestamps = pd.to_datetime(
        train[timestamp_column],
        errors="coerce",
    )

    test_timestamps = pd.to_datetime(
        test[timestamp_column],
        errors="coerce",
    )

    train_start = train_timestamps.iloc[0]
    train_end = train_timestamps.iloc[-1]
    test_start = test_timestamps.iloc[0]
    test_end = test_timestamps.iloc[-1]

    if train_end >= test_start:
        raise WalkForwardFoldError(
            "Training data must end strictly before test data."
        )

    train_index_set = set(train.index)
    test_index_set = set(test.index)

    if train_index_set.intersection(test_index_set):
        raise WalkForwardFoldError(
            "Training and test observations must not overlap."
        )

    return WalkForwardFoldData(
        fold_id=fold_id,
        train=train,
        test=test,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def create_fold_from_positions(
    dataframe: pd.DataFrame,
    *,
    fold_id: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    timestamp_column: str = "timestamp",
) -> WalkForwardFoldData:
    """
    Construct one fold using positional boundaries.

    End positions are exclusive, following standard Python slicing rules.
    """

    _validate_input(
        dataframe,
        timestamp_column=timestamp_column,
    )

    if train_start < 0:
        raise WalkForwardFoldError(
            "train_start must be non-negative."
        )

    if train_end <= train_start:
        raise WalkForwardFoldError(
            "train_end must be greater than train_start."
        )

    if test_start < train_end:
        raise WalkForwardFoldError(
            "test_start must be greater than or equal to train_end."
        )

    if test_end <= test_start:
        raise WalkForwardFoldError(
            "test_end must be greater than test_start."
        )

    if test_end > len(dataframe):
        raise WalkForwardFoldError(
            "test_end exceeds dataframe length."
        )

    ordered = _sort_by_timestamp(
        dataframe,
        timestamp_column=timestamp_column,
    )

    train = ordered.iloc[
        train_start:train_end
    ].copy()

    test = ordered.iloc[
        test_start:test_end
    ].copy()

    if train.empty:
        raise WalkForwardFoldError(
            "Training fold is empty."
        )

    if test.empty:
        raise WalkForwardFoldError(
            "Test fold is empty."
        )

    train_timestamps = pd.to_datetime(
        train[timestamp_column],
        errors="coerce",
    )

    test_timestamps = pd.to_datetime(
        test[timestamp_column],
        errors="coerce",
    )

    train_start_timestamp = train_timestamps.iloc[0]
    train_end_timestamp = train_timestamps.iloc[-1]
    test_start_timestamp = test_timestamps.iloc[0]
    test_end_timestamp = test_timestamps.iloc[-1]

    if train_end_timestamp >= test_start_timestamp:
        raise WalkForwardFoldError(
            "Training data must end strictly before test data."
        )

    return WalkForwardFoldData(
        fold_id=fold_id,
        train=train,
        test=test,
        train_start=train_start_timestamp,
        train_end=train_end_timestamp,
        test_start=test_start_timestamp,
        test_end=test_end_timestamp,
    )


def _sort_by_timestamp(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    result = dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    if result[timestamp_column].isna().any():
        raise WalkForwardFoldError(
            f"{timestamp_column} contains invalid timestamps."
        )

    return (
        result
        .sort_values(
            by=timestamp_column,
            kind="stable",
        )
        .copy()
    )


def _validate_input(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise WalkForwardFoldError(
            "dataframe must not be empty."
        )

    if timestamp_column not in dataframe.columns:
        raise WalkForwardFoldError(
            f"Missing timestamp column: {timestamp_column}"
        )

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise WalkForwardFoldError(
            f"{timestamp_column} contains invalid timestamps."
        )