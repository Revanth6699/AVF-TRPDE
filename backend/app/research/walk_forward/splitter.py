from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class WalkForwardSplitError(ValueError):
    """Raised when a walk-forward split cannot be constructed."""


@dataclass(frozen=True)
class WalkForwardFold:
    """Represents one chronological walk-forward fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """
    Generate chronological walk-forward train/test folds.

    The splitter never shuffles observations and never allows the test
    period to precede or overlap the training period.

    Parameters
    ----------
    train_size:
        Number of historical observations used for each training window.

    test_size:
        Number of observations used for each out-of-sample test window.

    step_size:
        Number of observations by which the window advances after each fold.

    expanding:
        If True, the training window expands from the original training
        start. If False, a fixed-length rolling training window is used.
    """

    def __init__(
        self,
        *,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        expanding: bool = False,
    ) -> None:
        if not isinstance(train_size, int) or isinstance(train_size, bool):
            raise WalkForwardSplitError(
                "train_size must be an integer."
            )

        if not isinstance(test_size, int) or isinstance(test_size, bool):
            raise WalkForwardSplitError(
                "test_size must be an integer."
            )

        if step_size is None:
            step_size = test_size

        if not isinstance(step_size, int) or isinstance(step_size, bool):
            raise WalkForwardSplitError(
                "step_size must be an integer."
            )

        if train_size < 1:
            raise WalkForwardSplitError(
                "train_size must be greater than zero."
            )

        if test_size < 1:
            raise WalkForwardSplitError(
                "test_size must be greater than zero."
            )

        if step_size < 1:
            raise WalkForwardSplitError(
                "step_size must be greater than zero."
            )

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding

    def split(
        self,
        dataframe: pd.DataFrame,
        *,
        timestamp_column: str = "timestamp",
    ) -> list[WalkForwardFold]:
        """
        Generate chronological walk-forward folds.

        The dataframe must contain a valid timestamp column. Rows are
        ordered chronologically before fold construction.

        Returns
        -------
        list[WalkForwardFold]
            Chronological train/test fold definitions.
        """

        self._validate_input(
            dataframe,
            timestamp_column=timestamp_column,
        )

        timestamps = pd.to_datetime(
            dataframe[timestamp_column],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise WalkForwardSplitError(
                f"{timestamp_column} contains invalid timestamps."
            )

        ordered_timestamps = (
            timestamps
            .sort_values()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        total_observations = len(ordered_timestamps)

        minimum_required = self.train_size + self.test_size

        if total_observations < minimum_required:
            raise WalkForwardSplitError(
                "Insufficient observations for walk-forward splitting. "
                f"Required at least {minimum_required}, "
                f"received {total_observations}."
            )

        folds: list[WalkForwardFold] = []

        train_start_index = 0
        fold_id = 1

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = train_start_index

            train_end = train_start_index + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > total_observations:
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=ordered_timestamps.iloc[train_start],
                train_end=ordered_timestamps.iloc[train_end - 1],
                test_start=ordered_timestamps.iloc[test_start],
                test_end=ordered_timestamps.iloc[test_end - 1],
            )

            folds.append(fold)

            train_start_index += self.step_size
            fold_id += 1

        if not folds:
            raise WalkForwardSplitError(
                "No valid walk-forward folds could be constructed."
            )

        return folds

    def split_indices(
        self,
        dataframe: pd.DataFrame,
        *,
        timestamp_column: str = "timestamp",
    ) -> list[tuple[list[int], list[int]]]:
        """
        Return integer train/test indices for each fold.

        Indices refer to the dataframe after chronological ordering.
        """

        self._validate_input(
            dataframe,
            timestamp_column=timestamp_column,
        )

        timestamps = pd.to_datetime(
            dataframe[timestamp_column],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise WalkForwardSplitError(
                f"{timestamp_column} contains invalid timestamps."
            )

        ordered = (
            dataframe.assign(
                __walk_forward_timestamp=timestamps,
                __walk_forward_original_index=dataframe.index,
            )
            .sort_values(
                "__walk_forward_timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        total_observations = len(ordered)

        minimum_required = self.train_size + self.test_size

        if total_observations < minimum_required:
            raise WalkForwardSplitError(
                "Insufficient observations for walk-forward splitting. "
                f"Required at least {minimum_required}, "
                f"received {total_observations}."
            )

        splits: list[tuple[list[int], list[int]]] = []

        train_start_index = 0

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = train_start_index

            train_end = train_start_index + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > total_observations:
                break

            train_indices = ordered.loc[
                train_start:train_end - 1,
                "__walk_forward_original_index",
            ].tolist()

            test_indices = ordered.loc[
                test_start:test_end - 1,
                "__walk_forward_original_index",
            ].tolist()

            splits.append(
                (
                    train_indices,
                    test_indices,
                )
            )

            train_start_index += self.step_size

        if not splits:
            raise WalkForwardSplitError(
                "No valid walk-forward splits could be constructed."
            )

        return splits

    @staticmethod
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
            raise WalkForwardSplitError(
                "dataframe must not be empty."
            )

        if timestamp_column not in dataframe.columns:
            raise WalkForwardSplitError(
                f"Missing timestamp column: {timestamp_column}"
            )