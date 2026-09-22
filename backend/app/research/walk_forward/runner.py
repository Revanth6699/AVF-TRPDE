from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backend.app.research.walk_forward.fold import WalkForwardFoldData
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


class WalkForwardRunnerError(ValueError):
    """Raised when a walk-forward run cannot be completed."""


@dataclass(frozen=True)
class WalkForwardRunResult:
    """Results produced by a walk-forward experiment."""

    predictions: pd.DataFrame
    folds: tuple[WalkForwardFoldData, ...]
    fold_count: int
    prediction_count: int


PredictionFunction = Callable[
    [pd.DataFrame, pd.DataFrame, WalkForwardFoldData],
    pd.DataFrame,
]


class WalkForwardRunner:
    """
    Execute a model through chronological walk-forward folds.

    The runner is responsible only for:
    - generating folds
    - selecting train/test observations from fold boundaries
    - calling the supplied prediction function
    - validating and combining predictions

    Model fitting remains inside the supplied prediction function.
    """

    def __init__(
        self,
        splitter: WalkForwardSplitter,
    ) -> None:
        if not isinstance(splitter, WalkForwardSplitter):
            raise TypeError(
                "splitter must be a WalkForwardSplitter."
            )

        self.splitter = splitter

    def run(
        self,
        dataframe: pd.DataFrame,
        predict: PredictionFunction,
    ) -> WalkForwardRunResult:
        """
        Execute the supplied prediction function across all folds.

        Parameters
        ----------
        dataframe:
            Chronological research dataset.

        predict:
            Callable receiving:

                train_dataframe
                test_dataframe
                fold

            and returning a DataFrame containing at minimum:
                timestamp
                asset_id
                prediction
        """

        self._validate_input(dataframe)

        if not callable(predict):
            raise TypeError(
                "predict must be callable."
            )

        folds = tuple(
            self.splitter.split(dataframe)
        )

        if not folds:
            raise WalkForwardRunnerError(
                "Walk-forward splitter produced no folds."
            )

        prediction_frames: list[pd.DataFrame] = []

        for fold in folds:
            train_data, test_data = self._select_fold_data(
                dataframe,
                fold,
            )

            predictions = predict(
                train_data,
                test_data,
                fold,
            )

            predictions = self._validate_predictions(
                predictions=predictions,
                test_data=test_data,
                fold=fold,
            )

            prediction_frames.append(predictions)

        combined_predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        combined_predictions = combined_predictions.sort_values(
            by=["timestamp", "asset_id", "fold_id"],
            ascending=True,
        ).reset_index(drop=True)

        return WalkForwardRunResult(
            predictions=combined_predictions,
            folds=folds,
            fold_count=len(folds),
            prediction_count=len(combined_predictions),
        )

    @staticmethod
    def _select_fold_data(
        dataframe: pd.DataFrame,
        fold: WalkForwardFoldData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise WalkForwardRunnerError(
                "timestamp contains invalid values."
            )

        train_mask = (
            (timestamps >= fold.train_start)
            & (timestamps <= fold.train_end)
        )

        test_mask = (
            (timestamps >= fold.test_start)
            & (timestamps <= fold.test_end)
        )

        train_data = dataframe.loc[
            train_mask
        ].copy()

        test_data = dataframe.loc[
            test_mask
        ].copy()

        if train_data.empty:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} produced an empty "
                "training dataset."
            )

        if test_data.empty:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} produced an empty "
                "test dataset."
            )

        train_data = train_data.sort_values(
            by=["timestamp", "asset_id"],
            ascending=True,
        ).reset_index(drop=True)

        test_data = test_data.sort_values(
            by=["timestamp", "asset_id"],
            ascending=True,
        ).reset_index(drop=True)

        return train_data, test_data

    @staticmethod
    def _validate_predictions(
        predictions: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
    ) -> pd.DataFrame:
        if not isinstance(
            predictions,
            pd.DataFrame,
        ):
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} prediction function "
                "must return a pandas DataFrame."
            )

        required_columns = {
            "timestamp",
            "asset_id",
            "prediction",
        }

        missing_columns = (
            required_columns
            - set(predictions.columns)
        )

        if missing_columns:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions are missing "
                "required columns: "
                + ", ".join(sorted(missing_columns))
            )

        result = predictions.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        )

        if result["timestamp"].isna().any():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "invalid timestamps."
            )

        result["asset_id"] = (
            result["asset_id"]
            .astype("string")
        )

        result["prediction"] = pd.to_numeric(
            result["prediction"],
            errors="coerce",
        )

        if result["prediction"].isna().any():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "invalid prediction values."
            )

        if not result["prediction"].map(
            pd.notna
        ).all():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} contains missing "
                "prediction values."
            )

        if len(result) != len(test_data):
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} returned "
                f"{len(result)} predictions for "
                f"{len(test_data)} test observations."
            )

        result["fold_id"] = fold.fold_id

        test_keys = set(
            zip(
                test_data["timestamp"],
                test_data["asset_id"],
            )
        )

        prediction_keys = set(
            zip(
                result["timestamp"],
                result["asset_id"],
            )
        )

        if prediction_keys != test_keys:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions do not "
                "match the test observations."
            )

        result = result.sort_values(
            by=["timestamp", "asset_id"],
            ascending=True,
        ).reset_index(drop=True)

        return result

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame,
    ) -> None:
        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas.DataFrame."
            )

        if dataframe.empty:
            raise WalkForwardRunnerError(
                "Input dataframe must not be empty."
            )

        required_columns = {
            "timestamp",
            "asset_id",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise WalkForwardRunnerError(
                "Input dataframe is missing required "
                "columns: "
                + ", ".join(sorted(missing_columns))
            )

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise WalkForwardRunnerError(
                "Input dataframe contains invalid timestamps."
            )

        if dataframe["asset_id"].isna().any():
            raise WalkForwardRunnerError(
                "asset_id contains missing values."
            )