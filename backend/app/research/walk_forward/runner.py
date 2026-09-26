from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from backend.app.research.models.ewma import (
    EWMAVolatility,
)
from backend.app.research.models.gjr_garch import (
    GJRGARCH,
)
from backend.app.research.models.xgboost import (
    XGBoostConfig,
    XGBoostVolatility,
)
from backend.app.research.regime_models.regime_xgboost import (
    RegimeXGBoost,
)
from backend.app.research.regimes.hmm import (
    HMMConfig,
    HMMRegimeDetector,
)
from backend.app.research.walk_forward.fold import (
    WalkForwardFoldData,
)
from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)


class WalkForwardRunnerError(ValueError):
    """Raised when a walk-forward run cannot be completed."""


ModelName = Literal[
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "Regime-XGBoost",
]


@dataclass(frozen=True)
class WalkForwardRunResult:
    """Results produced by a walk-forward experiment."""

    predictions: pd.DataFrame
    folds: tuple[WalkForwardFoldData, ...]
    fold_count: int
    prediction_count: int


@dataclass(frozen=True)
class WalkForwardResearchResult:
    """Result of an actual AVF-TRPDE model execution."""

    model: str
    predictions: pd.DataFrame
    folds: tuple[WalkForwardFoldData, ...]
    fold_count: int
    prediction_count: int


@dataclass(frozen=True)
class AblationResult:
    """Result of one AVF-TRPDE ablation experiment."""

    ablation: str
    description: str
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
    Execute AVF-TRPDE chronological walk-forward research.

    The generic runner handles:
    - chronological fold generation
    - fold-local train/test selection
    - prediction validation
    - prediction aggregation

    The research methods additionally execute the locked models:
    - EWMA
    - GJR-GARCH
    - XGBoost
    - Regime-XGBoost

    HMM is treated as a regime detector and is executed inside
    Regime-XGBoost and the A2/A3 ablation pipelines.

    No random train/test split is performed.
    """

    def __init__(
        self,
        splitter: WalkForwardSplitter,
    ) -> None:
        if not isinstance(
            splitter,
            WalkForwardSplitter,
        ):
            raise TypeError(
                "splitter must be a WalkForwardSplitter."
            )

        self.splitter = splitter

    # ------------------------------------------------------------------
    # Generic walk-forward execution
    # ------------------------------------------------------------------

    def run(
        self,
        dataframe: pd.DataFrame,
        predict: PredictionFunction,
    ) -> WalkForwardRunResult:
        """
        Execute a supplied prediction function across all folds.

        The prediction function receives:

            train_dataframe
            test_dataframe
            fold

        and must return:

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

            prediction_frames.append(
                predictions
            )

        combined_predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        combined_predictions = (
            combined_predictions
            .sort_values(
                by=[
                    "timestamp",
                    "asset_id",
                    "fold_id",
                ],
                ascending=True,
            )
            .reset_index(drop=True)
        )

        return WalkForwardRunResult(
            predictions=combined_predictions,
            folds=folds,
            fold_count=len(folds),
            prediction_count=len(
                combined_predictions
            ),
        )

    # ------------------------------------------------------------------
    # Locked AVF-TRPDE model execution
    # ------------------------------------------------------------------

    def run_model(
        self,
        dataframe: pd.DataFrame,
        *,
        model: ModelName,
        target_column: str = "target",
        ewma_decay: float | None = None,
        hmm_config: HMMConfig | None = None,
        xgboost_config: XGBoostConfig | None = None,
        feature_columns: tuple[str, ...] | None = None,
    ) -> WalkForwardResearchResult:
        """
        Execute one locked AVF-TRPDE volatility model.

        Supported models:

            EWMA
            GJR-GARCH
            XGBoost
            Regime-XGBoost

        Regime-XGBoost performs fold-local:

            GJR-GARCH
                +
            HMM
                +
            XGBoost

        Parameters required by a model are deliberately explicit.
        No hidden model configuration is introduced.
        """

        self._validate_research_input(
            dataframe,
            target_column=target_column,
        )

        if model not in {
            "EWMA",
            "GJR-GARCH",
            "XGBoost",
            "Regime-XGBoost",
        }:
            raise WalkForwardRunnerError(
                f"Unsupported walk-forward model: {model}"
            )

        if model == "EWMA":
            if ewma_decay is None:
                raise WalkForwardRunnerError(
                    "ewma_decay is required for EWMA."
                )

        if model == "Regime-XGBoost":
            if hmm_config is None:
                raise WalkForwardRunnerError(
                    "hmm_config is required for Regime-XGBoost."
                )

        if model in {
            "GJR-GARCH",
            "Regime-XGBoost",
        }:
            self._require_one_step_test_window()

        def predict(
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            fold: WalkForwardFoldData,
        ) -> pd.DataFrame:
            if model == "EWMA":
                return self._predict_ewma(
                    train_data=train_data,
                    test_data=test_data,
                    fold=fold,
                    decay=float(ewma_decay),
                )

            if model == "GJR-GARCH":
                return self._predict_gjr_garch(
                    train_data=train_data,
                    test_data=test_data,
                    fold=fold,
                )

            if model == "XGBoost":
                return self._predict_xgboost(
                    train_data=train_data,
                    test_data=test_data,
                    fold=fold,
                    target_column=target_column,
                    xgboost_config=xgboost_config,
                    feature_columns=feature_columns,
                )

            return self._predict_regime_xgboost(
                train_data=train_data,
                test_data=test_data,
                fold=fold,
                target_column=target_column,
                hmm_config=hmm_config,
                xgboost_config=xgboost_config,
                feature_columns=feature_columns,
            )

        result = self.run(
            dataframe=dataframe,
            predict=predict,
        )

        predictions = result.predictions.copy()

        predictions["model"] = model

        return WalkForwardResearchResult(
            model=model,
            predictions=predictions,
            folds=result.folds,
            fold_count=result.fold_count,
            prediction_count=result.prediction_count,
        )

    # ------------------------------------------------------------------
    # Locked ablation framework
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        dataframe: pd.DataFrame,
        *,
        ablation: Literal[
            "A0",
            "A1",
            "A2",
            "A3",
        ],
        target_column: str = "target",
        hmm_config: HMMConfig | None = None,
        xgboost_config: XGBoostConfig | None = None,
        feature_columns: tuple[str, ...] | None = None,
    ) -> AblationResult:
        """
        Execute one locked ablation experiment.

        A0:
            XGBoost

        A1:
            XGBoost + GJR-GARCH

        A2:
            XGBoost + HMM

        A3:
            XGBoost + GJR-GARCH + HMM

        The ablation directly follows the locked research protocol.
        """

        self._validate_research_input(
            dataframe,
            target_column=target_column,
        )

        descriptions = {
            "A0": "XGBoost",
            "A1": "XGBoost + GJR-GARCH",
            "A2": "XGBoost + HMM",
            "A3": "XGBoost + GJR-GARCH + HMM",
        }

        if ablation not in descriptions:
            raise WalkForwardRunnerError(
                f"Unsupported ablation: {ablation}"
            )

        include_garch = ablation in {
            "A1",
            "A3",
        }

        include_hmm = ablation in {
            "A2",
            "A3",
        }

        if include_hmm and hmm_config is None:
            raise WalkForwardRunnerError(
                f"hmm_config is required for {ablation}."
            )

        if include_garch:
            self._require_one_step_test_window()

        def predict(
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            fold: WalkForwardFoldData,
        ) -> pd.DataFrame:
            train_features = train_data.copy()
            test_features = test_data.copy()

            if include_garch:
                train_features, test_features = (
                    self._attach_garch_features(
                        train_features,
                        test_features,
                        fold,
                    )
                )

            if include_hmm:
                (
                    train_features,
                    test_features,
                    regime_columns,
                ) = self._attach_hmm_features(
                    train_features,
                    test_features,
                    fold,
                    hmm_config=hmm_config,
                )
            else:
                regime_columns = ()

            xgboost = XGBoostVolatility(
                config=xgboost_config,
            )

            xgboost.fit(
                features=train_features,
                target=train_features[
                    target_column
                ],
                feature_columns=(
                    feature_columns
                    if feature_columns is not None
                    else None
                ),
            )

            predictions = xgboost.predict(
                test_features
            )

            result = test_features.loc[
                :,
                [
                    "timestamp",
                    "asset_id",
                ],
            ].copy()

            result["prediction"] = predictions
            result["actual_volatility"] = (
                test_features[target_column]
                .to_numpy(dtype=float)
            )

            if regime_columns:
                for column in regime_columns:
                    result[column] = (
                        test_features[column]
                        .to_numpy(dtype=float)
                    )

            if include_garch:
                result["garch_forecast"] = (
                    test_features[
                        "garch_forecast"
                    ].to_numpy(dtype=float)
                )

            return result

        result = self.run(
            dataframe=dataframe,
            predict=predict,
        )

        predictions = result.predictions.copy()

        predictions["ablation"] = ablation

        return AblationResult(
            ablation=ablation,
            description=descriptions[
                ablation
            ],
            predictions=predictions,
            folds=result.folds,
            fold_count=result.fold_count,
            prediction_count=result.prediction_count,
        )

    # ------------------------------------------------------------------
    # EWMA
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_ewma(
        *,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
        decay: float,
    ) -> pd.DataFrame:
        required = {
            "return",
            "target",
        }

        missing = required - set(
            train_data.columns
        )

        if missing:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} EWMA is missing "
                "required columns: "
                + ", ".join(sorted(missing))
            )

        rows: list[dict[str, object]] = []

        for asset_id in sorted(
            train_data["asset_id"]
            .astype(str)
            .unique()
        ):
            train_asset = train_data.loc[
                train_data["asset_id"].astype(str)
                == asset_id
            ].copy()

            test_asset = test_data.loc[
                test_data["asset_id"].astype(str)
                == asset_id
            ].copy()

            if test_asset.empty:
                continue

            model = EWMAVolatility(
                decay=decay,
            )

            model.fit(
                pd.to_numeric(
                    train_asset["return"],
                    errors="coerce",
                )
            )

            forecast = model.forecast_volatility

            for _, row in test_asset.iterrows():
                rows.append(
                    {
                        "timestamp": row[
                            "timestamp"
                        ],
                        "asset_id": asset_id,
                        "prediction": forecast,
                        "actual_volatility": float(
                            row["target"]
                        ),
                    }
                )

        if not rows:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} EWMA produced "
                "no predictions."
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # GJR-GARCH
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_gjr_garch(
        *,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
    ) -> pd.DataFrame:
        required = {
            "return",
            "target",
        }

        missing = required - set(
            train_data.columns
        )

        if missing:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} GJR-GARCH is missing "
                "required columns: "
                + ", ".join(sorted(missing))
            )

        rows: list[dict[str, object]] = []

        for asset_id in sorted(
            train_data["asset_id"]
            .astype(str)
            .unique()
        ):
            train_asset = train_data.loc[
                train_data["asset_id"].astype(str)
                == asset_id
            ].copy()

            test_asset = test_data.loc[
                test_data["asset_id"].astype(str)
                == asset_id
            ].copy()

            if test_asset.empty:
                continue

            model = GJRGARCH()

            model.fit(
                pd.to_numeric(
                    train_asset["return"],
                    errors="coerce",
                )
            )

            forecast = model.forecast_volatility()

            for _, row in test_asset.iterrows():
                rows.append(
                    {
                        "timestamp": row[
                            "timestamp"
                        ],
                        "asset_id": asset_id,
                        "prediction": forecast,
                        "actual_volatility": float(
                            row["target"]
                        ),
                    }
                )

        if not rows:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} GJR-GARCH "
                "produced no predictions."
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_xgboost(
        *,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
        target_column: str,
        xgboost_config: XGBoostConfig | None,
        feature_columns: tuple[str, ...] | None,
    ) -> pd.DataFrame:
        model = XGBoostVolatility(
            config=xgboost_config,
        )

        model.fit(
            features=train_data,
            target=train_data[
                target_column
            ],
            feature_columns=feature_columns,
        )

        predictions = model.predict(
            test_data
        )

        result = test_data.loc[
            :,
            [
                "timestamp",
                "asset_id",
            ],
        ].copy()

        result["prediction"] = predictions

        result["actual_volatility"] = (
            test_data[target_column]
            .to_numpy(dtype=float)
        )

        return result

    # ------------------------------------------------------------------
    # Regime-XGBoost
    # ------------------------------------------------------------------

    def _predict_regime_xgboost(
        self,
        *,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
        target_column: str,
        hmm_config: HMMConfig,
        xgboost_config: XGBoostConfig | None,
        feature_columns: tuple[str, ...] | None,
    ) -> pd.DataFrame:
        (
            train_features,
            test_features,
            regime_columns,
        ) = self._attach_hmm_features(
            train_data,
            test_data,
            fold,
            hmm_config=hmm_config,
        )

        (
            train_features,
            test_features,
        ) = self._attach_garch_features(
            train_features,
            test_features,
            fold,
        )

        model = RegimeXGBoost(
            config=xgboost_config,
        )

        model.fit(
            features=train_features,
            target=train_features[
                target_column
            ],
            regime_columns=regime_columns,
            feature_columns=feature_columns,
        )

        predictions = model.predict(
            test_features
        )

        result = test_features.loc[
            :,
            [
                "timestamp",
                "asset_id",
            ],
        ].copy()

        result["prediction"] = predictions

        result["actual_volatility"] = (
            test_features[target_column]
            .to_numpy(dtype=float)
        )

        result["garch_forecast"] = (
            test_features[
                "garch_forecast"
            ].to_numpy(dtype=float)
        )

        for column in regime_columns:
            result[column] = (
                test_features[column]
                .to_numpy(dtype=float)
            )

        return result

    # ------------------------------------------------------------------
    # GARCH feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_garch_features(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if "return" not in train_data.columns:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} GARCH features "
                "require a 'return' column."
            )

        train_result = train_data.copy()
        test_result = test_data.copy()

        train_result["garch_forecast"] = np.nan
        test_result["garch_forecast"] = np.nan

        for asset_id in sorted(
            train_result["asset_id"]
            .astype(str)
            .unique()
        ):
            train_mask = (
                train_result["asset_id"]
                .astype(str)
                == asset_id
            )

            test_mask = (
                test_result["asset_id"]
                .astype(str)
                == asset_id
            )

            train_asset = train_result.loc[
                train_mask
            ]

            test_asset = test_result.loc[
                test_mask
            ]

            if test_asset.empty:
                continue

            model = GJRGARCH()

            model.fit(
                pd.to_numeric(
                    train_asset["return"],
                    errors="coerce",
                )
            )

            conditional_volatility = (
                model.conditional_volatility()
                .to_numpy(dtype=float)
            )

            if len(
                conditional_volatility
            ) != len(train_asset):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} GARCH conditional "
                    f"volatility length does not match "
                    f"training observations for asset "
                    f"'{asset_id}'."
                )

            train_result.loc[
                train_asset.index,
                "garch_forecast",
            ] = conditional_volatility

            forecast = (
                model.forecast_volatility()
            )

            test_result.loc[
                test_asset.index,
                "garch_forecast",
            ] = forecast

        if (
            train_result["garch_forecast"]
            .isna()
            .any()
        ):
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} GARCH feature "
                "generation produced missing training values."
            )

        if (
            test_result["garch_forecast"]
            .isna()
            .any()
        ):
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} GARCH feature "
                "generation produced missing test values."
            )

        return (
            train_result,
            test_result,
        )

    # ------------------------------------------------------------------
    # HMM feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_hmm_features(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WalkForwardFoldData,
        *,
        hmm_config: HMMConfig,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        tuple[str, ...],
    ]:
        required = {
            "return",
            "realized_volatility",
            "high",
            "low",
            "close",
        }

        missing = required - set(
            train_data.columns
        )

        if missing:
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} HMM is missing "
                "required columns: "
                + ", ".join(sorted(missing))
            )

        train_result = train_data.copy()
        test_result = test_data.copy()

        train_result["range"] = (
            (
                pd.to_numeric(
                    train_result["high"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    train_result["low"],
                    errors="coerce",
                )
            )
            / pd.to_numeric(
                train_result["close"],
                errors="coerce",
            ).abs()
        )

        test_result["range"] = (
            (
                pd.to_numeric(
                    test_result["high"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    test_result["low"],
                    errors="coerce",
                )
            )
            / pd.to_numeric(
                test_result["close"],
                errors="coerce",
            ).abs()
        )

        observation_columns = (
            "return",
            "realized_volatility",
            "range",
        )

        regime_columns = tuple(
            f"regime_probability_{index}"
            for index in range(
                hmm_config.n_components
            )
        )

        for column in regime_columns:
            train_result[column] = np.nan
            test_result[column] = np.nan

        for asset_id in sorted(
            train_result["asset_id"]
            .astype(str)
            .unique()
        ):
            train_mask = (
                train_result["asset_id"]
                .astype(str)
                == asset_id
            )

            test_mask = (
                test_result["asset_id"]
                .astype(str)
                == asset_id
            )

            train_asset = train_result.loc[
                train_mask
            ].copy()

            test_asset = test_result.loc[
                test_mask
            ].copy()

            if test_asset.empty:
                continue

            train_observations = (
                train_asset.loc[
                    :,
                    observation_columns,
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            )

            test_observations = (
                test_asset.loc[
                    :,
                    observation_columns,
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            )

            if (
                train_observations
                .isna()
                .any()
                .any()
            ):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} HMM training "
                    f"observations contain missing values "
                    f"for asset '{asset_id}'."
                )

            if (
                test_observations
                .isna()
                .any()
                .any()
            ):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} HMM test "
                    f"observations contain missing values "
                    f"for asset '{asset_id}'."
                )

            hmm = HMMRegimeDetector(
                config=hmm_config,
            )

            hmm.fit(
                train_observations
            )

            train_probabilities = (
                hmm.filter_probabilities(
                    train_observations
                )
            )

            test_probabilities = (
                hmm.filter_after_training(
                    test_observations
                )
            )

            if (
                train_probabilities.shape[1]
                != len(regime_columns)
            ):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} HMM produced "
                    "an unexpected number of regime probabilities."
                )

            train_result.loc[
                train_asset.index,
                list(regime_columns),
            ] = train_probabilities

            test_result.loc[
                test_asset.index,
                list(regime_columns),
            ] = test_probabilities

        for column in regime_columns:
            if (
                train_result[column]
                .isna()
                .any()
            ):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} HMM produced "
                    f"missing training probabilities in '{column}'."
                )

            if (
                test_result[column]
                .isna()
                .any()
            ):
                raise WalkForwardRunnerError(
                    f"Fold {fold.fold_id} HMM produced "
                    f"missing test probabilities in '{column}'."
                )

        return (
            train_result,
            test_result,
            regime_columns,
        )

    # ------------------------------------------------------------------
    # Fold utilities
    # ------------------------------------------------------------------

    def _require_one_step_test_window(
        self,
    ) -> None:
        if self.splitter.test_size != 1:
            raise WalkForwardRunnerError(
                "AVF-TRPDE volatility forecasting requires "
                "test_size=1 for one-step-ahead walk-forward "
                "execution when GJR-GARCH features or forecasts "
                "are used."
            )

    @staticmethod
    def _select_fold_data(
        dataframe: pd.DataFrame,
        fold: WalkForwardFoldData,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
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

        train_data = (
            train_data
            .sort_values(
                by=[
                    "timestamp",
                    "asset_id",
                ],
                ascending=True,
            )
            .reset_index(drop=True)
        )

        test_data = (
            test_data
            .sort_values(
                by=[
                    "timestamp",
                    "asset_id",
                ],
                ascending=True,
            )
            .reset_index(drop=True)
        )

        return (
            train_data,
            test_data,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_research_input(
        dataframe: pd.DataFrame,
        *,
        target_column: str,
    ) -> None:
        WalkForwardRunner._validate_input(
            dataframe
        )

        required = {
            target_column,
            "return",
        }

        missing = required - set(
            dataframe.columns
        )

        if missing:
            raise WalkForwardRunnerError(
                "Research dataframe is missing required "
                "columns: "
                + ", ".join(sorted(missing))
            )

        target = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        )

        if target.isna().any():
            raise WalkForwardRunnerError(
                f"{target_column} contains missing or "
                "non-numeric values."
            )

        if not np.isfinite(
            target.to_numpy(
                dtype=float
            )
        ).all():
            raise WalkForwardRunnerError(
                f"{target_column} contains non-finite values."
            )

        if (
            target.to_numpy(
                dtype=float
            )
            < 0
        ).any():
            raise WalkForwardRunnerError(
                f"{target_column} contains negative "
                "volatility values."
            )

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
                + ", ".join(
                    sorted(missing_columns)
                )
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
            .str.strip()
            .str.upper()
        )

        if result["asset_id"].isna().any():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "missing asset IDs."
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

        prediction_values = (
            result["prediction"]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            prediction_values
        ).all():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "non-finite values."
            )

        if (
            prediction_values < 0
        ).any():
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "negative volatility values."
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
                pd.to_datetime(
                    test_data["timestamp"]
                ),
                test_data[
                    "asset_id"
                ]
                .astype("string")
                .str.strip()
                .str.upper(),
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

        if (
            result.duplicated(
                subset=[
                    "timestamp",
                    "asset_id",
                ]
            )
            .any()
        ):
            raise WalkForwardRunnerError(
                f"Fold {fold.fold_id} predictions contain "
                "duplicate timestamp/asset observations."
            )

        return (
            result
            .sort_values(
                by=[
                    "timestamp",
                    "asset_id",
                ],
                ascending=True,
            )
            .reset_index(drop=True)
        )

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
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise WalkForwardRunnerError(
                "Input dataframe contains invalid timestamps."
            )

        if dataframe[
            "asset_id"
        ].isna().any():
            raise WalkForwardRunnerError(
                "asset_id contains missing values."
            )

        normalized_assets = (
            dataframe["asset_id"]
            .astype("string")
            .str.strip()
        )

        if normalized_assets.eq("").any():
            raise WalkForwardRunnerError(
                "asset_id contains empty values."
            )