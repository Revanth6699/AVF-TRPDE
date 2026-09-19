from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


IDENTIFIER_COLUMNS = (
    "timestamp",
    "asset_id",
)

DEFAULT_EXCLUDED_COLUMNS = (
    "target",
    "realized_volatility",
    "realized_variance",
)


class XGBoostModelError(ValueError):
    """Raised when the XGBoost volatility model cannot be used."""


@dataclass(frozen=True)
class XGBoostConfig:
    """
    Configuration for the XGBoost volatility model.

    Hyperparameters remain configurable because the locked AVF-TRPDE
    specification defines the model and feature groups but does not
    prescribe fixed hyperparameter values.
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    objective: str = "reg:squarederror"


@dataclass(frozen=True)
class XGBoostResult:
    """Result metadata from an XGBoost volatility forecast."""

    predictions: np.ndarray
    feature_columns: tuple[str, ...]
    observation_count: int


class XGBoostVolatility:
    """
    XGBoost regression model for next-day realized volatility.

    This is the non-regime ML baseline.

    The model does not perform:
    - HMM regime detection
    - regime-probability feature generation
    - portfolio trading decisions
    - rule-based decisions

    Walk-forward fitting is handled by the walk-forward engine. This
    class only fits the model on the data supplied to it.
    """

    def __init__(
        self,
        config: XGBoostConfig | None = None,
    ) -> None:
        self.config = config or XGBoostConfig()

        self._validate_config()

        self._model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            objective=self.config.objective,
            n_jobs=1,
        )

        self._feature_columns: tuple[str, ...] | None = None
        self._fitted = False

    @property
    def model(self) -> XGBRegressor:
        """Return the underlying XGBoost regressor."""

        return self._model

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the feature columns used during fitting."""

        if self._feature_columns is None:
            raise XGBoostModelError(
                "XGBoost model has not been fitted."
            )

        return self._feature_columns

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

        return self._fitted

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> "XGBoostVolatility":
        """
        Fit XGBoost on a training dataset.

        Only the supplied training observations are used. No train/test
        splitting or random validation is performed here because temporal
        validation belongs to the walk-forward engine.
        """

        X, y, columns = self._prepare_training_data(
            features=features,
            target=target,
            feature_columns=feature_columns,
        )

        try:
            self._model.fit(
                X,
                y,
            )
        except Exception as exc:
            raise XGBoostModelError(
                f"XGBoost fitting failed: {exc}"
            ) from exc

        self._feature_columns = columns
        self._fitted = True

        return self

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate volatility forecasts for supplied observations.
        """

        self._require_fitted()

        X = self._prepare_prediction_data(features)

        try:
            predictions = np.asarray(
                self._model.predict(X),
                dtype=float,
            )
        except Exception as exc:
            raise XGBoostModelError(
                f"XGBoost prediction failed: {exc}"
            ) from exc

        if not np.isfinite(predictions).all():
            raise XGBoostModelError(
                "XGBoost produced non-finite predictions."
            )

        if (predictions < 0).any():
            raise XGBoostModelError(
                "XGBoost produced negative volatility predictions."
            )

        return predictions

    def fit_predict(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> XGBoostResult:
        """Fit XGBoost and generate forecasts for the same supplied data."""

        self.fit(
            features=features,
            target=target,
            feature_columns=feature_columns,
        )

        predictions = self.predict(features)

        return XGBoostResult(
            predictions=predictions,
            feature_columns=self.feature_columns,
            observation_count=len(predictions),
        )

    def predict_dataframe(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return predictions alongside timestamp and asset identifiers
        when those columns are present.
        """

        predictions = self.predict(features)

        result_columns = [
            column
            for column in IDENTIFIER_COLUMNS
            if column in features.columns
        ]

        result = features.loc[:, result_columns].copy()

        result["xgboost_forecast"] = predictions

        return result

    def feature_importance(self) -> pd.DataFrame:
        """
        Return fitted feature importance values.

        Importance values are reported directly from the fitted XGBoost
        model and are not used as a research ranking.
        """

        self._require_fitted()

        importance = np.asarray(
            self._model.feature_importances_,
            dtype=float,
        )

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": importance,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def _prepare_training_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_columns: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "features must be a pandas.DataFrame."
            )

        if not isinstance(target, pd.Series):
            raise TypeError(
                "target must be a pandas.Series."
            )

        if features.empty:
            raise XGBoostModelError(
                "features must not be empty."
            )

        if target.empty:
            raise XGBoostModelError(
                "target must not be empty."
            )

        if len(features) != len(target):
            raise XGBoostModelError(
                "features and target must contain the same number "
                "of observations."
            )

        columns = self._resolve_feature_columns(
            features=features,
            feature_columns=feature_columns,
        )

        X = features.loc[:, columns].copy()

        y = pd.to_numeric(
            target,
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(y).all():
            raise XGBoostModelError(
                "target contains non-finite values."
            )

        if (y < 0).any():
            raise XGBoostModelError(
                "realized-volatility targets must not be negative."
            )

        X = self._validate_feature_matrix(X)

        return X, y, columns

    def _prepare_prediction_data(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "features must be a pandas.DataFrame."
            )

        if features.empty:
            raise XGBoostModelError(
                "features must not be empty."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in features.columns
        ]

        if missing_columns:
            raise XGBoostModelError(
                "Prediction data is missing feature columns: "
                + ", ".join(missing_columns)
            )

        X = features.loc[
            :,
            self.feature_columns,
        ].copy()

        return self._validate_feature_matrix(X)

    @staticmethod
    def _resolve_feature_columns(
        features: pd.DataFrame,
        feature_columns: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if feature_columns is not None:
            columns = tuple(feature_columns)

            if not columns:
                raise XGBoostModelError(
                    "At least one feature column is required."
                )

            missing_columns = [
                column
                for column in columns
                if column not in features.columns
            ]

            if missing_columns:
                raise XGBoostModelError(
                    "Missing feature columns: "
                    + ", ".join(missing_columns)
                )

        else:
            columns = tuple(
                column
                for column in features.columns
                if column not in IDENTIFIER_COLUMNS
                and column not in DEFAULT_EXCLUDED_COLUMNS
            )

        if not columns:
            raise XGBoostModelError(
                "No usable feature columns were found."
            )

        for column in columns:
            if not pd.api.types.is_numeric_dtype(
                features[column]
            ):
                raise XGBoostModelError(
                    f"Feature column '{column}' must be numeric."
                )

        return columns

    @staticmethod
    def _validate_feature_matrix(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        numeric = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if numeric.isna().any().any():
            invalid_columns = [
                column
                for column in numeric.columns
                if numeric[column].isna().any()
            ]

            raise XGBoostModelError(
                "Feature matrix contains invalid or missing values: "
                + ", ".join(invalid_columns)
            )

        values = numeric.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise XGBoostModelError(
                "Feature matrix contains non-finite values."
            )

        return numeric

    def _validate_config(self) -> None:
        if self.config.n_estimators <= 0:
            raise XGBoostModelError(
                "n_estimators must be greater than zero."
            )

        if self.config.max_depth <= 0:
            raise XGBoostModelError(
                "max_depth must be greater than zero."
            )

        if self.config.learning_rate <= 0:
            raise XGBoostModelError(
                "learning_rate must be greater than zero."
            )

        if not 0 < self.config.subsample <= 1:
            raise XGBoostModelError(
                "subsample must be in the interval (0, 1]."
            )

        if not 0 < self.config.colsample_bytree <= 1:
            raise XGBoostModelError(
                "colsample_bytree must be in the interval (0, 1]."
            )

        if self.config.min_child_weight < 0:
            raise XGBoostModelError(
                "min_child_weight must not be negative."
            )

        if self.config.reg_alpha < 0:
            raise XGBoostModelError(
                "reg_alpha must not be negative."
            )

        if self.config.reg_lambda < 0:
            raise XGBoostModelError(
                "reg_lambda must not be negative."
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise XGBoostModelError(
                "XGBoost model has not been fitted."
            )


def fit_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    feature_columns: Sequence[str] | None = None,
    config: XGBoostConfig | None = None,
) -> XGBoostVolatility:
    """Create and fit the non-regime XGBoost volatility model."""

    model = XGBoostVolatility(
        config=config,
    )

    model.fit(
        features=features,
        target=target,
        feature_columns=feature_columns,
    )

    return model