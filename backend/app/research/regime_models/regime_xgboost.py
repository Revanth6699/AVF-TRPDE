from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from backend.app.research.models.xgboost import (
    XGBoostConfig,
    XGBoostModelError,
    XGBoostVolatility,
)


class RegimeXGBoostError(ValueError):
    """Raised when the regime-aware XGBoost model cannot be used."""


@dataclass(frozen=True)
class RegimeXGBoostResult:
    """Result metadata from a regime-aware XGBoost forecast."""

    predictions: np.ndarray
    feature_columns: tuple[str, ...]
    regime_columns: tuple[str, ...]
    observation_count: int


class RegimeXGBoost:
    """
    Regime-Aware XGBoost volatility model.

    The model extends the non-regime XGBoost baseline with HMM
    regime-probability features.

    Feature structure:

        Market features
        + GJR-GARCH forecast
        + HMM regime probabilities
        --------------------------------
        -> XGBoost
        -> next-day realized-volatility forecast

    HMM fitting and probability generation are intentionally outside
    this class. They must be performed inside the appropriate
    walk-forward training fold.
    """

    def __init__(
        self,
        config: XGBoostConfig | None = None,
    ) -> None:
        self.config = config or XGBoostConfig()

        self._model = XGBoostVolatility(
            config=self.config,
        )

        self._feature_columns: tuple[str, ...] | None = None
        self._regime_columns: tuple[str, ...] | None = None
        self._fitted = False

    @property
    def model(self) -> XGBoostVolatility:
        """Return the underlying XGBoost volatility model."""

        return self._model

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return all features used by the fitted model."""

        if self._feature_columns is None:
            raise RegimeXGBoostError(
                "Regime-XGBoost model has not been fitted."
            )

        return self._feature_columns

    @property
    def regime_columns(self) -> tuple[str, ...]:
        """Return the HMM probability columns used by the model."""

        if self._regime_columns is None:
            raise RegimeXGBoostError(
                "Regime-XGBoost model has not been fitted."
            )

        return self._regime_columns

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

        return self._fitted

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        regime_columns: Sequence[str],
        feature_columns: Sequence[str] | None = None,
    ) -> "RegimeXGBoost":
        """
        Fit Regime-XGBoost.

        Parameters
        ----------
        features:
            Feature dataframe containing normal market/GARCH features
            and HMM regime-probability columns.

        target:
            Next-day realized-volatility target.

        regime_columns:
            HMM probability columns, for example:
            regime_probability_0,
            regime_probability_1.

        feature_columns:
            Optional explicit list of non-regime features.

        Only the supplied training observations are used. Temporal
        splitting is handled by the walk-forward engine.
        """

        regime_columns_tuple = self._validate_regime_columns(
            features,
            regime_columns,
        )

        base_columns = self._resolve_base_features(
            features=features,
            regime_columns=regime_columns_tuple,
            feature_columns=feature_columns,
        )

        combined_columns = (
            *base_columns,
            *regime_columns_tuple,
        )

        training_features = self._prepare_features(
            features=features,
            columns=combined_columns,
            regime_columns=regime_columns_tuple,
        )

        try:
            self._model.fit(
                features=training_features,
                target=target,
                feature_columns=combined_columns,
            )
        except XGBoostModelError as exc:
            raise RegimeXGBoostError(
                f"Regime-XGBoost fitting failed: {exc}"
            ) from exc

        self._feature_columns = combined_columns
        self._regime_columns = regime_columns_tuple
        self._fitted = True

        return self

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Generate regime-aware volatility forecasts."""

        self._require_fitted()

        prepared_features = self._prepare_features(
            features=features,
            columns=self.feature_columns,
            regime_columns=self.regime_columns,
        )

        try:
            predictions = self._model.predict(
                prepared_features
            )
        except XGBoostModelError as exc:
            raise RegimeXGBoostError(
                f"Regime-XGBoost prediction failed: {exc}"
            ) from exc

        return predictions

    def fit_predict(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        regime_columns: Sequence[str],
        feature_columns: Sequence[str] | None = None,
    ) -> RegimeXGBoostResult:
        """Fit Regime-XGBoost and generate predictions."""

        self.fit(
            features=features,
            target=target,
            regime_columns=regime_columns,
            feature_columns=feature_columns,
        )

        predictions = self.predict(
            features
        )

        return RegimeXGBoostResult(
            predictions=predictions,
            feature_columns=self.feature_columns,
            regime_columns=self.regime_columns,
            observation_count=len(predictions),
        )

    def predict_dataframe(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return forecasts together with timestamp and asset identifiers
        when available.
        """

        predictions = self.predict(
            features
        )

        identifier_columns = [
            column
            for column in (
                "timestamp",
                "asset_id",
            )
            if column in features.columns
        ]

        result = features.loc[
            :,
            identifier_columns,
        ].copy()

        result["regime_xgboost_forecast"] = predictions

        return result

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance from the fitted XGBoost model.

        This reports model output only and does not constitute a
        research ranking of models.
        """

        self._require_fitted()

        return self._model.feature_importance()

    def _prepare_features(
        self,
        features: pd.DataFrame,
        columns: Sequence[str],
        regime_columns: Sequence[str],
    ) -> pd.DataFrame:
        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas.DataFrame."
            )

        if features.empty:
            raise RegimeXGBoostError(
                "features must not be empty."
            )

        columns_tuple = tuple(columns)

        missing_columns = [
            column
            for column in columns_tuple
            if column not in features.columns
        ]

        if missing_columns:
            raise RegimeXGBoostError(
                "Feature dataframe is missing columns: "
                + ", ".join(missing_columns)
            )

        prepared = features.loc[
            :,
            columns_tuple,
        ].copy()

        for column in columns_tuple:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

        if prepared.isna().any().any():
            invalid_columns = [
                column
                for column in prepared.columns
                if prepared[column].isna().any()
            ]

            raise RegimeXGBoostError(
                "Feature dataframe contains invalid or missing "
                "values: "
                + ", ".join(invalid_columns)
            )

        values = prepared.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise RegimeXGBoostError(
                "Feature dataframe contains non-finite values."
            )

        self._validate_regime_probabilities(
            prepared,
            regime_columns,
        )

        return prepared

    @staticmethod
    def _validate_regime_columns(
        features: pd.DataFrame,
        regime_columns: Sequence[str],
    ) -> tuple[str, ...]:
        if not regime_columns:
            raise RegimeXGBoostError(
                "At least one HMM regime-probability column is required."
            )

        columns = tuple(regime_columns)

        if len(set(columns)) != len(columns):
            raise RegimeXGBoostError(
                "Regime-probability columns must be unique."
            )

        missing_columns = [
            column
            for column in columns
            if column not in features.columns
        ]

        if missing_columns:
            raise RegimeXGBoostError(
                "Missing regime-probability columns: "
                + ", ".join(missing_columns)
            )

        return columns

    @staticmethod
    def _resolve_base_features(
        features: pd.DataFrame,
        regime_columns: Sequence[str],
        feature_columns: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if feature_columns is not None:
            columns = tuple(feature_columns)

            if not columns:
                raise RegimeXGBoostError(
                    "At least one base feature is required."
                )
        else:
            excluded = {
                "timestamp",
                "asset_id",
                "target",
                "realized_volatility",
                "realized_variance",
                *regime_columns,
            }

            columns = tuple(
                column
                for column in features.columns
                if column not in excluded
            )

        if not columns:
            raise RegimeXGBoostError(
                "No base features were found."
            )

        missing_columns = [
            column
            for column in columns
            if column not in features.columns
        ]

        if missing_columns:
            raise RegimeXGBoostError(
                "Missing base feature columns: "
                + ", ".join(missing_columns)
            )

        overlap = set(columns).intersection(
            regime_columns
        )

        if overlap:
            raise RegimeXGBoostError(
                "Base features cannot also be regime-probability "
                "columns: "
                + ", ".join(sorted(overlap))
            )

        return columns

    @staticmethod
    def _validate_regime_probabilities(
        features: pd.DataFrame,
        regime_columns: Sequence[str],
    ) -> None:
        probabilities = features.loc[
            :,
            regime_columns,
        ].to_numpy(
            dtype=float
        )

        if (probabilities < 0).any():
            raise RegimeXGBoostError(
                "HMM regime probabilities must not be negative."
            )

        if (probabilities > 1).any():
            raise RegimeXGBoostError(
                "HMM regime probabilities must not exceed 1."
            )

        row_sums = probabilities.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            raise RegimeXGBoostError(
                "HMM regime probabilities must sum to one "
                "for every observation."
            )

    def _require_fitted(self) -> None:
        """Ensure the model has been fitted."""

        if not self._fitted:
            raise RegimeXGBoostError(
                "Regime-XGBoost model has not been fitted."
            )


def fit_regime_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    regime_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    config: XGBoostConfig | None = None,
) -> RegimeXGBoost:
    """Create and fit the Regime-Aware XGBoost model."""

    model = RegimeXGBoost(
        config=config,
    )

    model.fit(
        features=features,
        target=target,
        regime_columns=regime_columns,
        feature_columns=feature_columns,
    )

    return model