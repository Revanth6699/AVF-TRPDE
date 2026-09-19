from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "return",
)


class EWMAError(ValueError):
    """Raised when EWMA volatility cannot be calculated."""


@dataclass(frozen=True)
class EWMAResult:
    """Result of an EWMA volatility calculation."""

    dataframe: pd.DataFrame
    decay: float
    forecast_variance: float
    forecast_volatility: float


class EWMAVolatility:
    """
    Exponentially Weighted Moving Average volatility model.

    The model follows:

        sigma_t^2 =
            lambda * sigma_{t-1}^2
            + (1 - lambda) * r_{t-1}^2

    The one-step-ahead forecast uses information available through
    the final observed return only.
    """

    def __init__(self, decay: float) -> None:
        self._validate_decay(decay)
        self.decay = float(decay)

        self._variance: float | None = None
        self._forecast_variance: float | None = None
        self._forecast_volatility: float | None = None

    def fit(
        self,
        returns: pd.Series,
    ) -> "EWMAVolatility":
        """
        Fit EWMA variance recursively on a return series.

        The initial variance is the squared first observed return.
        Each subsequent variance uses the previous return.
        """

        values = self._validate_returns(returns)

        variance = np.empty(
            len(values),
            dtype=float,
        )

        variance[0] = values[0] ** 2

        for index in range(1, len(values)):
            variance[index] = (
                self.decay * variance[index - 1]
                + (1.0 - self.decay) * values[index - 1] ** 2
            )

        self._variance = float(variance[-1])

        last_return = float(values[-1])

        self._forecast_variance = (
            self.decay * self._variance
            + (1.0 - self.decay) * last_return**2
        )

        self._forecast_volatility = float(
            np.sqrt(self._forecast_variance)
        )

        return self

    @property
    def variance(self) -> float:
        """Return the latest fitted conditional variance."""

        if self._variance is None:
            raise EWMAError(
                "EWMA model has not been fitted."
            )

        return self._variance

    @property
    def forecast_variance(self) -> float:
        """Return the one-step-ahead forecast variance."""

        if self._forecast_variance is None:
            raise EWMAError(
                "EWMA model has not been fitted."
            )

        return self._forecast_variance

    @property
    def forecast_volatility(self) -> float:
        """Return the one-step-ahead forecast volatility."""

        if self._forecast_volatility is None:
            raise EWMAError(
                "EWMA model has not been fitted."
            )

        return self._forecast_volatility

    def transform(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculate the EWMA conditional variance and volatility
        for every observed return.
        """

        values = self._validate_returns(returns)

        variance = np.empty(
            len(values),
            dtype=float,
        )

        variance[0] = values[0] ** 2

        for index in range(1, len(values)):
            variance[index] = (
                self.decay * variance[index - 1]
                + (1.0 - self.decay) * values[index - 1] ** 2
            )

        return pd.DataFrame(
            {
                "ewma_variance": variance,
                "ewma_volatility": np.sqrt(variance),
            },
            index=returns.index,
        )

    def fit_transform(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Fit EWMA and return the fitted variance series."""

        result = self.fit(returns)
        transformed = result.transform(returns)

        transformed["ewma_forecast_variance"] = (
            result.forecast_variance
        )

        transformed["ewma_forecast_volatility"] = (
            result.forecast_volatility
        )

        return transformed

    def result(
        self,
        returns: pd.Series,
    ) -> EWMAResult:
        """Fit EWMA and return structured model output."""

        fitted = self.fit_transform(returns)

        return EWMAResult(
            dataframe=fitted,
            decay=self.decay,
            forecast_variance=self.forecast_variance,
            forecast_volatility=self.forecast_volatility,
        )

    @staticmethod
    def _validate_decay(decay: float) -> None:
        if not isinstance(decay, (float, int)):
            raise TypeError(
                "decay must be a numeric value."
            )

        if not np.isfinite(decay):
            raise EWMAError(
                "decay must be finite."
            )

        if not 0.0 < float(decay) < 1.0:
            raise EWMAError(
                "decay must be strictly between 0 and 1."
            )

    @staticmethod
    def _validate_returns(
        returns: pd.Series,
    ) -> np.ndarray:
        if not isinstance(returns, pd.Series):
            raise TypeError(
                "returns must be a pandas.Series."
            )

        if returns.empty:
            raise EWMAError(
                "returns must not be empty."
            )

        values = pd.to_numeric(
            returns,
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise EWMAError(
                "returns must contain only finite numeric values."
            )

        return values


def calculate_ewma(
    returns: pd.Series,
    decay: float,
) -> EWMAResult:
    """
    Calculate EWMA volatility and the one-step-ahead forecast.
    """

    model = EWMAVolatility(
        decay=decay,
    )

    return model.result(returns)