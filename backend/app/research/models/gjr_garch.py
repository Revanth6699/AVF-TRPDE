from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


class GJRGARCHError(ValueError):
    """Raised when GJR-GARCH fitting or forecasting fails."""


@dataclass(frozen=True)
class GJRGARCHResult:
    """Structured result from a fitted GJR-GARCH model."""

    conditional_variance: pd.Series
    conditional_volatility: pd.Series
    forecast_variance: float
    forecast_volatility: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    mu: float


class GJRGARCH:
    """
    GJR-GARCH(1,1,1) volatility model.

    Model:

        r_t = mu + epsilon_t

        sigma_t^2 =
            omega
            + alpha * epsilon_{t-1}^2
            + gamma * I_{t-1} * epsilon_{t-1}^2
            + beta * sigma_{t-1}^2

    where:

        I_{t-1} = 1 if epsilon_{t-1} < 0
                  0 otherwise

    The implementation uses the arch package for maximum-likelihood
    estimation and one-step-ahead conditional variance forecasting.
    """

    def __init__(
        self,
        *,
        mean: str = "Constant",
        p: int = 1,
        o: int = 1,
        q: int = 1,
        dist: str = "normal",
    ) -> None:
        if mean != "Constant":
            raise GJRGARCHError(
                "GJR-GARCH requires a constant mean specification."
            )

        if p != 1 or o != 1 or q != 1:
            raise GJRGARCHError(
                "Only GJR-GARCH(1,1,1) is supported by the "
                "locked model specification."
            )

        if dist != "normal":
            raise GJRGARCHError(
                "Only the normal innovation specification is supported."
            )

        self.mean = mean
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist

        self._model = None
        self._fit_result = None
        self._return_index: pd.Index | None = None

    def fit(
        self,
        returns: pd.Series,
    ) -> "GJRGARCH":
        """
        Fit GJR-GARCH(1,1,1) to a return series.

        Returns are scaled by 100 for numerical stability.
        Forecasts are converted back to the original return scale.
        """

        values = self._validate_returns(returns)

        scaled_returns = values * 100.0

        model = arch_model(
            scaled_returns,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
            rescale=False,
        )

        try:
            fit_result = model.fit(
                disp="off",
            )
        except Exception as exc:
            raise GJRGARCHError(
                f"GJR-GARCH fitting failed: {exc}"
            ) from exc

        if fit_result.convergence_flag != 0:
            raise GJRGARCHError(
                "GJR-GARCH optimizer did not converge."
            )

        self._model = model
        self._fit_result = fit_result
        self._return_index = returns.index.copy()

        return self

    @property
    def fitted_model(self):
        """Return the fitted arch model result."""

        self._require_fitted()
        return self._fit_result

    @property
    def mu(self) -> float:
        """Return the estimated constant mean on the original scale."""

        self._require_fitted()

        return float(
            self._fit_result.params["mu"] / 100.0
        )

    @property
    def omega(self) -> float:
        """Return the estimated omega parameter on the original scale."""

        self._require_fitted()

        return float(
            self._fit_result.params["omega"] / 10_000.0
        )

    @property
    def alpha(self) -> float:
        """Return the estimated alpha parameter."""

        self._require_fitted()

        return float(
            self._fit_result.params["alpha[1]"]
        )

    @property
    def gamma(self) -> float:
        """Return the estimated asymmetric gamma parameter."""

        self._require_fitted()

        return float(
            self._fit_result.params["gamma[1]"]
        )

    @property
    def beta(self) -> float:
        """Return the estimated beta parameter."""

        self._require_fitted()

        return float(
            self._fit_result.params["beta[1]"]
        )

    def conditional_variance(self) -> pd.Series:
        """
        Return fitted conditional variance on the original return scale.
        """

        self._require_fitted()

        conditional_volatility = np.asarray(
            self._fit_result.conditional_volatility,
            dtype=float,
        )

        if not np.isfinite(conditional_volatility).all():
            raise GJRGARCHError(
                "GJR-GARCH produced non-finite conditional volatility."
            )

        variance = (
            conditional_volatility**2
        ) / 10_000.0

        return pd.Series(
            variance,
            index=self._return_index,
            name="gjr_garch_variance",
        )

    def conditional_volatility(self) -> pd.Series:
        """
        Return fitted conditional volatility on the original return scale.
        """

        self._require_fitted()

        volatility = np.asarray(
            self._fit_result.conditional_volatility,
            dtype=float,
        ) / 100.0

        if not np.isfinite(volatility).all():
            raise GJRGARCHError(
                "GJR-GARCH produced non-finite conditional volatility."
            )

        return pd.Series(
            volatility,
            index=self._return_index,
            name="gjr_garch_volatility",
        )

    def forecast_variance(self) -> float:
        """Return the one-step-ahead conditional variance."""

        self._require_fitted()

        try:
            forecast = self._fit_result.forecast(
                horizon=1,
                reindex=False,
            )
        except Exception as exc:
            raise GJRGARCHError(
                f"GJR-GARCH variance forecasting failed: {exc}"
            ) from exc

        variance_scaled = float(
            np.asarray(
                forecast.variance,
                dtype=float,
            )[-1, 0]
        )

        variance = variance_scaled / 10_000.0

        if not np.isfinite(variance) or variance < 0:
            raise GJRGARCHError(
                "GJR-GARCH produced an invalid forecast variance."
            )

        return variance

    def forecast_volatility(self) -> float:
        """Return the one-step-ahead conditional volatility."""

        variance = self.forecast_variance()

        return float(np.sqrt(variance))

    def result(self) -> GJRGARCHResult:
        """Return the fitted model and forecast results."""

        self._require_fitted()

        return GJRGARCHResult(
            conditional_variance=self.conditional_variance(),
            conditional_volatility=self.conditional_volatility(),
            forecast_variance=self.forecast_variance(),
            forecast_volatility=self.forecast_volatility(),
            omega=self.omega,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            mu=self.mu,
        )

    def fit_result(
        self,
        returns: pd.Series,
    ) -> GJRGARCHResult:
        """Fit the model and return its structured result."""

        self.fit(returns)

        return self.result()

    def _require_fitted(self) -> None:
        if self._fit_result is None:
            raise GJRGARCHError(
                "GJR-GARCH model has not been fitted."
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
            raise GJRGARCHError(
                "returns must not be empty."
            )

        values = pd.to_numeric(
            returns,
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise GJRGARCHError(
                "returns must contain only finite numeric values."
            )

        if len(values) < 30:
            raise GJRGARCHError(
                "At least 30 return observations are required "
                "to fit GJR-GARCH."
            )

        return values


def fit_gjr_garch(
    returns: pd.Series,
) -> GJRGARCHResult:
    """Fit GJR-GARCH(1,1,1) and return the model result."""

    model = GJRGARCH()

    return model.fit_result(returns)