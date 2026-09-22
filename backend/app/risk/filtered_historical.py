from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "return",
    "forecast_volatility",
)


class FilteredHistoricalSimulationError(ValueError):
    """Raised when filtered historical simulation cannot be performed."""


@dataclass(frozen=True)
class FHSResult:
    """Result of filtered historical simulation."""

    standardized_returns: pd.Series
    simulated_returns: np.ndarray
    forecast_volatility: float
    observation_count: int


class FilteredHistoricalSimulation:
    """
    Filtered Historical Simulation (FHS).

    Historical returns are standardized using their corresponding
    volatility forecasts:

        z_t = r_t / sigma_t

    The empirical standardized-return distribution is then rescaled
    by the current forecast volatility:

        r_(t+1) = sigma_(t+1) * z

    This produces a forecast return distribution for downstream
    VaR and Expected Shortfall calculations.
    """

    def __init__(
        self,
        *,
        min_observations: int = 100,
        random_state: int | None = 42,
    ) -> None:
        if (
            not isinstance(min_observations, int)
            or isinstance(min_observations, bool)
        ):
            raise TypeError(
                "min_observations must be an integer."
            )

        if min_observations < 1:
            raise ValueError(
                "min_observations must be at least 1."
            )

        self.min_observations = min_observations
        self.random_state = random_state

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate standardized historical returns.

        Returns:
            Series containing valid standardized returns.
        """

        _validate_input(dataframe)

        df = dataframe.loc[:, REQUIRED_COLUMNS].copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df["asset_id"] = df["asset_id"].astype("string")

        df["return"] = pd.to_numeric(
            df["return"],
            errors="coerce",
        )

        df["forecast_volatility"] = pd.to_numeric(
            df["forecast_volatility"],
            errors="coerce",
        )

        df = df.sort_values(
            by=["asset_id", "timestamp"],
            ascending=True,
        ).reset_index(drop=True)

        valid = (
            np.isfinite(df["return"])
            & np.isfinite(df["forecast_volatility"])
            & (df["forecast_volatility"] > 0)
        )

        df = df.loc[valid].copy()

        if len(df) < self.min_observations:
            raise FilteredHistoricalSimulationError(
                "Insufficient valid observations for FHS. "
                f"Required at least {self.min_observations}, "
                f"received {len(df)}."
            )

        df["standardized_return"] = (
            df["return"]
            / df["forecast_volatility"]
        )

        standardized = df["standardized_return"]

        standardized = standardized.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(standardized) < self.min_observations:
            raise FilteredHistoricalSimulationError(
                "Insufficient finite standardized returns for FHS. "
                f"Required at least {self.min_observations}, "
                f"received {len(standardized)}."
            )

        return standardized.reset_index(drop=True)

    def simulate(
        self,
        standardized_returns: pd.Series | np.ndarray,
        forecast_volatility: float,
        *,
        n_simulations: int = 10_000,
    ) -> np.ndarray:
        """
        Rescale empirical standardized returns using forecast volatility.

        Sampling is performed with replacement from the empirical
        standardized-return distribution.
        """

        standardized = _validate_standardized_returns(
            standardized_returns
        )

        if not np.isfinite(forecast_volatility):
            raise FilteredHistoricalSimulationError(
                "forecast_volatility must be finite."
            )

        if forecast_volatility <= 0:
            raise FilteredHistoricalSimulationError(
                "forecast_volatility must be greater than zero."
            )

        if (
            not isinstance(n_simulations, int)
            or isinstance(n_simulations, bool)
        ):
            raise TypeError(
                "n_simulations must be an integer."
            )

        if n_simulations < 1:
            raise ValueError(
                "n_simulations must be at least 1."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        sampled = rng.choice(
            standardized,
            size=n_simulations,
            replace=True,
        )

        simulated_returns = (
            forecast_volatility * sampled
        )

        return np.asarray(
            simulated_returns,
            dtype=float,
        )

    def fit_simulate(
        self,
        dataframe: pd.DataFrame,
        forecast_volatility: float,
        *,
        n_simulations: int = 10_000,
    ) -> FHSResult:
        """
        Fit FHS on historical observations and generate
        the forecast return distribution.
        """

        standardized = self.fit(dataframe)

        simulated_returns = self.simulate(
            standardized_returns=standardized,
            forecast_volatility=forecast_volatility,
            n_simulations=n_simulations,
        )

        return FHSResult(
            standardized_returns=standardized,
            simulated_returns=simulated_returns,
            forecast_volatility=float(
                forecast_volatility
            ),
            observation_count=len(standardized),
        )


def filtered_historical_simulation(
    dataframe: pd.DataFrame,
    forecast_volatility: float,
    *,
    min_observations: int = 100,
    n_simulations: int = 10_000,
    random_state: int | None = 42,
) -> FHSResult:
    """
    Convenience function for running FHS in one call.
    """

    model = FilteredHistoricalSimulation(
        min_observations=min_observations,
        random_state=random_state,
    )

    return model.fit_simulate(
        dataframe=dataframe,
        forecast_volatility=forecast_volatility,
        n_simulations=n_simulations,
    )


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise FilteredHistoricalSimulationError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise FilteredHistoricalSimulationError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise FilteredHistoricalSimulationError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise FilteredHistoricalSimulationError(
            "asset_id contains missing values."
        )

    returns = pd.to_numeric(
        dataframe["return"],
        errors="coerce",
    )

    volatility = pd.to_numeric(
        dataframe["forecast_volatility"],
        errors="coerce",
    )

    if returns.isna().any():
        raise FilteredHistoricalSimulationError(
            "return contains invalid or missing values."
        )

    if volatility.isna().any():
        raise FilteredHistoricalSimulationError(
            "forecast_volatility contains invalid or missing values."
        )

    if (~np.isfinite(returns)).any():
        raise FilteredHistoricalSimulationError(
            "return contains non-finite values."
        )

    if (~np.isfinite(volatility)).any():
        raise FilteredHistoricalSimulationError(
            "forecast_volatility contains non-finite values."
        )

    if (volatility <= 0).any():
        raise FilteredHistoricalSimulationError(
            "forecast_volatility must be greater than zero."
        )


def _validate_standardized_returns(
    standardized_returns: pd.Series | np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        standardized_returns,
        dtype=float,
    )

    if values.ndim != 1:
        raise FilteredHistoricalSimulationError(
            "standardized_returns must be one-dimensional."
        )

    if values.size == 0:
        raise FilteredHistoricalSimulationError(
            "standardized_returns must not be empty."
        )

    if not np.isfinite(values).all():
        raise FilteredHistoricalSimulationError(
            "standardized_returns contains non-finite values."
        )

    return values