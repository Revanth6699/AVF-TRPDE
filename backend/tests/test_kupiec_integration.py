from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.risk.filtered_historical import (
    FilteredHistoricalSimulation,
)
from backend.app.risk.var import (
    calculate_var_95,
    calculate_var_99,
)
from backend.app.risk.kupiec import (
    kupiec_95,
    kupiec_99,
)


def main() -> None:
    print(
        "\n========== "
        "PHASE 3 / KUPIEC INTEGRATION "
        "=========="
    )

    # ---------------------------------------------------------------
    # FHS
    # ---------------------------------------------------------------

    rng = np.random.default_rng(42)

    n = 500

    historical_returns = rng.normal(
        0.0,
        0.01,
        n,
    )

    historical_volatility = rng.uniform(
        0.008,
        0.015,
        n,
    )

    historical_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=n,
                freq="D",
            ),
            "asset_id": ["AAPL"] * n,
            "return": historical_returns,
            "forecast_volatility": historical_volatility,
        }
    )

    fhs = FilteredHistoricalSimulation(
        min_observations=100,
        random_state=42,
    )

    standardized = fhs.fit(
        historical_data
    )

    print("\n--- FHS ---")

    print(
        "HISTORICAL OBSERVATIONS:",
        len(historical_data),
    )

    print(
        "STANDARDIZED OBSERVATIONS:",
        len(standardized),
    )

    standardized_values = np.asarray(
        standardized,
        dtype=float,
    )

    print(
        "STANDARDIZED FINITE:",
        np.isfinite(
            standardized_values
        ).all(),
    )

    # ---------------------------------------------------------------
    # FHS simulation
    # ---------------------------------------------------------------

    forecast_volatility = 0.012

    simulated = fhs.simulate(
        standardized,
        forecast_volatility,
        n_simulations=10000,
    )

    simulated_values = np.asarray(
        simulated,
        dtype=float,
    )

    print("\n--- FHS Simulation ---")

    print(
        "SIMULATIONS:",
        len(simulated_values),
    )

    print(
        "SIMULATED FINITE:",
        np.isfinite(
            simulated_values
        ).all(),
    )

    print(
        "FORECAST VOLATILITY:",
        forecast_volatility,
    )

    # ---------------------------------------------------------------
    # VaR
    # ---------------------------------------------------------------

    var95 = calculate_var_95(
        simulated_values
    )

    var99 = calculate_var_99(
        simulated_values
    )

    print("\n--- VaR ---")

    print(
        "VaR95:",
        var95.var,
    )

    print(
        "VaR99:",
        var99.var,
    )

    # ---------------------------------------------------------------
    # Deterministic backtest sample
    # ---------------------------------------------------------------

    test_returns = rng.normal(
        0.0,
        0.01,
        250,
    )

    var95_losses = np.full(
        250,
        var95.var,
        dtype=float,
    )

    var99_losses = np.full(
        250,
        var99.var,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Kupiec 95%
    # ---------------------------------------------------------------

    result95 = kupiec_95(
        test_returns,
        var95_losses,
    )

    print("\n--- Kupiec 95% ---")

    print(
        "CONFIDENCE LEVEL:",
        result95.confidence_level,
    )

    print(
        "EXPECTED EXCEPTION RATE:",
        result95.expected_exception_rate,
    )

    print(
        "OBSERVED EXCEPTION RATE:",
        result95.observed_exception_rate,
    )

    print(
        "OBSERVATION COUNT:",
        result95.observation_count,
    )

    print(
        "EXCEPTION COUNT:",
        result95.exception_count,
    )

    print(
        "LR STATISTIC:",
        result95.lr_statistic,
    )

    print(
        "P-VALUE:",
        result95.p_value,
    )

    print(
        "REJECT NULL:",
        result95.reject_null,
    )

    # ---------------------------------------------------------------
    # Kupiec 99%
    # ---------------------------------------------------------------

    result99 = kupiec_99(
        test_returns,
        var99_losses,
    )

    print("\n--- Kupiec 99% ---")

    print(
        "CONFIDENCE LEVEL:",
        result99.confidence_level,
    )

    print(
        "EXPECTED EXCEPTION RATE:",
        result99.expected_exception_rate,
    )

    print(
        "OBSERVED EXCEPTION RATE:",
        result99.observed_exception_rate,
    )

    print(
        "OBSERVATION COUNT:",
        result99.observation_count,
    )

    print(
        "EXCEPTION COUNT:",
        result99.exception_count,
    )

    print(
        "LR STATISTIC:",
        result99.lr_statistic,
    )

    print(
        "P-VALUE:",
        result99.p_value,
    )

    print(
        "REJECT NULL:",
        result99.reject_null,
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print("\n--- Kupiec Validation ---")

    assert result95.confidence_level == 0.95
    assert result99.confidence_level == 0.99

    assert np.isclose(
        result95.expected_exception_rate,
        0.05,
    )

    assert np.isclose(
        result99.expected_exception_rate,
        0.01,
    )

    assert (
        result95.observation_count
        == 250
    )

    assert (
        result99.observation_count
        == 250
    )

    assert (
        result95.exception_count
        >= 0
    )

    assert (
        result99.exception_count
        >= 0
    )

    assert (
        result95.exception_count
        <= result95.observation_count
    )

    assert (
        result99.exception_count
        <= result99.observation_count
    )

    assert np.isfinite(
        result95.lr_statistic
    )

    assert np.isfinite(
        result95.p_value
    )

    assert np.isfinite(
        result99.lr_statistic
    )

    assert np.isfinite(
        result99.p_value
    )

    assert (
        0.0
        <= result95.p_value
        <= 1.0
    )

    assert (
        0.0
        <= result99.p_value
        <= 1.0
    )

    assert isinstance(
        result95.reject_null,
        bool,
    )

    assert isinstance(
        result99.reject_null,
        bool,
    )

    print(
        "95% RESULT VALID: True"
    )

    print(
        "99% RESULT VALID: True"
    )

    print(
        "\nKUPIEC INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()