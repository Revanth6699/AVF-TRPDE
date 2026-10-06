from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.risk.filtered_historical import (
    FilteredHistoricalSimulation,
)
from backend.app.risk.var import (
    calculate_required_var_levels,
)


def main() -> None:
    print("\n========== STEP 11: VaR INTEGRATION ==========")

    rng = np.random.default_rng(42)

    observation_count = 500
    simulation_count = 10000

    historical_returns = rng.normal(
        0.0,
        0.01,
        observation_count,
    )

    historical_volatility = rng.uniform(
        0.008,
        0.015,
        observation_count,
    )

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=observation_count,
                freq="D",
            ),
            "asset_id": ["AAPL"] * observation_count,
            "return": historical_returns,
            "forecast_volatility": historical_volatility,
        }
    )

    print("\n---------- FHS ----------")

    fhs = FilteredHistoricalSimulation(
        min_observations=100,
        random_state=42,
    )

    standardized_returns = fhs.fit(
        dataframe
    )

    print(
        "STANDARDIZED OBSERVATIONS:",
        len(standardized_returns),
    )

    if len(standardized_returns) != observation_count:
        raise AssertionError(
            "Unexpected number of standardized FHS observations."
        )

    current_volatility = 0.012

    simulated_returns = fhs.simulate(
        standardized_returns,
        current_volatility,
        n_simulations=simulation_count,
    )

    print(
        "SIMULATED RETURNS:",
        len(simulated_returns),
    )

    if len(simulated_returns) != simulation_count:
        raise AssertionError(
            "Unexpected number of FHS simulations."
        )

    if not np.isfinite(
        simulated_returns
    ).all():
        raise AssertionError(
            "FHS simulations contain non-finite values."
        )

    print(
        "SIMULATIONS FINITE:",
        np.isfinite(simulated_returns).all(),
    )

    print("\n---------- VaR95 / VaR99 ----------")

    levels = calculate_required_var_levels(
        simulated_returns
    )

    var95 = levels.var_95
    var99 = levels.var_99

    print(
        "VaR95:",
        var95,
    )

    print(
        "VaR99:",
        var99,
    )

    if var95.confidence_level != 0.95:
        raise AssertionError(
            "VaR95 confidence level is incorrect."
        )

    if var99.confidence_level != 0.99:
        raise AssertionError(
            "VaR99 confidence level is incorrect."
        )

    if not np.isfinite(var95.var):
        raise AssertionError(
            "VaR95 is non-finite."
        )

    if not np.isfinite(var99.var):
        raise AssertionError(
            "VaR99 is non-finite."
        )

    if var95.var <= 0.0:
        raise AssertionError(
            "VaR95 must be positive."
        )

    if var99.var <= 0.0:
        raise AssertionError(
            "VaR99 must be positive."
        )

    if var99.var < var95.var:
        raise AssertionError(
            "VaR99 must not be below VaR95."
        )

    print(
        "VaR95 FINITE: True"
    )

    print(
        "VaR99 FINITE: True"
    )

    print(
        "VaR99 >= VaR95: True"
    )

    print(
        "\n========== STEP 11 RESULT =========="
    )

    print(
        "FHS → VaR95 → VaR99: PASS"
    )

    print(
        "STEP 11 VaR INTEGRATION OK"
    )


if __name__ == "__main__":
    main()