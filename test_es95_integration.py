from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.risk.expected_shortfall import (
    calculate_es_95,
)
from backend.app.risk.filtered_historical import (
    FilteredHistoricalSimulation,
)


def main() -> None:
    rng = np.random.default_rng(42)

    observation_count = 500
    simulation_count = 10_000
    forecast_volatility = 0.012

    historical_returns = rng.normal(
        loc=0.0,
        scale=0.01,
        size=observation_count,
    )

    historical_volatility = rng.uniform(
        0.008,
        0.015,
        size=observation_count,
    )

    historical_data = pd.DataFrame(
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

    print(
        "\n========== "
        "PHASE 3 / ES95 INTEGRATION "
        "=========="
    )

    # --------------------------------------------------------------
    # 1. FHS
    # --------------------------------------------------------------

    fhs = FilteredHistoricalSimulation(
        min_observations=100,
        random_state=42,
    )

    standardized_returns = fhs.fit(
        historical_data
    )

    print(
        "\n--- FHS ---"
    )

    print(
        "HISTORICAL OBSERVATIONS:",
        observation_count,
    )

    print(
        "STANDARDIZED OBSERVATIONS:",
        len(standardized_returns),
    )

    print(
        "STANDARDIZED FINITE:",
        np.isfinite(
            standardized_returns.to_numpy(
                dtype=float
            )
        ).all(),
    )

    assert len(standardized_returns) == observation_count

    assert np.isfinite(
        standardized_returns.to_numpy(
            dtype=float
        )
    ).all()

    # --------------------------------------------------------------
    # 2. FHS simulation
    # --------------------------------------------------------------

    simulated_returns = fhs.simulate(
        standardized_returns=standardized_returns,
        forecast_volatility=forecast_volatility,
        n_simulations=simulation_count,
    )

    print(
        "\n--- FHS Simulation ---"
    )

    print(
        "SIMULATIONS:",
        len(simulated_returns),
    )

    print(
        "SIMULATED FINITE:",
        np.isfinite(
            simulated_returns
        ).all(),
    )

    print(
        "FORECAST VOLATILITY:",
        forecast_volatility,
    )

    assert len(simulated_returns) == simulation_count

    assert np.isfinite(
        simulated_returns
    ).all()

    # --------------------------------------------------------------
    # 3. ES95
    # --------------------------------------------------------------

    es95 = calculate_es_95(
        simulated_returns
    )

    print(
        "\n--- ES95 ---"
    )

    print(
        "CONFIDENCE LEVEL:",
        es95.confidence_level,
    )

    print(
        "VaR95:",
        es95.var,
    )

    print(
        "ES95:",
        es95.expected_shortfall,
    )

    print(
        "TAIL OBSERVATIONS:",
        es95.tail_observation_count,
    )

    # --------------------------------------------------------------
    # 4. ES95 validation
    # --------------------------------------------------------------

    simulated_values = np.asarray(
        simulated_returns,
        dtype=float,
    )

    quantile_95 = float(
        np.quantile(
            simulated_values,
            0.05,
        )
    )

    expected_var95 = -quantile_95

    tail_returns = simulated_values[
        simulated_values <= quantile_95
    ]

    expected_es95 = -float(
        np.mean(tail_returns)
    )

    print(
        "\n--- ES95 Validation ---"
    )

    print(
        "EXPECTED VaR95:",
        expected_var95,
    )

    print(
        "EXPECTED ES95:",
        expected_es95,
    )

    print(
        "VaR MATCH:",
        np.isclose(
            es95.var,
            expected_var95,
        ),
    )

    print(
        "ES95 MATCH:",
        np.isclose(
            es95.expected_shortfall,
            expected_es95,
        ),
    )

    print(
        "TAIL COUNT MATCH:",
        es95.tail_observation_count
        == len(tail_returns),
    )

    assert np.isclose(
        es95.var,
        expected_var95,
    )

    assert np.isclose(
        es95.expected_shortfall,
        expected_es95,
    )

    assert (
        es95.tail_observation_count
        == len(tail_returns)
    )

    # ES must be at least as large as VaR
    # when both are expressed as positive
    # loss magnitudes.
    print(
        "ES95 >= VaR95:",
        es95.expected_shortfall
        >= es95.var,
    )

    assert (
        es95.expected_shortfall
        >= es95.var
    )

    # --------------------------------------------------------------
    # 5. Result validity
    # --------------------------------------------------------------

    print(
        "\n--- Result Validation ---"
    )

    print(
        "ES95 FINITE:",
        np.isfinite(
            es95.expected_shortfall
        ),
    )

    print(
        "VaR95 FINITE:",
        np.isfinite(
            es95.var
        ),
    )

    print(
        "TAIL COUNT POSITIVE:",
        es95.tail_observation_count > 0,
    )

    print(
        "ES95 POSITIVE:",
        es95.expected_shortfall > 0,
    )

    assert np.isfinite(
        es95.expected_shortfall
    )

    assert np.isfinite(
        es95.var
    )

    assert (
        es95.tail_observation_count > 0
    )

    assert (
        es95.expected_shortfall > 0
    )

    print(
        "\nES95 INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()