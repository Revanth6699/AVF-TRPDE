from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter

from backend.app.risk.filtered_historical import (
    FilteredHistoricalSimulation,
)


def main() -> None:
    print(
        "\n"
        "============================================================\n"
        "PHASE 3 — FILTERED HISTORICAL SIMULATION INTEGRATION\n"
        "============================================================"
    )

    # ------------------------------------------------------------
    # Controlled research data
    # ------------------------------------------------------------

    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["asset_id"] = (
        df["asset_id"]
        .astype(str)
        .str.upper()
    )

    print(
        "\n--- INPUT DATA ---"
    )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "ASSETS:",
        df["asset_id"].unique().tolist(),
    )

    print(
        "TIMESTAMP RANGE:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    # ------------------------------------------------------------
    # Locked walk-forward configuration
    # ------------------------------------------------------------

    runner = WalkForwardRunner(
        WalkForwardSplitter(
            train_size=120,
            test_size=1,
            step_size=1,
            expanding=False,
        )
    )

    # ------------------------------------------------------------
    # Regime-XGBoost OOS forecasts
    # ------------------------------------------------------------

    result = runner.run_model(
        df,
        model="Regime-XGBoost",
        target_column="target",
        hmm_config=HMMConfig(
            n_components=3,
            random_state=42,
        ),
        feature_columns=(
            "return",
            "realized_volatility",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ),
    )

    predictions = result.predictions[
        [
            "timestamp",
            "asset_id",
            "prediction",
        ]
    ].copy()

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

    predictions["asset_id"] = (
        predictions["asset_id"]
        .astype(str)
        .str.upper()
    )

    predictions = predictions.rename(
        columns={
            "prediction": "forecast_volatility"
        }
    )

    print(
        "\n--- WALK-FORWARD FORECASTS ---"
    )

    print(
        "FOLDS:",
        result.fold_count,
    )

    print(
        "OOS FORECASTS:",
        len(predictions),
    )

    print(
        "FINITE FORECASTS:",
        np.isfinite(
            predictions[
                "forecast_volatility"
            ].to_numpy(dtype=float)
        ).all(),
    )

    print(
        "POSITIVE FORECASTS:",
        (
            predictions[
                "forecast_volatility"
            ].to_numpy(dtype=float)
            > 0
        ).all(),
    )

    # ------------------------------------------------------------
    # Align forecast at t with next-day return.
    #
    # The locked target is next-period realized volatility.
    # Therefore the risk observation associated with forecast t
    # is the return at t+1.
    # ------------------------------------------------------------

    return_data = df[
        [
            "timestamp",
            "asset_id",
            "return",
        ]
    ].copy()

    return_data["timestamp"] = pd.to_datetime(
        return_data["timestamp"]
    )

    return_data["asset_id"] = (
        return_data["asset_id"]
        .astype(str)
        .str.upper()
    )

    return_data = (
        return_data
        .sort_values(
            [
                "asset_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    return_data["risk_timestamp"] = (
        return_data
        .groupby("asset_id")[
            "timestamp"
        ]
        .shift(-1)
    )

    return_data["next_return"] = (
        return_data
        .groupby("asset_id")[
            "return"
        ]
        .shift(-1)
    )

    aligned = predictions.merge(
        return_data[
            [
                "timestamp",
                "asset_id",
                "risk_timestamp",
                "next_return",
            ]
        ],
        on=[
            "timestamp",
            "asset_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    aligned = aligned.dropna(
        subset=[
            "risk_timestamp",
            "next_return",
        ]
    ).reset_index(drop=True)

    print(
        "\n--- FORECAST / NEXT-RETURN ALIGNMENT ---"
    )

    print(
        "ALIGNED OBSERVATIONS:",
        len(aligned),
    )

    print(
        "EXPECTED OBSERVATIONS:",
        len(predictions) - 1,
    )

    print(
        "FINITE NEXT RETURNS:",
        np.isfinite(
            aligned[
                "next_return"
            ].to_numpy(dtype=float)
        ).all(),
    )

    print(
        "FINITE FORECAST VOLATILITY:",
        np.isfinite(
            aligned[
                "forecast_volatility"
            ].to_numpy(dtype=float)
        ).all(),
    )

    # ------------------------------------------------------------
    # Build FHS input.
    #
    # Each row contains:
    #   timestamp
    #   asset_id
    #   next-day return
    #   corresponding OOS volatility forecast
    # ------------------------------------------------------------

    fhs_input = aligned[
        [
            "risk_timestamp",
            "asset_id",
            "next_return",
            "forecast_volatility",
        ]
    ].copy()

    fhs_input = fhs_input.rename(
        columns={
            "risk_timestamp": "timestamp",
            "next_return": "return",
        }
    )

    fhs_input = (
        fhs_input
        .sort_values(
            [
                "asset_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # FHS
    # ------------------------------------------------------------

    fhs = FilteredHistoricalSimulation(
        min_observations=100,
        random_state=42,
    )

    standardized_returns = fhs.fit(
        fhs_input
    )

    print(
        "\n--- FHS STANDARDIZATION ---"
    )

    print(
        "STANDARDIZED OBSERVATIONS:",
        len(standardized_returns),
    )

    print(
        "FINITE STANDARDIZED RETURNS:",
        np.isfinite(
            standardized_returns
            .to_numpy(dtype=float)
        ).all(),
    )

    print(
        "STANDARDIZED MEAN:",
        float(
            standardized_returns.mean()
        ),
    )

    print(
        "STANDARDIZED STD:",
        float(
            standardized_returns.std()
        ),
    )

    # ------------------------------------------------------------
    # Current forecast for the next risk distribution
    # ------------------------------------------------------------

    current_forecast = float(
        aligned.iloc[-1][
            "forecast_volatility"
        ]
    )

    print(
        "\n--- CURRENT FORECAST ---"
    )

    print(
        "FORECAST VOLATILITY:",
        current_forecast,
    )

    # ------------------------------------------------------------
    # Simulated return distribution
    # ------------------------------------------------------------

    simulated_returns = fhs.simulate(
        standardized_returns=(
            standardized_returns
        ),
        forecast_volatility=(
            current_forecast
        ),
        n_simulations=10_000,
    )

    print(
        "\n--- FHS SIMULATION ---"
    )

    print(
        "SIMULATIONS:",
        len(simulated_returns),
    )

    print(
        "FINITE:",
        np.isfinite(
            simulated_returns
        ).all(),
    )

    print(
        "MEAN:",
        float(
            simulated_returns.mean()
        ),
    )

    print(
        "STD:",
        float(
            simulated_returns.std()
        ),
    )

    print(
        "MIN:",
        float(
            simulated_returns.min()
        ),
    )

    print(
        "MAX:",
        float(
            simulated_returns.max()
        ),
    )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    if len(aligned) < 100:
        raise AssertionError(
            "FHS integration did not produce "
            "the minimum required observations."
        )

    if not np.isfinite(
        standardized_returns
        .to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Standardized returns contain "
            "non-finite values."
        )

    if not np.isfinite(
        simulated_returns
    ).all():
        raise AssertionError(
            "FHS simulated returns contain "
            "non-finite values."
        )

    if current_forecast <= 0:
        raise AssertionError(
            "Current forecast volatility "
            "must be positive."
        )

    print(
        "\n============================================================\n"
        "FHS INTEGRATION STACK OK\n"
        "============================================================"
    )


if __name__ == "__main__":
    main()