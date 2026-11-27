from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter
from backend.app.portfolio.risk_targeting import (
    calculate_risk_targeted_weights,
)
from backend.app.portfolio.constraints import (
    apply_constraints,
    validate_constraints,
)
from backend.app.portfolio.portfolio import (
    calculate_portfolio_returns,
    calculate_performance,
)
from backend.app.portfolio.costs import (
    calculate_net_returns,
    build_cost_analysis,
    calculate_cost_sensitivity,
)


def main() -> None:
    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset_id"] = (
        df["asset_id"]
        .astype(str)
        .str.upper()
    )

    runner = WalkForwardRunner(
        WalkForwardSplitter(
            train_size=120,
            test_size=1,
            step_size=1,
            expanding=False,
        )
    )

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

    predictions["base_weight"] = 1.0

    weights = calculate_risk_targeted_weights(
        predictions,
        target_volatility=0.01,
        max_position=0.80,
    )

    # Forecast at t targets the next available
    # return observation.
    return_index = (
        df[
            [
                "timestamp",
                "asset_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "asset_id",
                "timestamp",
            ]
        )
        .copy()
    )

    return_index["execution_timestamp"] = (
        return_index
        .groupby("asset_id")["timestamp"]
        .shift(-1)
    )

    weights = weights.merge(
        return_index,
        on=[
            "timestamp",
            "asset_id",
        ],
        how="left",
        validate="one_to_one",
    )

    weights = (
        weights
        .dropna(
            subset=["execution_timestamp"]
        )
        .reset_index(drop=True)
    )

    previous = None
    constrained_rows = []
    turnovers = []

    for _, row in weights.iterrows():
        current = pd.DataFrame(
            {
                "timestamp": [
                    row["execution_timestamp"]
                ],
                "asset_id": [
                    row["asset_id"]
                ],
                "target_weight": [
                    row["target_weight"]
                ],
            }
        )

        constrained_result = apply_constraints(
            current,
            max_position=0.80,
            previous_weights=previous,
            max_turnover=0.20,
        )

        constrained_rows.append(
            constrained_result.weights.iloc[0].to_dict()
        )

        turnovers.append(
            float(
                constrained_result.total_turnover
            )
        )

        previous = (
            constrained_result.weights[
                [
                    "asset_id",
                    "target_weight",
                ]
            ].copy()
        )

    constrained = pd.DataFrame(
        constrained_rows
    )

    constrained["timestamp"] = pd.to_datetime(
        constrained["timestamp"]
    )

    constrained["asset_id"] = (
        constrained["asset_id"]
        .astype(str)
        .str.upper()
    )

    constrained = constrained.rename(
        columns={
            "target_weight": "weight"
        }
    )

    portfolio = calculate_portfolio_returns(
        df[
            [
                "timestamp",
                "asset_id",
                "return",
            ]
        ],
        constrained[
            [
                "timestamp",
                "asset_id",
                "weight",
            ]
        ],
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    performance = calculate_performance(
        portfolio.portfolio_returns
    )

    portfolio_returns = (
        portfolio.portfolio_returns[
            [
                "timestamp",
                "portfolio_return",
            ]
        ].copy()
    )

    portfolio_returns["turnover"] = np.asarray(
        turnovers[:len(portfolio_returns)],
        dtype=float,
    )

    portfolio_returns = portfolio_returns.rename(
        columns={
            "portfolio_return": "gross_return"
        }
    )

    net_returns = calculate_net_returns(
        portfolio_returns,
        cost_rate=0.0004,
    )

    cost_analysis = build_cost_analysis(
        portfolio_returns,
        cost_rate=0.0004,
    )

    sensitivity = calculate_cost_sensitivity(
        portfolio_returns,
        cost_rates=(
            0.0,
            0.0004,
            0.0010,
        ),
    )

    print(
        "\n========== "
        "PHASE 1 / PORTFOLIO INTEGRATION "
        "=========="
    )

    print(
        "OOS FORECASTS:",
        len(result.predictions),
    )

    print(
        "RISK-TARGETED WEIGHTS:",
        len(weights),
    )

    print(
        "WEIGHT FINITE:",
        np.isfinite(
            weights[
                "target_weight"
            ].to_numpy(dtype=float)
        ).all(),
    )

    print(
        "MAX ABS RAW WEIGHT:",
        float(
            weights[
                "target_weight"
            ].abs().max()
        ),
    )

    print(
        "CONSTRAINED WEIGHTS:",
        len(constrained),
    )

    print(
        "CONSTRAINTS VALID:",
        validate_constraints(
            constrained.rename(
                columns={
                    "weight": "target_weight"
                }
            ),
            max_position=0.80,
        ),
    )

    print(
        "MAX ABS FINAL WEIGHT:",
        float(
            constrained[
                "weight"
            ].abs().max()
        ),
    )

    print(
        "PORTFOLIO OBSERVATIONS:",
        portfolio.observation_count,
    )

    print(
        "TOTAL RETURN:",
        performance.total_return,
    )

    print(
        "ANNUALIZED RETURN:",
        performance.annualized_return,
    )

    print(
        "ANNUALIZED VOLATILITY:",
        performance.annualized_volatility,
    )

    print(
        "SHARPE:",
        performance.sharpe_ratio,
    )

    print(
        "SORTINO:",
        performance.sortino_ratio,
    )

    print(
        "MAX DRAWDOWN:",
        performance.max_drawdown,
    )

    print(
        "CALMAR:",
        performance.calmar_ratio,
    )

    print(
        "GROSS RETURN:",
        cost_analysis.total_gross_return,
    )

    print(
        "TRANSACTION COSTS:",
        cost_analysis.total_transaction_cost,
    )

    print(
        "NET RETURN:",
        cost_analysis.total_net_return,
    )

    print(
        "NET FINITE:",
        np.isfinite(
            net_returns[
                "net_return"
            ].to_numpy(dtype=float)
        ).all(),
    )

    print("\nCOST SENSITIVITY:")
    print(
        sensitivity.to_string(
            index=False
        )
    )

    print(
        "\nPORTFOLIO INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()