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


TARGET_VOLATILITY = 0.01
MAX_POSITION = 0.80
MAX_TURNOVER = 0.20
COST_RATE = 0.0004
COST_RATES = (
    0.0,
    0.0004,
    0.0010,
    0.0020,
)

BASE_WEIGHT = 1.0


def _prepare_data() -> pd.DataFrame:
    data = make_controlled_data().copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data["asset_id"] = (
        data["asset_id"]
        .astype(str)
        .str.upper()
    )

    return data


def _build_forecasts(
    data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    runner = WalkForwardRunner(
        WalkForwardSplitter(
            train_size=120,
            test_size=1,
            step_size=1,
            expanding=False,
        )
    )

    forecasts: dict[str, pd.DataFrame] = {}

    print(
        "\n========== "
        "GENERATING WALK-FORWARD FORECASTS "
        "=========="
    )

    ewma = runner.run_model(
        data,
        model="EWMA",
        target_column="target",
        ewma_decay=0.94,
    )

    forecasts["EWMA"] = (
        ewma.predictions[
            [
                "timestamp",
                "asset_id",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": "forecast_volatility",
            }
        )
        .copy()
    )

    print(
        "EWMA:",
        len(forecasts["EWMA"]),
        "forecasts",
    )

    gjr = runner.run_model(
        data,
        model="GJR-GARCH",
        target_column="target",
    )

    forecasts["GJR-GARCH"] = (
        gjr.predictions[
            [
                "timestamp",
                "asset_id",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": "forecast_volatility",
            }
        )
        .copy()
    )

    print(
        "GJR-GARCH:",
        len(forecasts["GJR-GARCH"]),
        "forecasts",
    )

    xgb = runner.run_model(
        data,
        model="XGBoost",
        target_column="target",
    )

    forecasts["XGBoost"] = (
        xgb.predictions[
            [
                "timestamp",
                "asset_id",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": "forecast_volatility",
            }
        )
        .copy()
    )

    print(
        "XGBoost:",
        len(forecasts["XGBoost"]),
        "forecasts",
    )

    regime_xgb = runner.run_model(
        data,
        model="Regime-XGBoost",
        target_column="target",
        hmm_config=HMMConfig(
            n_components=3,
            random_state=42,
        ),
    )

    forecasts["Regime-XGBoost"] = (
        regime_xgb.predictions[
            [
                "timestamp",
                "asset_id",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": "forecast_volatility",
            }
        )
        .copy()
    )

    print(
        "Regime-XGBoost:",
        len(forecasts["Regime-XGBoost"]),
        "forecasts",
    )

    for name, frame in forecasts.items():
        frame["base_weight"] = BASE_WEIGHT

        values = frame[
            "forecast_volatility"
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise AssertionError(
                f"{name}: non-finite forecasts."
            )

        if (values <= 0).any():
            raise AssertionError(
                f"{name}: forecast volatility must "
                "be greater than zero."
            )

    return forecasts


def _align_weights_to_next_return(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align forecast-time weights to the next available
    return observation.

    The forecast at time t targets next-day volatility,
    so the portfolio weight generated at t is applied
    to the next return observation.
    """

    weight_data = weights.copy()
    return_data = returns.copy()

    weight_data["timestamp"] = pd.to_datetime(
        weight_data["timestamp"]
    )

    return_data["timestamp"] = pd.to_datetime(
        return_data["timestamp"]
    )

    weight_data["asset_id"] = (
        weight_data["asset_id"]
        .astype(str)
        .str.upper()
    )

    return_data["asset_id"] = (
        return_data["asset_id"]
        .astype(str)
        .str.upper()
    )

    return_data = (
        return_data
        .sort_values(
            ["asset_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []

    for _, row in weight_data.iterrows():
        future_returns = return_data.loc[
            (
                return_data["asset_id"]
                == row["asset_id"]
            )
            & (
                return_data["timestamp"]
                > row["timestamp"]
            )
        ]

        if future_returns.empty:
            continue

        next_return = future_returns.iloc[0]

        rows.append(
            {
                "timestamp": next_return[
                    "timestamp"
                ],
                "asset_id": row["asset_id"],
                "weight": float(
                    row["target_weight"]
                ),
            }
        )

    if not rows:
        raise AssertionError(
            "No forecast-to-next-return "
            "observations could be aligned."
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["timestamp", "asset_id"]
        )
        .reset_index(drop=True)
    )


def _apply_constraints_sequentially(
    targeted: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    constrained_rows: list[dict[str, object]] = []
    turnovers: list[float] = []

    previous_weights: pd.DataFrame | None = None

    for _, row in targeted.iterrows():
        current = pd.DataFrame(
            {
                "timestamp": [
                    row["timestamp"]
                ],
                "asset_id": [
                    row["asset_id"]
                ],
                "target_weight": [
                    row["target_weight"]
                ],
            }
        )

        result = apply_constraints(
            current,
            max_position=MAX_POSITION,
            previous_weights=previous_weights,
            max_turnover=MAX_TURNOVER,
        )

        turnovers.append(
            float(result.total_turnover)
        )

        constrained_rows.append(
            result.weights.iloc[0].to_dict()
        )

        previous_weights = result.weights[
            [
                "asset_id",
                "target_weight",
            ]
        ].copy()

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

    return constrained, turnovers


def _calculate_strategy(
    name: str,
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, object]:
    targeted = calculate_risk_targeted_weights(
        forecasts,
        target_volatility=TARGET_VOLATILITY,
        max_position=MAX_POSITION,
    )

    raw_values = targeted[
        "target_weight"
    ].to_numpy(dtype=float)

    if not np.isfinite(raw_values).all():
        raise AssertionError(
            f"{name}: risk-targeted weights "
            "are not finite."
        )

    if (
        np.abs(raw_values)
        > MAX_POSITION + 1e-12
    ).any():
        raise AssertionError(
            f"{name}: position limit violated "
            "during risk targeting."
        )

    constrained, turnovers = (
        _apply_constraints_sequentially(
            targeted
        )
    )

    validation_frame = constrained.rename(
        columns={
            "weight": "target_weight"
        }
    )

    constraints_valid = validate_constraints(
        validation_frame,
        max_position=MAX_POSITION,
    )

    if not constraints_valid:
        raise AssertionError(
            f"{name}: final position "
            "constraints failed."
        )

    weight_values = constrained[
        "weight"
    ].to_numpy(dtype=float)

    if not np.isfinite(weight_values).all():
        raise AssertionError(
            f"{name}: constrained weights "
            "are not finite."
        )

    if (
        np.abs(weight_values)
        > MAX_POSITION + 1e-12
    ).any():
        raise AssertionError(
            f"{name}: final position limit "
            "violated."
        )

    turnover_values = np.asarray(
        turnovers,
        dtype=float,
    )

    if not np.isfinite(
        turnover_values
    ).all():
        raise AssertionError(
            f"{name}: turnover contains "
            "non-finite values."
        )

    if (
        turnover_values
        > MAX_TURNOVER + 1e-12
    ).any():
        raise AssertionError(
            f"{name}: turnover limit violated."
        )

    portfolio = calculate_portfolio_returns(
        returns,
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

    if portfolio.portfolio_returns.empty:
        raise AssertionError(
            f"{name}: empty portfolio return series."
        )

    gross_returns = (
        portfolio.portfolio_returns[
            [
                "timestamp",
                "portfolio_return",
            ]
        ]
        .copy()
        .rename(
            columns={
                "portfolio_return":
                    "gross_return"
            }
        )
    )

    turnover_by_timestamp = (
        constrained
        .assign(
            turnover=np.asarray(
                turnovers,
                dtype=float,
            )
        )
        .groupby(
            "timestamp",
            as_index=False,
        )["turnover"]
        .sum()
    )

    gross_returns = gross_returns.merge(
        turnover_by_timestamp,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    gross_returns["turnover"] = (
        gross_returns["turnover"]
        .fillna(0.0)
    )

    performance = calculate_performance(
        portfolio.portfolio_returns
    )

    net = calculate_net_returns(
        gross_returns,
        cost_rate=COST_RATE,
    )

    cost_analysis = build_cost_analysis(
        gross_returns,
        cost_rate=COST_RATE,
    )

    sensitivity = calculate_cost_sensitivity(
        gross_returns,
        cost_rates=COST_RATES,
    )

    return {
        "name": name,
        "targeted": targeted,
        "constrained": constrained,
        "turnovers": turnovers,
        "portfolio": portfolio,
        "performance": performance,
        "gross_returns": gross_returns,
        "net": net,
        "cost_analysis": cost_analysis,
        "sensitivity": sensitivity,
    }


def main() -> None:
    print(
        "\n========== "
        "PHASE 4 PORTFOLIO RESEARCH TEST "
        "=========="
    )

    data = _prepare_data()

    returns = data[
        [
            "timestamp",
            "asset_id",
            "return",
        ]
    ].copy()

    forecasts = _build_forecasts(
        data
    )

    results: dict[
        str,
        dict[str, object],
    ] = {}

    for name, forecast_data in (
        forecasts.items()
    ):
        print(
            f"\n---------- "
            f"{name} PORTFOLIO "
            f"----------"
        )

        result = _calculate_strategy(
            name,
            forecast_data,
            returns,
        )

        results[name] = result

        performance = result[
            "performance"
        ]

        cost_analysis = result[
            "cost_analysis"
        ]

        sensitivity = result[
            "sensitivity"
        ]

        print(
            "RISK-TARGETED WEIGHTS:",
            len(result["targeted"]),
        )

        print(
            "CONSTRAINED WEIGHTS:",
            len(result["constrained"]),
        )

        print(
            "MAX ABS WEIGHT:",
            float(
                result[
                    "constrained"
                ]["weight"]
                .abs()
                .max()
            ),
        )

        print(
            "MAX TURNOVER:",
            float(
                max(
                    result["turnovers"]
                )
            ),
        )

        print(
            "PORTFOLIO OBSERVATIONS:",
            result[
                "portfolio"
            ].observation_count,
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
            "TOTAL TURNOVER:",
            cost_analysis.total_turnover,
        )

        print(
            "TRANSACTION COST:",
            cost_analysis.total_transaction_cost,
        )

        print(
            "GROSS RETURN:",
            cost_analysis.total_gross_return,
        )

        print(
            "NET RETURN:",
            cost_analysis.total_net_return,
        )

        sensitivity_rates = (
            sensitivity[
                "cost_rate"
            ].to_numpy(dtype=float)
        )

        sensitivity_net = (
            sensitivity[
                "total_net_return"
            ].to_numpy(dtype=float)
        )

        if not np.isfinite(
            sensitivity_net
        ).all():
            raise AssertionError(
                f"{name}: cost sensitivity "
                "contains non-finite values."
            )

        if not np.all(
            np.diff(
                sensitivity_net
            ) <= 1e-12
        ):
            raise AssertionError(
                f"{name}: net return did not "
                "decrease as transaction cost "
                "increased."
            )

        if not np.array_equal(
            sensitivity_rates,
            np.asarray(
                COST_RATES,
                dtype=float,
            ),
        ):
            raise AssertionError(
                f"{name}: cost-rate sensitivity "
                "grid mismatch."
            )

        print(
            "COST SENSITIVITY:"
        )

        print(
            sensitivity.to_string(
                index=False
            )
        )

        print(
            f"{name}: PASS"
        )

    print(
        "\n========== "
        "PHASE 4 SUMMARY "
        "=========="
    )

    print(
        "STRATEGIES TESTED:",
        tuple(results.keys()),
    )

    print(
        "RISK TARGETING: PASS"
    )

    print(
        "POSITION CONSTRAINTS: PASS"
    )

    print(
        "TURNOVER CONSTRAINTS: PASS"
    )

    print(
        "PORTFOLIO RETURNS: PASS"
    )

    print(
        "PERFORMANCE ANALYSIS: PASS"
    )

    print(
        "TRANSACTION COSTS: PASS"
    )

    print(
        "COST SENSITIVITY: PASS"
    )

    print(
        "\nPHASE 4 PORTFOLIO "
        "RESEARCH TEST: PASS"
    )


if __name__ == "__main__":
    main()