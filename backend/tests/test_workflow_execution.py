from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.portfolio.engine import PortfolioExecutionError, execute_portfolio
from backend.app.stress.runner import run_stress_scenario


def _forecast_panel(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for index, timestamp in enumerate(dates):
        for asset, scale in (("AAA", 1.0), ("BBB", 1.5)):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset_id": asset,
                    "actual_return": float(np.sin(index * 0.7 + scale) * 0.01),
                    "predicted_volatility": 0.012 * scale,
                }
            )
    return pd.DataFrame(rows)


def test_static_portfolio_executes_and_compounds_realized_returns() -> None:
    frame = _forecast_panel()
    result = execute_portfolio(
        frame,
        strategy="static",
        volatility_target=0.01,
        max_position=0.8,
        max_turnover=2.0,
        transaction_cost=0.0,
    )
    expected_first = frame.loc[frame.timestamp == frame.timestamp.min(), "actual_return"].mean()
    assert result.observation_count == 30
    assert result.asset_count == 2
    assert result.returns.iloc[0].gross_return == pytest.approx(expected_first)
    assert result.metrics["total_return"] == pytest.approx(
        float(np.prod(1.0 + result.returns.net_return) - 1.0)
    )


def test_buy_and_hold_weights_drift_after_realized_returns() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": timestamp, "asset_id": asset, "actual_return": ret, "predicted_volatility": 0.01}
            for timestamp, values in [
                (pd.Timestamp("2024-01-01"), {"AAA": 0.10, "BBB": 0.00}),
                (pd.Timestamp("2024-01-02"), {"AAA": 0.00, "BBB": 0.00}),
            ]
            for asset, ret in values.items()
        ]
    )
    result = execute_portfolio(
        frame,
        strategy="buy_and_hold",
        volatility_target=0.01,
        max_position=0.8,
        max_turnover=2.0,
        transaction_cost=0.0,
    )
    second_day = result.positions.loc[
        result.positions.timestamp == pd.Timestamp("2024-01-02")
    ].set_index("asset_id")["weight"]
    assert second_day["AAA"] == pytest.approx(0.55 / 1.05)
    assert second_day["BBB"] == pytest.approx(0.50 / 1.05)


def test_risk_targeting_uses_only_prior_returns_and_cash_warmup() -> None:
    frame = _forecast_panel(35)
    result = execute_portfolio(
        frame,
        strategy="risk_targeted",
        volatility_target=0.01,
        max_position=0.8,
        max_turnover=2.0,
        transaction_cost=0.0,
        covariance_window=25,
        min_covariance_observations=20,
    )
    assert result.observation_count == 15
    assert result.returns.timestamp.min() == pd.Timestamp("2024-01-29")
    assert result.returns.predicted_volatility.notna().all()


def test_portfolio_rejects_incomplete_asset_date_panel() -> None:
    frame = _forecast_panel()
    frame = frame.drop(frame.index[-1])
    with pytest.raises(PortfolioExecutionError, match="panel is incomplete"):
        execute_portfolio(
            frame,
            strategy="static",
            volatility_target=0.01,
            max_position=0.8,
            max_turnover=2.0,
            transaction_cost=0.0,
        )


def test_stress_runner_compounds_total_return_and_includes_initial_capital_drawdown() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="B"),
            "asset_id": ["PORTFOLIO", "PORTFOLIO"],
            "return": [-0.10, 0.20],
        }
    )
    result = run_stress_scenario(frame, scenario_name="test")
    assert result.total_return == pytest.approx(0.08)
    assert result.cumulative_return == pytest.approx(0.08)
    assert result.max_drawdown == pytest.approx(-0.10)


def test_stress_runner_rejects_unaggregated_multi_asset_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "asset_id": ["AAA", "BBB"],
            "return": [0.01, -0.02],
        }
    )
    with pytest.raises(ValueError, match="expects one portfolio/asset"):
        run_stress_scenario(frame, scenario_name="pooled")


def test_stress_analysis_accepts_empty_scenario_without_zero_filling() -> None:
    from backend.app.stress.analysis import analyze_stress_results

    results = pd.DataFrame(
        [
            {
                "scenario": "normal_volatility",
                "observation_count": 3,
                "asset_count": 1,
                "total_return": 0.02,
                "mean_return": 0.006,
                "volatility": 0.1,
                "downside_deviation": 0.02,
                "max_drawdown": -0.01,
                "cumulative_return": 0.02,
            },
            {
                "scenario": "prolonged_drawdown",
                "observation_count": 0,
                "asset_count": 0,
                "total_return": np.nan,
                "mean_return": np.nan,
                "volatility": np.nan,
                "downside_deviation": np.nan,
                "max_drawdown": np.nan,
                "cumulative_return": np.nan,
            },
        ]
    )
    analysis = analyze_stress_results(results)
    assert analysis.scenario_count == 2
    assert analysis.valid_scenario_count == 1
