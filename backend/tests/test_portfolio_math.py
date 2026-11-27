from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.portfolio.constraints import (
    apply_turnover_limit,
    calculate_turnover,
)
from backend.app.portfolio.costs import build_cost_analysis
from backend.app.portfolio.portfolio import (
    PortfolioError,
    calculate_performance,
    calculate_portfolio_returns,
    create_buy_and_hold_weights,
)


def test_performance_uses_arithmetic_excess_returns_and_initial_capital_drawdown() -> None:
    returns = np.array([-0.10, 0.20, 0.05])
    result = calculate_performance(
        pd.Series(returns),
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    expected_total = float(np.prod(1.0 + returns) - 1.0)
    expected_annualized = float(np.prod(1.0 + returns) ** (252 / 3) - 1.0)
    expected_volatility = float(np.std(returns, ddof=1) * np.sqrt(252))
    expected_sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
    downside = np.minimum(returns, 0.0)
    periodic_downside = float(np.sqrt(np.mean(np.square(downside))))
    expected_sortino = float(np.mean(returns) / periodic_downside * np.sqrt(252))

    assert result.total_return == pytest.approx(expected_total)
    assert result.annualized_return == pytest.approx(expected_annualized)
    assert result.annualized_volatility == pytest.approx(expected_volatility)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe)
    assert result.downside_deviation == pytest.approx(periodic_downside * np.sqrt(252))
    assert result.sortino_ratio == pytest.approx(expected_sortino)
    assert result.max_drawdown == pytest.approx(-0.10)
    assert result.calmar_ratio == pytest.approx(expected_annualized / 0.10)


def test_performance_rejects_single_observation() -> None:
    with pytest.raises(PortfolioError, match="At least two"):
        calculate_performance(pd.Series([0.01]))


def test_portfolio_returns_apply_pre_return_weights_to_simple_returns() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset_id": ["AAA", "BBB", "AAA", "BBB"],
            "return": [0.01, 0.02, -0.01, 0.03],
        }
    )
    weights = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset_id": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.6, 0.4, 0.6, 0.4],
        }
    )

    result = calculate_portfolio_returns(returns, weights)
    np.testing.assert_allclose(
        result.portfolio_returns["portfolio_return"].to_numpy(),
        [0.014, 0.006],
    )
    np.testing.assert_allclose(
        result.portfolio_returns["gross_exposure"].to_numpy(),
        [1.0, 1.0],
    )


def test_undefined_zero_denominator_ratios_are_not_reported_as_zero() -> None:
    result = calculate_performance(pd.Series([0.01, 0.01, 0.01]))
    assert result.sharpe_ratio is None
    assert result.sortino_ratio is None
    assert result.calmar_ratio is None


def test_portfolio_return_calculation_rejects_partial_asset_panels() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "asset_id": ["AAA", "BBB"],
            "return": [0.01, -0.02],
        }
    )
    weights = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "asset_id": ["AAA"],
            "weight": [1.0],
        }
    )

    with pytest.raises(PortfolioError, match="same timestamp/asset keys"):
        calculate_portfolio_returns(returns, weights)


def test_buy_and_hold_weights_drift_without_rebalancing() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "asset_id": ["AAA", "BBB", "AAA", "BBB"],
            "return": [0.10, 0.00, 0.00, 0.00],
        }
    )

    weights = create_buy_and_hold_weights(
        returns,
        {"AAA": 0.5, "BBB": 0.5},
    )
    day_one = weights.loc[weights["timestamp"] == pd.Timestamp("2024-01-01")]
    day_two = weights.loc[weights["timestamp"] == pd.Timestamp("2024-01-02")]

    assert day_one.set_index("asset_id")["weight"].to_dict() == {
        "AAA": pytest.approx(0.5),
        "BBB": pytest.approx(0.5),
    }
    assert day_two.set_index("asset_id")["weight"]["AAA"] == pytest.approx(0.55 / 1.05)
    assert day_two.set_index("asset_id")["weight"]["BBB"] == pytest.approx(0.50 / 1.05)


def test_turnover_limit_counts_full_liquidations_and_new_positions() -> None:
    current = pd.DataFrame(
        {
            "timestamp": ["2024-01-02"],
            "asset_id": ["BBB"],
            "target_weight": [0.5],
        }
    )
    previous = pd.DataFrame(
        {
            "asset_id": ["AAA"],
            "target_weight": [0.5],
        }
    )

    constrained = apply_turnover_limit(
        current,
        previous,
        max_turnover=0.2,
    )
    turnover = calculate_turnover(constrained, previous)

    weights = constrained.set_index("asset_id")["target_weight"].to_dict()
    assert weights["AAA"] == pytest.approx(0.4)
    assert weights["BBB"] == pytest.approx(0.1)
    assert turnover.turnover == pytest.approx(0.2)


def test_cost_analysis_compounds_gross_and_net_returns() -> None:
    portfolio_returns = pd.DataFrame(
        {
            "gross_return": [0.10, 0.10],
            "turnover": [0.20, 0.20],
        }
    )
    result = build_cost_analysis(
        portfolio_returns,
        cost_rate=0.01,
    )

    expected_gross = (1.10 * 1.10) - 1.0
    expected_net = (1.098 * 1.098) - 1.0
    assert result.total_gross_return == pytest.approx(expected_gross)
    assert result.total_net_return == pytest.approx(expected_net)
    assert result.total_transaction_cost == pytest.approx(0.004)
