from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.app.research.models.ewma import EWMAVolatility
from backend.app.research.walk_forward.runner import (
    WalkForwardRunner,
    _update_gjr_variance,
)


def test_ewma_forecast_uses_observed_test_day_return_for_next_day_target() -> None:
    train_returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    test_return = 0.04
    decay = 0.94

    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(train_returns), freq="B"),
            "asset_id": ["AAPL"] * len(train_returns),
            "return": train_returns,
            "target": [0.02] * len(train_returns),
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-08")],
            "asset_id": ["AAPL"],
            "return": [test_return],
            "target": [0.03],
        }
    )

    result = WalkForwardRunner._predict_ewma(
        train_data=train,
        test_data=test,
        fold=SimpleNamespace(fold_id=1),
        decay=decay,
    )

    expected = EWMAVolatility(decay=decay).fit(
        pd.Series(train_returns + [test_return], dtype=float)
    ).forecast_volatility
    train_only = EWMAVolatility(decay=decay).fit(
        pd.Series(train_returns, dtype=float)
    ).forecast_volatility

    assert np.isclose(result.loc[0, "prediction"], expected)
    assert not np.isclose(result.loc[0, "prediction"], train_only)


def test_gjr_filter_updates_with_observed_return_without_refitting_parameters() -> None:
    fitted_parameters = SimpleNamespace(
        mu=0.001,
        omega=0.00001,
        alpha=0.08,
        gamma=0.12,
        beta=0.82,
    )
    previous_variance = 0.0004
    observed_return = -0.03
    residual = observed_return - fitted_parameters.mu
    expected = (
        fitted_parameters.omega
        + fitted_parameters.alpha * residual**2
        + fitted_parameters.gamma * residual**2
        + fitted_parameters.beta * previous_variance
    )

    actual = _update_gjr_variance(
        fitted_parameters,
        previous_variance,
        observed_return,
    )

    assert np.isclose(actual, expected)
