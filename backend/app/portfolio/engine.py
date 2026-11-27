from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.portfolio.portfolio import calculate_performance


class PortfolioExecutionError(ValueError):
    """Raised when a portfolio cannot be constructed from real forecasts."""


@dataclass(frozen=True)
class PortfolioExecution:
    positions: pd.DataFrame
    returns: pd.DataFrame
    metrics: dict[str, float | None]
    asset_count: int
    observation_count: int


def execute_portfolio(
    forecasts: pd.DataFrame,
    *,
    strategy: str,
    volatility_target: float,
    max_position: float,
    max_turnover: float,
    transaction_cost: float,
    covariance_window: int = 60,
    min_covariance_observations: int = 20,
) -> PortfolioExecution:
    """Execute a one-step-ahead portfolio from aligned OOS forecast records."""
    required = {"timestamp", "asset_id", "actual_return", "predicted_volatility"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise PortfolioExecutionError(
            "Forecast records are missing required fields: " + ", ".join(missing)
        )
    if forecasts.empty:
        raise PortfolioExecutionError("No forecast records were supplied.")
    if strategy not in {"static", "buy_and_hold", "risk_targeted"}:
        raise PortfolioExecutionError(
            "strategy must be one of: static, buy_and_hold, risk_targeted."
        )
    if not np.isfinite([volatility_target, max_position, max_turnover, transaction_cost]).all():
        raise PortfolioExecutionError("Portfolio parameters must be finite.")
    if volatility_target <= 0 or max_position <= 0 or max_turnover <= 0 or transaction_cost < 0:
        raise PortfolioExecutionError("Portfolio constraints and costs are outside valid ranges.")
    if covariance_window < min_covariance_observations or min_covariance_observations < 2:
        raise PortfolioExecutionError("Covariance window must be at least the minimum history.")

    frame = forecasts.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["asset_id"] = frame["asset_id"].astype(str).str.strip().str.upper()
    frame["actual_return"] = pd.to_numeric(frame["actual_return"], errors="coerce")
    frame["predicted_volatility"] = pd.to_numeric(
        frame["predicted_volatility"], errors="coerce"
    )
    numeric = frame[["actual_return", "predicted_volatility"]].to_numpy(dtype=float)
    if frame[["timestamp", "asset_id"]].isna().any().any() or not np.isfinite(numeric).all():
        raise PortfolioExecutionError("Forecast records contain invalid or non-finite values.")
    if (frame["predicted_volatility"] <= 0).any():
        raise PortfolioExecutionError("Predicted volatility must be strictly positive.")
    if (frame["actual_return"] <= -1).any():
        raise PortfolioExecutionError("Simple asset returns must be greater than -100%.")
    if frame.duplicated(["timestamp", "asset_id"]).any():
        raise PortfolioExecutionError("Duplicate timestamp/asset forecast records exist.")

    assets = sorted(frame["asset_id"].unique().tolist())
    dates = sorted(frame["timestamp"].unique().tolist())
    expected = {(date, asset) for date in dates for asset in assets}
    observed = set(zip(frame["timestamp"], frame["asset_id"], strict=True))
    if observed != expected:
        raise PortfolioExecutionError(
            "Forecast panel is incomplete. Every requested asset must have a forecast "
            "and realized return for every portfolio date."
        )
    frame = frame.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    returns_matrix = frame.pivot(index="timestamp", columns="asset_id", values="actual_return")
    forecast_matrix = frame.pivot(
        index="timestamp", columns="asset_id", values="predicted_volatility"
    )
    if returns_matrix.isna().any().any() or forecast_matrix.isna().any().any():
        raise PortfolioExecutionError("Forecast panel contains missing asset/date values.")

    count = len(assets)
    base = np.full(count, 1.0 / count, dtype=float)
    current = base.copy()
    previous = np.zeros(count, dtype=float)
    position_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    historical = returns_matrix.copy()

    for date_index, timestamp in enumerate(dates):
        if strategy == "static":
            proposed = base.copy()
        elif strategy == "buy_and_hold":
            proposed = current.copy()
        else:
            history = historical.loc[historical.index < timestamp].tail(covariance_window)
            if len(history) < min_covariance_observations:
                # Do not backfill risk estimates with future observations. The
                # strategy remains in cash until a strictly prior history exists.
                continue
            corr = history.corr().to_numpy(dtype=float)
            if not np.isfinite(corr).all():
                raise PortfolioExecutionError(
                    f"Historical correlation matrix is invalid at {timestamp}."
                )
            corr = (corr + corr.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
            corr = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
            diagonal = np.sqrt(np.diag(corr))
            corr = corr / np.outer(diagonal, diagonal)
            sigma = forecast_matrix.loc[timestamp, assets].to_numpy(dtype=float)
            covariance = corr * np.outer(sigma, sigma)
            estimated_risk = float(np.sqrt(max(base @ covariance @ base, 0.0)))
            if not np.isfinite(estimated_risk) or estimated_risk <= 0:
                raise PortfolioExecutionError(
                    f"Portfolio risk estimate is not positive at {timestamp}."
                )
            proposed = base * (volatility_target / estimated_risk)

        proposed = np.clip(proposed, -max_position, max_position)
        proposed_turnover = float(np.abs(proposed - previous).sum())
        if proposed_turnover > max_turnover and proposed_turnover > 0:
            proposed = previous + (proposed - previous) * (max_turnover / proposed_turnover)
        turnover = float(np.abs(proposed - previous).sum())
        day_returns = returns_matrix.loc[timestamp, assets].to_numpy(dtype=float)
        gross_return = float(proposed @ day_returns)
        cost = float(transaction_cost * turnover)
        net_return = gross_return - cost
        if not np.isfinite([gross_return, cost, net_return]).all() or net_return <= -1:
            raise PortfolioExecutionError(
                f"Portfolio return or transaction cost is invalid at {timestamp}."
            )

        for asset, weight in zip(assets, proposed, strict=True):
            position_rows.append(
                {"timestamp": timestamp, "asset_id": asset, "weight": float(weight)}
            )
        return_rows.append(
            {
                "timestamp": timestamp,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "gross_exposure": float(np.abs(proposed).sum()),
                "net_exposure": float(proposed.sum()),
                "predicted_volatility": (
                    float(np.sqrt(max(proposed @ covariance @ proposed, 0.0)))
                    if strategy == "risk_targeted"
                    else None
                ),
            }
        )

        if strategy == "buy_and_hold":
            growth = 1.0 + gross_return
            if growth <= 0:
                raise PortfolioExecutionError("Buy-and-hold portfolio wealth became non-positive.")
            current = proposed * (1.0 + day_returns) / growth
        else:
            current = proposed.copy()
        previous = proposed.copy()

    if not return_rows:
        raise PortfolioExecutionError(
            "No portfolio observations were executable. Risk targeting requires at least "
            f"{min_covariance_observations + 2} aligned dates to initialize and evaluate."
        )
    position_frame = pd.DataFrame(position_rows)
    return_frame = pd.DataFrame(return_rows)
    performance = calculate_performance(return_frame["net_return"])
    losses = -return_frame["net_return"].to_numpy(dtype=float)
    var95 = float(np.quantile(losses, 0.95))
    var99 = float(np.quantile(losses, 0.99))
    tail = losses[losses >= var95]
    metrics: dict[str, float | None] = {
        "total_return": performance.total_return,
        "annualized_return": performance.annualized_return,
        "annualized_volatility": performance.annualized_volatility,
        "sharpe_ratio": performance.sharpe_ratio,
        "downside_deviation": performance.downside_deviation,
        "sortino_ratio": performance.sortino_ratio,
        "max_drawdown": performance.max_drawdown,
        "calmar_ratio": performance.calmar_ratio,
        "var_95": var95,
        "var_99": var99,
        "es_95": float(np.mean(tail)),
        "total_turnover": float(return_frame["turnover"].sum()),
        "total_transaction_cost": float(return_frame["transaction_cost"].sum()),
        "gross_pnl": float(np.prod(1.0 + return_frame["gross_return"]) - 1.0),
        "net_pnl": float(np.prod(1.0 + return_frame["net_return"]) - 1.0),
    }
    return PortfolioExecution(
        positions=position_frame,
        returns=return_frame,
        metrics=metrics,
        asset_count=count,
        observation_count=len(return_frame),
    )
