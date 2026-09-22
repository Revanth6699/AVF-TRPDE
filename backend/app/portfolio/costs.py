from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class TransactionCostError(ValueError):
    """Raised when transaction costs cannot be calculated."""


@dataclass(frozen=True)
class TransactionCostResult:
    """Transaction-cost calculation result."""

    turnover: float
    cost_rate: float
    transaction_cost: float


@dataclass(frozen=True)
class NetReturnResult:
    """Gross-to-net return calculation result."""

    gross_return: float
    transaction_cost: float
    net_return: float


@dataclass(frozen=True)
class CostAnalysisResult:
    """Portfolio transaction-cost analysis."""

    portfolio_returns: pd.DataFrame
    total_turnover: float
    total_transaction_cost: float
    total_gross_return: float
    total_net_return: float
    cost_rate: float


def calculate_transaction_cost(
    turnover: float,
    *,
    cost_rate: float,
) -> TransactionCostResult:
    """
    Calculate transaction cost.

    Formula:

        TC_t = cost_rate * Turnover_t
    """

    _validate_turnover(turnover)
    _validate_cost_rate(cost_rate)

    transaction_cost = float(
        cost_rate * turnover
    )

    return TransactionCostResult(
        turnover=float(turnover),
        cost_rate=float(cost_rate),
        transaction_cost=transaction_cost,
    )


def calculate_net_return(
    gross_return: float,
    transaction_cost: float,
) -> NetReturnResult:
    """
    Calculate net portfolio return.

    Formula:

        R_net = R_gross - TC
    """

    _validate_return(gross_return)
    _validate_transaction_cost(transaction_cost)

    net_return = float(
        gross_return - transaction_cost
    )

    return NetReturnResult(
        gross_return=float(gross_return),
        transaction_cost=float(transaction_cost),
        net_return=net_return,
    )


def calculate_net_returns(
    portfolio_returns: pd.DataFrame,
    *,
    cost_rate: float,
    turnover_column: str = "turnover",
    gross_return_column: str = "gross_return",
) -> pd.DataFrame:
    """
    Calculate transaction costs and net returns for each
    portfolio observation.

    Required columns:

        turnover
        gross_return

    Added columns:

        transaction_cost
        net_return
    """

    _validate_portfolio_returns(
        portfolio_returns,
        turnover_column=turnover_column,
        gross_return_column=gross_return_column,
    )

    _validate_cost_rate(cost_rate)

    result = portfolio_returns.copy()

    result[turnover_column] = pd.to_numeric(
        result[turnover_column],
        errors="coerce",
    )

    result[gross_return_column] = pd.to_numeric(
        result[gross_return_column],
        errors="coerce",
    )

    if result[turnover_column].isna().any():
        raise TransactionCostError(
            "turnover contains invalid or missing values."
        )

    if result[gross_return_column].isna().any():
        raise TransactionCostError(
            "gross_return contains invalid or missing values."
        )

    turnover_values = result[
        turnover_column
    ].to_numpy(dtype=float)

    gross_return_values = result[
        gross_return_column
    ].to_numpy(dtype=float)

    if not np.isfinite(turnover_values).all():
        raise TransactionCostError(
            "turnover contains non-finite values."
        )

    if not np.isfinite(gross_return_values).all():
        raise TransactionCostError(
            "gross_return contains non-finite values."
        )

    if (turnover_values < 0).any():
        raise TransactionCostError(
            "turnover must not contain negative values."
        )

    result["transaction_cost"] = (
        turnover_values * cost_rate
    )

    result["net_return"] = (
        gross_return_values
        - result["transaction_cost"].to_numpy(
            dtype=float
        )
    )

    return result


def build_cost_analysis(
    portfolio_returns: pd.DataFrame,
    *,
    cost_rate: float,
    turnover_column: str = "turnover",
    gross_return_column: str = "gross_return",
) -> CostAnalysisResult:
    """
    Build a complete transaction-cost analysis.
    """

    result = calculate_net_returns(
        portfolio_returns,
        cost_rate=cost_rate,
        turnover_column=turnover_column,
        gross_return_column=gross_return_column,
    )

    total_turnover = float(
        result[turnover_column].sum()
    )

    total_transaction_cost = float(
        result["transaction_cost"].sum()
    )

    total_gross_return = float(
        result[gross_return_column].sum()
    )

    total_net_return = float(
        result["net_return"].sum()
    )

    return CostAnalysisResult(
        portfolio_returns=result,
        total_turnover=total_turnover,
        total_transaction_cost=total_transaction_cost,
        total_gross_return=total_gross_return,
        total_net_return=total_net_return,
        cost_rate=float(cost_rate),
    )


def calculate_cost_sensitivity(
    portfolio_returns: pd.DataFrame,
    *,
    cost_rates: tuple[float, ...],
    turnover_column: str = "turnover",
    gross_return_column: str = "gross_return",
) -> pd.DataFrame:
    """
    Evaluate portfolio net returns across multiple
    transaction-cost assumptions.

    This supports the locked cost-sensitivity analysis.
    """

    _validate_portfolio_returns(
        portfolio_returns,
        turnover_column=turnover_column,
        gross_return_column=gross_return_column,
    )

    if not cost_rates:
        raise TransactionCostError(
            "At least one cost rate is required."
        )

    validated_rates: list[float] = []

    for rate in cost_rates:
        _validate_cost_rate(rate)
        validated_rates.append(float(rate))

    turnover = pd.to_numeric(
        portfolio_returns[turnover_column],
        errors="coerce",
    )

    gross_returns = pd.to_numeric(
        portfolio_returns[gross_return_column],
        errors="coerce",
    )

    if turnover.isna().any():
        raise TransactionCostError(
            "turnover contains invalid or missing values."
        )

    if gross_returns.isna().any():
        raise TransactionCostError(
            "gross_return contains invalid or missing values."
        )

    turnover_values = turnover.to_numpy(
        dtype=float
    )

    gross_return_values = gross_returns.to_numpy(
        dtype=float
    )

    if (turnover_values < 0).any():
        raise TransactionCostError(
            "turnover must not contain negative values."
        )

    rows: list[dict[str, float]] = []

    total_turnover = float(
        turnover_values.sum()
    )

    total_gross_return = float(
        gross_return_values.sum()
    )

    for rate in validated_rates:
        total_transaction_cost = float(
            total_turnover * rate
        )

        total_net_return = float(
            total_gross_return
            - total_transaction_cost
        )

        rows.append(
            {
                "cost_rate": rate,
                "total_turnover": total_turnover,
                "total_transaction_cost": (
                    total_transaction_cost
                ),
                "total_gross_return": (
                    total_gross_return
                ),
                "total_net_return": (
                    total_net_return
                ),
            }
        )

    return pd.DataFrame(rows)


def _validate_portfolio_returns(
    portfolio_returns: pd.DataFrame,
    *,
    turnover_column: str,
    gross_return_column: str,
) -> None:
    if not isinstance(
        portfolio_returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "portfolio_returns must be a pandas.DataFrame."
        )

    if portfolio_returns.empty:
        raise TransactionCostError(
            "portfolio_returns must not be empty."
        )

    required_columns = {
        turnover_column,
        gross_return_column,
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in portfolio_returns.columns
    ]

    if missing_columns:
        raise TransactionCostError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def _validate_turnover(
    turnover: float,
) -> None:
    if not isinstance(
        turnover,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "turnover must be numeric."
        )

    if not np.isfinite(turnover):
        raise TransactionCostError(
            "turnover must be finite."
        )

    if turnover < 0:
        raise TransactionCostError(
            "turnover must not be negative."
        )


def _validate_cost_rate(
    cost_rate: float,
) -> None:
    if not isinstance(
        cost_rate,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "cost_rate must be numeric."
        )

    if not np.isfinite(cost_rate):
        raise TransactionCostError(
            "cost_rate must be finite."
        )

    if cost_rate < 0:
        raise TransactionCostError(
            "cost_rate must not be negative."
        )


def _validate_return(
    value: float,
) -> None:
    if not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "gross_return must be numeric."
        )

    if not np.isfinite(value):
        raise TransactionCostError(
            "gross_return must be finite."
        )


def _validate_transaction_cost(
    value: float,
) -> None:
    if not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "transaction_cost must be numeric."
        )

    if not np.isfinite(value):
        raise TransactionCostError(
            "transaction_cost must be finite."
        )

    if value < 0:
        raise TransactionCostError(
            "transaction_cost must not be negative."
        )