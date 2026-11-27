from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_RETURN_COLUMNS = (
    "timestamp",
    "asset_id",
    "return",
)


class PortfolioError(ValueError):
    """Raised when portfolio calculations cannot be performed."""


@dataclass(frozen=True)
class PortfolioResult:
    """Result of a portfolio return calculation."""

    portfolio_returns: pd.DataFrame
    asset_count: int
    observation_count: int


@dataclass(frozen=True)
class PortfolioPerformance:
    """Basic portfolio performance statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    downside_deviation: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioResult:
    """
    Calculate portfolio returns from asset returns and portfolio weights.

    Parameters
    ----------
    returns:
        Long-format asset returns with:
        timestamp, asset_id, return.

    weights:
        Long-format portfolio weights with:
        timestamp, asset_id, weight.

        Weight at timestamp t is applied to the corresponding asset
        return at timestamp t.

    risk_free_rate:
        Annualized risk-free rate used for Sharpe and Sortino calculations.

    periods_per_year:
        Number of return observations per year.

    Returns
    -------
    PortfolioResult
        Portfolio-level return series and metadata.
    """

    _validate_returns(returns)
    _validate_weights(weights)
    _validate_periods_per_year(periods_per_year)

    returns_df = returns.loc[
        :,
        list(REQUIRED_RETURN_COLUMNS),
    ].copy()

    weights_df = weights.loc[
        :,
        ["timestamp", "asset_id", "weight"],
    ].copy()

    returns_df["timestamp"] = pd.to_datetime(
        returns_df["timestamp"],
        errors="coerce",
    )

    weights_df["timestamp"] = pd.to_datetime(
        weights_df["timestamp"],
        errors="coerce",
    )

    returns_df["asset_id"] = (
        returns_df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    weights_df["asset_id"] = (
        weights_df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    returns_df["return"] = pd.to_numeric(
        returns_df["return"],
        errors="coerce",
    )
    weights_df["weight"] = pd.to_numeric(
        weights_df["weight"],
        errors="coerce",
    )

    weights_df["weight"] = pd.to_numeric(
        weights_df["weight"],
        errors="coerce",
    )

    return_keys = set(
        zip(returns_df["timestamp"], returns_df["asset_id"], strict=True)
    )
    weight_keys = set(
        zip(weights_df["timestamp"], weights_df["asset_id"], strict=True)
    )
    missing_weights = return_keys - weight_keys
    missing_returns = weight_keys - return_keys
    if missing_weights or missing_returns:
        raise PortfolioError(
            "Returns and weights must cover the same timestamp/asset keys; "
            f"{len(missing_weights)} return keys lack weights and "
            f"{len(missing_returns)} weight keys lack returns."
        )

    merged = returns_df.merge(
        weights_df,
        on=["timestamp", "asset_id"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise PortfolioError(
            "No matching timestamp/asset observations exist "
            "between returns and weights."
        )

    merged["weighted_return"] = (
        merged["return"] * merged["weight"]
    )

    portfolio = (
        merged.groupby(
            "timestamp",
            sort=True,
        )
        .agg(
            portfolio_return=("weighted_return", "sum"),
            gross_exposure=("weight", lambda values: np.abs(values).sum()),
            net_exposure=("weight", "sum"),
        )
        .reset_index()
    )

    portfolio["portfolio_return"] = pd.to_numeric(
        portfolio["portfolio_return"],
        errors="coerce",
    )

    if portfolio["portfolio_return"].isna().any():
        raise PortfolioError(
            "Portfolio return calculation produced invalid values."
        )

    return PortfolioResult(
        portfolio_returns=portfolio,
        asset_count=int(merged["asset_id"].nunique()),
        observation_count=len(portfolio),
    )


def create_static_weights(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Create constant-mix weights for all timestamps.

    These weights imply rebalancing back to the supplied proportions
    at every timestamp. They are not a buy-and-hold weight path.
    """

    _validate_returns(returns)

    if not weights:
        raise PortfolioError(
            "weights must contain at least one asset."
        )

    normalized_weights: dict[str, float] = {}

    for asset_id, weight in weights.items():
        normalized_asset = str(asset_id).strip().upper()

        if not normalized_asset:
            raise PortfolioError(
                "Asset identifiers must not be empty."
            )

        numeric_weight = float(weight)

        if not np.isfinite(numeric_weight):
            raise PortfolioError(
                f"Weight for asset '{normalized_asset}' must be finite."
            )

        normalized_weights[normalized_asset] = numeric_weight

    base = returns.loc[
        :,
        ["timestamp", "asset_id"],
    ].copy()

    base["timestamp"] = pd.to_datetime(
        base["timestamp"],
        errors="coerce",
    )

    base["asset_id"] = (
        base["asset_id"]
        .astype("string")
        .str.upper()
    )

    base = base.drop_duplicates(
        subset=["timestamp", "asset_id"],
    )

    available_assets = set(base["asset_id"].astype(str))
    unknown_assets = sorted(set(normalized_weights) - available_assets)
    if unknown_assets:
        raise PortfolioError(
            "Weights reference assets absent from returns: "
            + ", ".join(unknown_assets)
        )

    base["weight"] = (
        base["asset_id"]
        .map(normalized_weights)
        .fillna(0.0)
    )

    return base.sort_values(
        by=["timestamp", "asset_id"],
        ascending=True,
    ).reset_index(drop=True)


def create_buy_and_hold_weights(
    returns: pd.DataFrame,
    initial_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Create a true buy-and-hold weight path.

    Initial weights must be long-only and sum to one. After each return
    observation, holdings drift with asset returns; no rebalancing occurs.
    Returned weights are the pre-return weights applied at each timestamp.
    """
    _validate_returns(returns)
    if not initial_weights:
        raise PortfolioError("initial_weights must contain at least one asset.")

    frame = returns.loc[:, list(REQUIRED_RETURN_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["asset_id"] = (
        frame["asset_id"].astype("string").str.strip().str.upper()
    )
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    if frame["return"].isna().any() or not np.isfinite(frame["return"]).all():
        raise PortfolioError("Buy-and-hold returns must be finite numeric simple returns.")
    if (frame["return"] <= -1.0).any():
        raise PortfolioError("Buy-and-hold simple returns must be greater than -100%.")

    normalized: dict[str, float] = {}
    for asset_id, raw_weight in initial_weights.items():
        asset = str(asset_id).strip().upper()
        if not asset or asset in normalized:
            raise PortfolioError("Initial asset identifiers must be unique and non-empty.")
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise PortfolioError("Buy-and-hold initial weights must be finite and non-negative.")
        normalized[asset] = weight

    if not np.isclose(sum(normalized.values()), 1.0, rtol=0.0, atol=1e-10):
        raise PortfolioError("Buy-and-hold initial weights must sum to one.")

    assets = set(frame["asset_id"].astype(str))
    if set(normalized) != assets:
        raise PortfolioError(
            "Initial weights must cover exactly the assets in the return dataset."
        )

    date_count = frame["timestamp"].nunique()
    asset_count = frame["asset_id"].nunique()
    if len(frame) != date_count * asset_count:
        raise PortfolioError(
            "Buy-and-hold returns must contain a complete timestamp/asset panel."
        )

    current_weights = dict(normalized)
    rows: list[dict[str, object]] = []

    for timestamp, group in frame.groupby("timestamp", sort=True):
        returns_by_asset = dict(
            zip(group["asset_id"].astype(str), group["return"].astype(float), strict=True)
        )
        if set(returns_by_asset) != assets:
            raise PortfolioError(
                f"Buy-and-hold return panel is incomplete at {timestamp}."
            )

        for asset_id in sorted(assets):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset_id": asset_id,
                    "weight": current_weights[asset_id],
                }
            )

        portfolio_return = sum(
            current_weights[asset_id] * returns_by_asset[asset_id]
            for asset_id in assets
        )
        growth = 1.0 + portfolio_return
        if not np.isfinite(growth) or growth <= 0.0:
            raise PortfolioError(
                "Buy-and-hold portfolio wealth must remain positive."
            )

        current_weights = {
            asset_id: (
                current_weights[asset_id]
                * (1.0 + returns_by_asset[asset_id])
                / growth
            )
            for asset_id in assets
        }

    return pd.DataFrame(rows).sort_values(
        ["timestamp", "asset_id"]
    ).reset_index(drop=True)


def calculate_performance(
    portfolio_returns: pd.Series | pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioPerformance:
    """
    Calculate portfolio performance statistics from periodic simple returns.

    Sharpe and Sortino use the arithmetic mean of periodic excess returns.
    Drawdown includes initial capital (wealth starts at 1.0).
    """
    _validate_periods_per_year(periods_per_year)

    if not isinstance(risk_free_rate, (int, float, np.integer, np.floating)):
        raise TypeError("risk_free_rate must be numeric.")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1.0:
        raise PortfolioError("risk_free_rate must be finite and greater than -1.")

    if isinstance(portfolio_returns, pd.DataFrame):
        if "portfolio_return" not in portfolio_returns.columns:
            raise PortfolioError("DataFrame must contain 'portfolio_return'.")
        series = portfolio_returns["portfolio_return"]
    elif isinstance(portfolio_returns, pd.Series):
        series = portfolio_returns
    else:
        raise TypeError("portfolio_returns must be a pandas Series or DataFrame.")

    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        raise PortfolioError(
            "At least two valid portfolio return observations are required."
        )

    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioError("Portfolio returns must contain only finite values.")
    if (values <= -1.0).any():
        raise PortfolioError(
            "Periodic simple returns must be greater than -100%."
        )

    gross_growth = float(np.prod(1.0 + values))
    if not np.isfinite(gross_growth) or gross_growth <= 0.0:
        raise PortfolioError("Compounded portfolio wealth must remain positive.")

    total_return = gross_growth - 1.0
    annualized_return = float(
        gross_growth ** (periods_per_year / len(values)) - 1.0
    )

    periodic_risk_free = (
        (1.0 + float(risk_free_rate)) ** (1.0 / periods_per_year) - 1.0
    )
    excess_returns = values - periodic_risk_free

    annualized_volatility = float(
        np.std(values, ddof=1) * np.sqrt(periods_per_year)
    )
    excess_std = float(np.std(excess_returns, ddof=1))
    sharpe_ratio = (
        float(np.mean(excess_returns) / excess_std * np.sqrt(periods_per_year))
        if excess_std > 0.0
        else 0.0
    )

    downside = np.minimum(excess_returns, 0.0)
    periodic_downside_deviation = float(
        np.sqrt(np.mean(np.square(downside)))
    )
    downside_deviation = float(
        periodic_downside_deviation * np.sqrt(periods_per_year)
    )
    sortino_ratio = (
        float(
            np.mean(excess_returns)
            / periodic_downside_deviation
            * np.sqrt(periods_per_year)
        )
        if periodic_downside_deviation > 0.0
        else 0.0
    )

    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    max_drawdown = float(np.min(drawdowns))

    calmar_ratio = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown < 0.0
        else 0.0
    )

    return PortfolioPerformance(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        downside_deviation=downside_deviation,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
    )


def _validate_returns(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "returns must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise PortfolioError(
            "returns must not be empty."
        )

    missing = [
        column
        for column in REQUIRED_RETURN_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise PortfolioError(
            "Missing required return columns: "
            + ", ".join(missing)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise PortfolioError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise PortfolioError(
            "asset_id contains missing values."
        )

    asset_ids = (
        dataframe["asset_id"]
        .astype("string")
        .str.strip()
    )

    if (asset_ids == "").any():
        raise PortfolioError(
            "asset_id contains empty values."
        )

    return_values = pd.to_numeric(
        dataframe["return"],
        errors="coerce",
    )

    if return_values.isna().any():
        raise PortfolioError(
            "return contains invalid or missing values."
        )

    if not np.isfinite(
        return_values.to_numpy(dtype=float)
    ).all():
        raise PortfolioError(
            "return must contain only finite values."
        )

    duplicate_mask = dataframe.duplicated(
        subset=["timestamp", "asset_id"],
        keep=False,
    )

    if duplicate_mask.any():
        raise PortfolioError(
            "returns contains duplicate timestamp/asset observations."
        )


def _validate_weights(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "weights must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise PortfolioError(
            "weights must not be empty."
        )

    required = (
        "timestamp",
        "asset_id",
        "weight",
    )

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise PortfolioError(
            "Missing required weight columns: "
            + ", ".join(missing)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise PortfolioError(
            "weight timestamp contains invalid or missing values."
        )

    weights = pd.to_numeric(
        dataframe["weight"],
        errors="coerce",
    )

    if weights.isna().any():
        raise PortfolioError(
            "weight contains invalid or missing values."
        )

    if not np.isfinite(
        weights.to_numpy(dtype=float)
    ).all():
        raise PortfolioError(
            "weight must contain only finite values."
        )

    duplicate_mask = dataframe.duplicated(
        subset=["timestamp", "asset_id"],
        keep=False,
    )

    if duplicate_mask.any():
        raise PortfolioError(
            "weights contains duplicate timestamp/asset observations."
        )


def _validate_periods_per_year(
    periods_per_year: int,
) -> None:
    if (
        not isinstance(periods_per_year, int)
        or isinstance(periods_per_year, bool)
    ):
        raise TypeError(
            "periods_per_year must be an integer."
        )

    if periods_per_year <= 0:
        raise PortfolioError(
            "periods_per_year must be greater than zero."
        )