from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PortfolioRequest(BaseModel):
    """Request schema for a portfolio experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    assets: list[str] = Field(min_length=1)

    start_timestamp: datetime
    end_timestamp: datetime

    volatility_target: float = Field(gt=0.0)
    max_position: float = Field(gt=0.0)
    max_turnover: float = Field(gt=0.0)
    transaction_cost: float = Field(ge=0.0)


class PortfolioPosition(BaseModel):
    """Portfolio position for one asset at one timestamp."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    asset_id: str = Field(min_length=1)
    weight: float


class PortfolioResponse(BaseModel):
    """Portfolio experiment response."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    dataset_name: str
    strategy: str

    positions: list[PortfolioPosition]

    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)

    status: str


class PortfolioMetric(BaseModel):
    """Single portfolio performance metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    value: float


class PortfolioEvaluationResponse(BaseModel):
    """Portfolio performance evaluation."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    strategy: str

    metrics: list[PortfolioMetric]

    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)


class PortfolioComparisonItem(BaseModel):
    """Portfolio comparison result for one strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(min_length=1)

    total_return: float
    volatility: float = Field(ge=0.0)
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float

    var_95: float
    var_99: float
    es_95: float

    turnover: float = Field(ge=0.0)
    transaction_costs: float = Field(ge=0.0)
    gross_pnl: float
    net_pnl: float


class PortfolioComparisonResponse(BaseModel):
    """Comparison of portfolio strategies."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str

    strategies: list[PortfolioComparisonItem]

    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)


class PortfolioRunResult(BaseModel):
    """Result of a single portfolio strategy execution."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    status: str
    result: dict


class PortfolioRunResponse(BaseModel):
    """Response for a batch portfolio execution."""

    model_config = ConfigDict(extra="forbid")

    experiment_count: int = Field(ge=0)
    successful_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    all_successful: bool

    results: list[PortfolioRunResult]