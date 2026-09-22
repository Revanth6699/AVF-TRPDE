from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RiskRequest(BaseModel):
    """Request model for tail-risk calculation."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    assets: list[str] = Field(min_length=1)
    confidence_levels: list[float] = Field(min_length=1)
    start_timestamp: datetime
    end_timestamp: datetime


class RiskPoint(BaseModel):
    """Single VaR/ES observation."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    asset_id: str
    var_95: float
    var_99: float
    es_95: float


class RiskResponse(BaseModel):
    """Response containing tail-risk estimates."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    dataset_name: str
    model: str
    risk_points: list[RiskPoint] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    status: str


class RiskBacktestResult(BaseModel):
    """Backtesting result for one risk measure."""

    model_config = ConfigDict(extra="forbid")

    model: str
    confidence_level: float = Field(gt=0.0, lt=1.0)
    measure: str
    observations: int = Field(ge=0)
    violations: int = Field(ge=0)
    violation_rate: float = Field(ge=0.0, le=1.0)
    statistic: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)


class RiskBacktestResponse(BaseModel):
    """Response containing VaR backtesting results."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    results: list[RiskBacktestResult] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)


class RiskComparisonItem(BaseModel):
    """Risk comparison for one model."""

    model_config = ConfigDict(extra="forbid")

    model: str
    var_95: float
    var_99: float
    es_95: float


class RiskComparisonResponse(BaseModel):
    """Response containing cross-model tail-risk comparison."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    models: list[RiskComparisonItem] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)