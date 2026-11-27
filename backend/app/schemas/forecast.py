from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    """Request model for generating volatility forecasts."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    assets: list[str] = Field(min_length=1)
    start_timestamp: datetime
    end_timestamp: datetime


class ForecastPoint(BaseModel):
    """Single out-of-sample volatility forecast."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    origin_timestamp: datetime
    asset_id: str
    actual_return: float
    actual_volatility: float
    predicted_volatility: float = Field(gt=0.0)


class ForecastResponse(BaseModel):
    """Response containing volatility forecasts."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    dataset_name: str
    model: str
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    status: str


class ForecastMetric(BaseModel):
    """Forecast evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    model: str
    asset_id: str
    metric: str
    value: float


class ForecastEvaluationResponse(BaseModel):
    """Response containing forecast evaluation results."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    metrics: list[ForecastMetric] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)


class ForecastComparisonItem(BaseModel):
    """Comparison result for one volatility model."""

    model_config = ConfigDict(extra="forbid")

    model: str
    asset_id: str
    mae: float
    rmse: float
    qlike: float


class ForecastComparisonResponse(BaseModel):
    """Response containing multi-model forecast comparison."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    models: list[ForecastComparisonItem] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    aggregation: str = "per_asset_on_common_asset_date_observations"