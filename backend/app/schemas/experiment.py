from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRequest(BaseModel):
    """Request schema for creating an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=100,
    )

    description: str = Field(
        default="",
        max_length=500,
    )

    models: list[str] = Field(
        min_length=1,
    )

    metrics: list[str] = Field(
        min_length=1,
    )

    risk_measures: list[str] = Field(
        default_factory=list,
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ExperimentResponse(BaseModel):
    """Response schema describing an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str

    description: str = ""

    models: list[str]

    metrics: list[str]

    risk_measures: list[str] = Field(
        default_factory=list,
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    status: str = "configured"


class ExperimentRunRequest(BaseModel):
    """Request schema for running configured experiments."""

    model_config = ConfigDict(extra="forbid")

    experiment_names: list[str] = Field(
        min_length=1,
    )


class ExperimentRunResultResponse(BaseModel):
    """Response schema for one experiment execution."""

    model_config = ConfigDict(extra="forbid")

    name: str

    status: str

    result: dict[str, Any] = Field(
        default_factory=dict,
    )

    error: str | None = None


class ExperimentRunResponse(BaseModel):
    """Response schema for an experiment execution batch."""

    model_config = ConfigDict(extra="forbid")

    experiment_count: int = Field(
        ge=0,
    )

    successful_count: int = Field(
        ge=0,
    )

    failed_count: int = Field(
        ge=0,
    )

    all_successful: bool

    results: list[ExperimentRunResultResponse]


class ExperimentListResponse(BaseModel):
    """Response schema for listing experiments."""

    model_config = ConfigDict(extra="forbid")

    experiments: list[ExperimentResponse]

    count: int = Field(
        ge=0,
    )