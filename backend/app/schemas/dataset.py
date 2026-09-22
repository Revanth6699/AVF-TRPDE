from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetRequest(BaseModel):
    """Request model for registering a dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    assets: list[str] = Field(min_length=1)
    start_timestamp: datetime
    end_timestamp: datetime
    row_count: int = Field(ge=0)
    fingerprint: str = Field(min_length=1)


class DatasetResponse(BaseModel):
    """Response model describing a registered dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    frequency: str
    assets: list[str]
    start_timestamp: datetime
    end_timestamp: datetime
    row_count: int
    fingerprint: str
    status: str


class DatasetValidationRequest(BaseModel):
    """Request model for dataset validation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    assets: list[str] = Field(min_length=1)


class DatasetValidationResponse(BaseModel):
    """Response model for dataset validation results."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    row_count: int = Field(ge=0)
    asset_count: int = Field(ge=0)
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    assets: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetListItem(BaseModel):
    """Compact dataset representation for listings."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    frequency: str
    asset_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    fingerprint: str


class DatasetListResponse(BaseModel):
    """Response model for dataset listings."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetListItem] = Field(default_factory=list)
    count: int = Field(ge=0)