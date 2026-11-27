from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StressRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_name: str = Field(min_length=1)
    strategy: str = Field(min_length=1)


class StressRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_name: str
    strategy: str
    status: str
    scenario_count: int = Field(ge=0)
    total_observations: int = Field(ge=0)
    results: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
