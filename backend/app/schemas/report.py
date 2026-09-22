from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReportRequest(BaseModel):
    """Request schema for generating a research report."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)

    report_type: str = Field(min_length=1)

    start_timestamp: datetime
    end_timestamp: datetime

    include_forecasts: bool = True
    include_risk: bool = True
    include_portfolio: bool = True
    include_stress: bool = True
    include_statistical_tests: bool = True


class ReportSection(BaseModel):
    """Single section of a research report."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    content: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    """Generated research report response."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    dataset_name: str
    report_type: str

    generated_at: datetime

    sections: list[ReportSection]

    status: str


class ResearchConclusion(BaseModel):
    """Structured research conclusion."""

    model_config = ConfigDict(extra="forbid")

    research_question: str
    null_hypothesis: str
    alternative_hypothesis: str

    conclusion: str

    statistical_significance: bool
    economic_significance: bool

    supporting_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ReportArtifact(BaseModel):
    """Generated report artifact."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class ReportGenerationResult(BaseModel):
    """Result of generating a report."""

    model_config = ConfigDict(extra="forbid")

    report_type: str
    status: str

    artifacts: list[ReportArtifact] = Field(default_factory=list)

    error: str | None = None


class ReportRunResponse(BaseModel):
    """Response for report generation."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str

    status: str

    reports: list[ReportGenerationResult]

    conclusion: ResearchConclusion | None = None