from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.report import (
    ReportGenerationResult,
    ReportRequest,
    ReportResponse,
    ReportRunResponse,
)


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_200_OK,
)
def create_report(
    request: ReportRequest,
) -> ReportResponse:
    """
    Create a research report request.

    Actual report generation belongs to the experiments/reporting
    layer and is not executed inside the API router.
    """

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "start_timestamp must be earlier than or equal "
                "to end_timestamp."
            ),
        )

    return ReportResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        report_type=request.report_type,
        generated_at=None,
        sections=[],
        status="accepted",
    )


@router.get(
    "/{experiment_name}",
    response_model=ReportResponse,
)
def get_report(
    experiment_name: str,
) -> ReportResponse:
    """
    Retrieve a generated report for an experiment.
    """

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail=(
            f"Report retrieval for experiment "
            f"'{experiment_name}' is not connected to storage yet."
        ),
    )


@router.post(
    "/run",
    response_model=ReportRunResponse,
)
def run_report(
    request: ReportRequest,
) -> ReportRunResponse:
    """
    Execute report generation through the research/reporting layer.
    """

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "start_timestamp must be earlier than or equal "
                "to end_timestamp."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail="Report generation engine is not connected yet.",
    )