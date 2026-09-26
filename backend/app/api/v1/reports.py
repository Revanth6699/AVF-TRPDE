from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.report import (
    ReportRequest,
    ReportResponse,
    ReportRunResponse,
)


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


_REPORTS: dict[str, ReportResponse] = {}


def _report_key(
    experiment_name: str,
    report_type: str,
) -> str:
    """Build a deterministic report registry key."""

    return f"{experiment_name}:{report_type}"


def _validate_request(
    request: ReportRequest,
) -> None:
    """Validate report request constraints."""

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "start_timestamp must be earlier than or equal "
                "to end_timestamp."
            ),
        )


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_report(
    request: ReportRequest,
) -> ReportResponse:
    """
    Register a research report request.

    Actual report generation belongs to the research/reporting
    layer and is not fabricated by the API.
    """

    _validate_request(request)

    key = _report_key(
        request.experiment_name,
        request.report_type,
    )

    if key in _REPORTS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Report '{request.report_type}' for experiment "
                f"'{request.experiment_name}' already exists."
            ),
        )

    response = ReportResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        report_type=request.report_type,
        generated_at=datetime.now(timezone.utc),
        sections=[],
        status="accepted",
    )

    _REPORTS[key] = response

    return response


@router.get(
    "/{experiment_name}",
    response_model=ReportResponse,
)
def get_report(
    experiment_name: str,
    report_type: str | None = None,
) -> ReportResponse:
    """
    Retrieve a registered report request/result.

    If multiple report types exist for an experiment,
    report_type must be supplied.
    """

    matches = [
        report
        for key, report in _REPORTS.items()
        if key.startswith(f"{experiment_name}:")
    ]

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No report found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if report_type is not None:
        key = _report_key(
            experiment_name,
            report_type,
        )

        report = _REPORTS.get(key)

        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No report of type '{report_type}' found "
                    f"for experiment '{experiment_name}'."
                ),
            )

        return report

    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple report types exist for experiment "
                f"'{experiment_name}'. Supply the "
                f"'report_type' query parameter."
            ),
        )

    return matches[0]


@router.post(
    "/run",
    response_model=ReportRunResponse,
)
def run_report(
    request: ReportRequest,
) -> ReportRunResponse:
    """
    Execute report generation through the research/reporting layer.

    The API does not fabricate research findings when the
    reporting engine is not connected.
    """

    _validate_request(request)

    key = _report_key(
        request.experiment_name,
        request.report_type,
    )

    existing = _REPORTS.get(key)

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No report request registered for experiment "
                f"'{request.experiment_name}' and report type "
                f"'{request.report_type}'. Create the report "
                f"request before running it."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Report generation requires the existing "
            "research/reporting engine integration contract."
        ),
    )