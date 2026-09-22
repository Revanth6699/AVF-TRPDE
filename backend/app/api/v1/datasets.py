from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.dataset import (
    DatasetListResponse,
    DatasetRequest,
    DatasetResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
)

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset(request: DatasetRequest) -> DatasetResponse:
    """Register a dataset definition."""

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    return DatasetResponse(
        name=request.name,
        source=request.source,
        frequency=request.frequency,
        assets=request.assets,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        row_count=0,
        fingerprint=None,
        status="registered",
    )


@router.post(
    "/validate",
    response_model=DatasetValidationResponse,
)
def validate_dataset(
    request: DatasetValidationRequest,
) -> DatasetValidationResponse:
    """Validate a registered dataset definition."""

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    return DatasetValidationResponse(
        valid=True,
        row_count=0,
        asset_count=len(request.assets),
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        assets=request.assets,
        errors=[],
        warnings=[],
    )


@router.get(
    "",
    response_model=DatasetListResponse,
)
def list_datasets() -> DatasetListResponse:
    """List registered datasets."""

    return DatasetListResponse(
        datasets=[],
        count=0,
    )