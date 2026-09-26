from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.dataset import (
    DatasetListItem,
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


_DATASETS: dict[str, DatasetResponse] = {}


def _validate_date_range(
    start_timestamp,
    end_timestamp,
) -> None:
    """Validate dataset time boundaries."""

    if start_timestamp > end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "start_timestamp must be earlier than or equal "
                "to end_timestamp."
            ),
        )


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset(
    request: DatasetRequest,
) -> DatasetResponse:
    """Register a dataset definition."""

    _validate_date_range(
        request.start_timestamp,
        request.end_timestamp,
    )

    if request.name in _DATASETS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Dataset '{request.name}' is already registered."
            ),
        )

    response = DatasetResponse(
        name=request.name,
        source=request.source,
        frequency=request.frequency,
        assets=request.assets,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        row_count=request.row_count,
        fingerprint=request.fingerprint,
        status="registered",
    )

    _DATASETS[request.name] = response

    return response


@router.post(
    "/validate",
    response_model=DatasetValidationResponse,
)
def validate_dataset(
    request: DatasetValidationRequest,
) -> DatasetValidationResponse:
    """
    Validate a registered dataset definition.

    The API validates registration metadata here. It does not
    fabricate data-level validation results when no dataset
    storage/loader has been connected.
    """

    registered = _DATASETS.get(request.name)

    if registered is None:
        return DatasetValidationResponse(
            valid=False,
            row_count=0,
            asset_count=len(request.assets),
            start_timestamp=None,
            end_timestamp=None,
            assets=request.assets,
            errors=[
                (
                    f"Dataset '{request.name}' is not registered."
                )
            ],
            warnings=[],
        )

    errors: list[str] = []
    warnings: list[str] = []

    if registered.source != request.source:
        errors.append(
            "Requested source does not match the registered dataset."
        )

    if registered.frequency != request.frequency:
        errors.append(
            "Requested frequency does not match the registered dataset."
        )

    if registered.assets != request.assets:
        errors.append(
            "Requested assets do not match the registered dataset."
        )

    return DatasetValidationResponse(
        valid=not errors,
        row_count=registered.row_count,
        asset_count=len(registered.assets),
        start_timestamp=registered.start_timestamp,
        end_timestamp=registered.end_timestamp,
        assets=registered.assets,
        errors=errors,
        warnings=warnings,
    )


@router.get(
    "",
    response_model=DatasetListResponse,
)
def list_datasets() -> DatasetListResponse:
    """List registered datasets."""

    datasets = [
        DatasetListItem(
            name=dataset.name,
            source=dataset.source,
            frequency=dataset.frequency,
            asset_count=len(dataset.assets),
            row_count=dataset.row_count,
            start_timestamp=dataset.start_timestamp,
            end_timestamp=dataset.end_timestamp,
            fingerprint=dataset.fingerprint,
        )
        for dataset in _DATASETS.values()
    ]

    return DatasetListResponse(
        datasets=datasets,
        count=len(datasets),
    )