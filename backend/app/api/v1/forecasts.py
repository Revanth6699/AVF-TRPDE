from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.forecast import (
    ForecastComparisonResponse,
    ForecastEvaluationResponse,
    ForecastRequest,
    ForecastResponse,
)


router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
)


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
)
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Create a forecast request.

    The API layer validates the request contract and exposes the
    forecasting interface. Actual model execution is handled by the
    research/walk-forward layer.
    """

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    if not request.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one asset is required.",
        )

    return ForecastResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        model=request.model,
        forecasts=[],
        observation_count=0,
        asset_count=len(request.assets),
        status="accepted",
    )


@router.get(
    "/{experiment_name}",
    response_model=ForecastResponse,
)
def get_forecast(
    experiment_name: str,
) -> ForecastResponse:
    """
    Retrieve forecast results for an experiment.

    Persistent forecast-result retrieval will be connected to the
    experiment/artifact storage layer.
    """

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail=(
            f"Forecast result retrieval for experiment "
            f"'{experiment_name}' is not connected to storage yet."
        ),
    )


@router.post(
    "/evaluate",
    response_model=ForecastEvaluationResponse,
)
def evaluate_forecast(
    request: ForecastRequest,
) -> ForecastEvaluationResponse:
    """
    Evaluate out-of-sample forecasts.

    MAE, RMSE and QLIKE calculation is performed by the research
    evaluation layer, not directly inside the API router.
    """

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail="Forecast evaluation engine is not connected yet.",
    )


@router.post(
    "/compare",
    response_model=ForecastComparisonResponse,
)
def compare_forecasts(
    request: ForecastRequest,
) -> ForecastComparisonResponse:
    """
    Compare forecast performance across models.

    Model comparison is performed by the research evaluation layer
    using the locked MAE, RMSE and QLIKE metrics.
    """

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail="Forecast comparison engine is not connected yet.",
    )