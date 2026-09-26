from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.forecast import (
    ForecastComparisonResponse,
    ForecastEvaluationResponse,
    ForecastRequest,
    ForecastResponse,
)
from backend.app.storage import (
    RecordExistsError,
    storage,
)


router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
)


def _forecast_key(
    experiment_name: str,
    model: str,
) -> str:
    """Build a deterministic forecast key."""

    return f"{experiment_name}:{model}"


def _validate_request(
    request: ForecastRequest,
) -> None:
    """Validate forecast request constraints not covered by Pydantic."""

    if request.start_timestamp > request.end_timestamp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "start_timestamp must be earlier than or equal "
                "to end_timestamp."
            ),
        )

    if not request.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one asset is required.",
        )


def _forecast_response(
    record: dict,
) -> ForecastResponse:
    """Convert a persistent forecast record into an API response."""

    return ForecastResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        model=record["model"],
        forecasts=record["forecasts"],
        observation_count=record["observation_count"],
        asset_count=record["asset_count"],
        status=record["status"],
    )


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Register a forecast request/result.

    The API contract is separated from model execution. Actual
    walk-forward model execution is performed by the research layer.
    """

    _validate_request(request)

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "model": request.model,
        "forecasts": [],
        "observation_count": 0,
        "asset_count": len(request.assets),
        "status": "accepted",
    }

    try:
        storage.save_forecast(record)
    except RecordExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _forecast_response(record)


@router.get(
    "/{experiment_name}",
    response_model=ForecastResponse,
)
def get_forecast(
    experiment_name: str,
    model: str | None = None,
) -> ForecastResponse:
    """
    Retrieve a registered forecast result.

    If multiple models exist for an experiment, the model must be
    supplied explicitly.
    """

    if model is not None:
        record = storage.get_forecast(
            experiment_name,
            model,
        )

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No forecast found for experiment "
                    f"'{experiment_name}' and model '{model}'."
                ),
            )

        return _forecast_response(record)

    records = storage.list_forecasts(
        experiment_name,
    )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No forecast found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if len(records) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple forecast models exist for experiment "
                f"'{experiment_name}'. Supply the 'model' query "
                f"parameter."
            ),
        )

    return _forecast_response(records[0])


@router.post(
    "/evaluate",
    response_model=ForecastEvaluationResponse,
)
def evaluate_forecast(
    request: ForecastRequest,
) -> ForecastEvaluationResponse:
    """
    Evaluate a registered out-of-sample forecast.

    Actual MAE/RMSE/QLIKE computation belongs to the research
    evaluation layer.
    """

    _validate_request(request)

    forecast = storage.get_forecast(
        request.experiment_name,
        request.model,
    )

    if forecast is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No forecast registered for experiment "
                f"'{request.experiment_name}' and model "
                f"'{request.model}'."
            ),
        )

    if not forecast["forecasts"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Forecast contains no observations. "
                "Run the research forecast pipeline before evaluation."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Forecast evaluation requires the existing research "
            "evaluation pipeline to expose its callable execution "
            "contract. No model evaluation is fabricated in the API layer."
        ),
    )


@router.post(
    "/compare",
    response_model=ForecastComparisonResponse,
)
def compare_forecasts(
    request: ForecastRequest,
) -> ForecastComparisonResponse:
    """
    Compare registered model forecasts.

    Comparison uses the locked MAE, RMSE and QLIKE metrics once
    the research evaluation layer provides the forecast-result
    integration contract.
    """

    _validate_request(request)

    experiment_forecasts = storage.list_forecasts(
        request.experiment_name,
    )

    if not experiment_forecasts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No forecasts registered for experiment "
                f"'{request.experiment_name}'."
            ),
        )

    populated = [
        forecast
        for forecast in experiment_forecasts
        if forecast["forecasts"]
    ]

    if not populated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Registered forecasts contain no observations. "
                "Run the research forecast pipeline first."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Forecast comparison requires the existing research "
            "evaluation pipeline integration contract."
        ),
    )