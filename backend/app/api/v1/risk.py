from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.risk import (
    RiskBacktestResponse,
    RiskComparisonResponse,
    RiskRequest,
    RiskResponse,
)
from backend.app.storage import (
    RecordExistsError,
    storage,
)


router = APIRouter(
    prefix="/risk",
    tags=["Risk"],
)


def _risk_key(
    experiment_name: str,
    model: str,
) -> str:
    """Build a deterministic key for a risk result."""

    return f"{experiment_name}:{model}"


def _validate_request(
    request: RiskRequest,
) -> None:
    """Validate request constraints not handled by Pydantic."""

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

    if not request.confidence_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one confidence level is required.",
        )

    for confidence_level in request.confidence_levels:
        if not 0.0 < confidence_level < 1.0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "confidence_levels must contain values "
                    "strictly between 0 and 1."
                ),
            )


def _risk_response(
    record: dict,
) -> RiskResponse:
    """Convert a persistent risk record to the API response."""

    return RiskResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        model=record["model"],
        risk_points=record["risk_points"],
        observation_count=record["observation_count"],
        asset_count=record["asset_count"],
        status=record["status"],
    )


@router.post(
    "",
    response_model=RiskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_risk_analysis(
    request: RiskRequest,
) -> RiskResponse:
    """
    Register a tail-risk analysis request.

    Actual FHS, VaR and ES calculations belong to the
    research/risk layer.
    """

    _validate_request(request)

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "model": request.model,
        "risk_points": [],
        "observation_count": 0,
        "asset_count": len(request.assets),
        "status": "accepted",
    }

    try:
        storage.save_risk(record)
    except RecordExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _risk_response(record)


@router.get(
    "/{experiment_name}",
    response_model=RiskResponse,
)
def get_risk_analysis(
    experiment_name: str,
    model: str | None = None,
) -> RiskResponse:
    """
    Retrieve a persistently registered risk result.

    When multiple models exist for an experiment, the model
    must be supplied explicitly.
    """

    if model is not None:
        record = storage.get_risk(
            experiment_name,
            model,
        )

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No risk analysis found for experiment "
                    f"'{experiment_name}' and model '{model}'."
                ),
            )

        return _risk_response(record)

    records = storage.list_risks(
        experiment_name,
    )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analysis found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if len(records) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple risk models exist for experiment "
                f"'{experiment_name}'. Supply the 'model' "
                "query parameter."
            ),
        )

    return _risk_response(records[0])


@router.post(
    "/backtest",
    response_model=RiskBacktestResponse,
)
def backtest_risk(
    request: RiskRequest,
) -> RiskBacktestResponse:
    """
    Backtest registered VaR forecasts.

    Kupiec unconditional-coverage and Christoffersen tests
    are performed by the research/risk layer.
    """

    _validate_request(request)

    risk_result = storage.get_risk(
        request.experiment_name,
        request.model,
    )

    if risk_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analysis registered for experiment "
                f"'{request.experiment_name}' and model "
                f"'{request.model}'."
            ),
        )

    if not risk_result["risk_points"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Risk analysis contains no observations. "
                "Run the risk pipeline before backtesting."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Risk backtesting requires the existing Kupiec and "
            "Christoffersen engines to expose their result "
            "integration contract to the API layer."
        ),
    )


@router.post(
    "/compare",
    response_model=RiskComparisonResponse,
)
def compare_risk(
    request: RiskRequest,
) -> RiskComparisonResponse:
    """
    Compare registered risk estimates across models.

    Comparison uses the locked VaR 95%, VaR 99% and ES 95%
    measures.
    """

    _validate_request(request)

    experiment_results = storage.list_risks(
        request.experiment_name,
    )

    if not experiment_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analyses registered for experiment "
                f"'{request.experiment_name}'."
            ),
        )

    populated = [
        result
        for result in experiment_results
        if result["risk_points"]
    ]

    if not populated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Registered risk analyses contain no observations. "
                "Run the risk pipeline before comparison."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Risk comparison requires the existing risk engine "
            "integration contract."
        ),
    )