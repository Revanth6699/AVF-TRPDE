from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.risk import (
    RiskBacktestResponse,
    RiskComparisonResponse,
    RiskRequest,
    RiskResponse,
)


router = APIRouter(
    prefix="/risk",
    tags=["Risk"],
)


# API-level result registry.
# Actual risk calculations remain in backend.app.risk.
_RISK_RESULTS: dict[str, RiskResponse] = {}


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

    Actual FHS, VaR and ES calculations belong to the risk engine.
    """

    _validate_request(request)

    key = _risk_key(
        request.experiment_name,
        request.model,
    )

    if key in _RISK_RESULTS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Risk analysis for experiment "
                f"'{request.experiment_name}' and model "
                f"'{request.model}' already exists."
            ),
        )

    response = RiskResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        model=request.model,
        risk_points=[],
        observation_count=0,
        asset_count=len(request.assets),
        status="accepted",
    )

    _RISK_RESULTS[key] = response

    return response


@router.get(
    "/{experiment_name}",
    response_model=RiskResponse,
)
def get_risk_analysis(
    experiment_name: str,
    model: str | None = None,
) -> RiskResponse:
    """
    Retrieve a registered risk result.

    When multiple models exist for an experiment, the model must
    be supplied explicitly.
    """

    matches = [
        result
        for key, result in _RISK_RESULTS.items()
        if key.startswith(f"{experiment_name}:")
    ]

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analysis found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if model is not None:
        key = _risk_key(
            experiment_name,
            model,
        )

        result = _RISK_RESULTS.get(key)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No risk analysis found for experiment "
                    f"'{experiment_name}' and model '{model}'."
                ),
            )

        return result

    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple risk models exist for experiment "
                f"'{experiment_name}'. Supply the 'model' query "
                f"parameter."
            ),
        )

    return matches[0]


@router.post(
    "/backtest",
    response_model=RiskBacktestResponse,
)
def backtest_risk(
    request: RiskRequest,
) -> RiskBacktestResponse:
    """
    Backtest registered VaR forecasts.

    Kupiec unconditional-coverage and Christoffersen tests are
    performed by the research/risk layer.
    """

    _validate_request(request)

    key = _risk_key(
        request.experiment_name,
        request.model,
    )

    risk_result = _RISK_RESULTS.get(key)

    if risk_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analysis registered for experiment "
                f"'{request.experiment_name}' and model "
                f"'{request.model}'."
            ),
        )

    if not risk_result.risk_points:
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

    Comparison uses the locked VaR 95%, VaR 99% and ES measures.
    """

    _validate_request(request)

    experiment_results = [
        result
        for key, result in _RISK_RESULTS.items()
        if key.startswith(f"{request.experiment_name}:")
    ]

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
        if result.risk_points
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