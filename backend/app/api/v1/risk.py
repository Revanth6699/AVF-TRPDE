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


@router.post(
    "",
    response_model=RiskResponse,
    status_code=status.HTTP_200_OK,
)
def create_risk_analysis(request: RiskRequest) -> RiskResponse:
    """
    Create a tail-risk analysis request.

    The API validates the request contract. Actual FHS, VaR, ES,
    and backtesting calculations belong to the research/risk layer.
    """

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

    return RiskResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        model=request.model,
        risk_points=[],
        observation_count=0,
        asset_count=len(request.assets),
        status="accepted",
    )


@router.get(
    "/{experiment_name}",
    response_model=RiskResponse,
)
def get_risk_analysis(
    experiment_name: str,
) -> RiskResponse:
    """
    Retrieve stored risk-analysis results.
    """

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail=(
            f"Risk result retrieval for experiment "
            f"'{experiment_name}' is not connected to storage yet."
        ),
    )


@router.post(
    "/backtest",
    response_model=RiskBacktestResponse,
)
def backtest_risk(
    request: RiskRequest,
) -> RiskBacktestResponse:
    """
    Backtest VaR/ES forecasts.

    Actual Kupiec and Christoffersen calculations belong to the
    research/risk backtesting layer.
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
        detail=(
            "Risk backtesting engine is not connected yet."
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
    Compare risk estimates across models.

    Actual FHS, VaR, ES and statistical backtesting calculations
    belong to the research/risk layer.
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
        detail=(
            "Risk comparison engine is not connected yet."
        ),
    )