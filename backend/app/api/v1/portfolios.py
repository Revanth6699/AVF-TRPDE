from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.portfolio import (
    PortfolioComparisonResponse,
    PortfolioRequest,
    PortfolioResponse,
)


router = APIRouter(
    prefix="/portfolios",
    tags=["Portfolios"],
)


@router.post(
    "",
    response_model=PortfolioResponse,
    status_code=status.HTTP_200_OK,
)
def create_portfolio_analysis(
    request: PortfolioRequest,
) -> PortfolioResponse:
    """
    Create a portfolio-analysis request.

    Portfolio construction, risk targeting, constraints, turnover,
    and transaction-cost calculations belong to the portfolio layer.
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

    return PortfolioResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        strategy=request.strategy,
        portfolios=[],
        observation_count=0,
        asset_count=len(request.assets),
        status="accepted",
    )


@router.get(
    "/{experiment_name}",
    response_model=PortfolioResponse,
)
def get_portfolio_analysis(
    experiment_name: str,
) -> PortfolioResponse:
    """
    Retrieve stored portfolio-analysis results.
    """

    raise HTTPException(
        status_code=status.HTTP_404_NOT_IMPLEMENTED,
        detail=(
            f"Portfolio result retrieval for experiment "
            f"'{experiment_name}' is not connected to storage yet."
        ),
    )


@router.post(
    "/compare",
    response_model=PortfolioComparisonResponse,
)
def compare_portfolios(
    request: PortfolioRequest,
) -> PortfolioComparisonResponse:
    """
    Compare portfolio strategies using the locked portfolio metrics.

    Actual portfolio construction and evaluation belong to the
    research/portfolio layer.
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
            "Portfolio comparison engine is not connected yet."
        ),
    )