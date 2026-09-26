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


# API-level registry.
# Portfolio calculations remain in backend.app.portfolio.
_PORTFOLIO_RESULTS: dict[str, PortfolioResponse] = {}


def _portfolio_key(
    experiment_name: str,
    strategy: str,
) -> str:
    """Build a deterministic portfolio-result key."""

    return f"{experiment_name}:{strategy}"


def _validate_request(
    request: PortfolioRequest,
) -> None:
    """Validate constraints not handled by Pydantic."""

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


@router.post(
    "",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_portfolio_analysis(
    request: PortfolioRequest,
) -> PortfolioResponse:
    """
    Register a portfolio-analysis request.

    Portfolio construction, risk targeting, constraints, turnover,
    transaction costs, and performance calculations belong to the
    portfolio research layer.
    """

    _validate_request(request)

    key = _portfolio_key(
        request.experiment_name,
        request.strategy,
    )

    if key in _PORTFOLIO_RESULTS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Portfolio analysis for experiment "
                f"'{request.experiment_name}' and strategy "
                f"'{request.strategy}' already exists."
            ),
        )

    response = PortfolioResponse(
        experiment_name=request.experiment_name,
        dataset_name=request.dataset_name,
        strategy=request.strategy,
        positions=[],
        observation_count=0,
        asset_count=len(request.assets),
        status="accepted",
    )

    _PORTFOLIO_RESULTS[key] = response

    return response


@router.get(
    "/{experiment_name}",
    response_model=PortfolioResponse,
)
def get_portfolio_analysis(
    experiment_name: str,
    strategy: str | None = None,
) -> PortfolioResponse:
    """
    Retrieve a registered portfolio result.

    If multiple strategies exist for an experiment, the strategy
    must be supplied explicitly.
    """

    matches = [
        result
        for key, result in _PORTFOLIO_RESULTS.items()
        if key.startswith(f"{experiment_name}:")
    ]

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No portfolio analysis found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if strategy is not None:
        key = _portfolio_key(
            experiment_name,
            strategy,
        )

        result = _PORTFOLIO_RESULTS.get(key)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No portfolio analysis found for experiment "
                    f"'{experiment_name}' and strategy "
                    f"'{strategy}'."
                ),
            )

        return result

    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple portfolio strategies exist for "
                f"experiment '{experiment_name}'. Supply the "
                f"'strategy' query parameter."
            ),
        )

    return matches[0]


@router.post(
    "/compare",
    response_model=PortfolioComparisonResponse,
)
def compare_portfolios(
    request: PortfolioRequest,
) -> PortfolioComparisonResponse:
    """
    Compare registered portfolio strategies.

    The locked comparison metrics are supplied by the portfolio
    research layer once populated results are available.
    """

    _validate_request(request)

    experiment_results = [
        result
        for key, result in _PORTFOLIO_RESULTS.items()
        if key.startswith(f"{request.experiment_name}:")
    ]

    if not experiment_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No portfolio analyses registered for experiment "
                f"'{request.experiment_name}'."
            ),
        )

    populated = [
        result
        for result in experiment_results
        if result.positions
    ]

    if not populated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Registered portfolio analyses contain no "
                "observations. Run the portfolio pipeline before "
                "comparison."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Portfolio comparison requires the existing portfolio "
            "engine integration contract."
        ),
    )