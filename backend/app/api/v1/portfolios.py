from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.portfolio import (
    PortfolioComparisonResponse,
    PortfolioRequest,
    PortfolioResponse,
)
from backend.app.storage import (
    RecordExistsError,
    storage,
)


router = APIRouter(
    prefix="/portfolios",
    tags=["Portfolios"],
)


def _validate_request(
    request: PortfolioRequest,
) -> None:
    """Validate portfolio request constraints."""

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


def _portfolio_response(
    record: dict,
) -> PortfolioResponse:
    """Convert a persistent portfolio record to an API response."""

    return PortfolioResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        strategy=record["strategy"],
        positions=record["positions"],
        observation_count=record["observation_count"],
        asset_count=record["asset_count"],
        status=record["status"],
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

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "strategy": request.strategy,
        "positions": [],
        "observation_count": 0,
        "asset_count": len(request.assets),
        "status": "accepted",
    }

    try:
        storage.save_portfolio(record)
    except RecordExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _portfolio_response(record)


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

    if strategy is not None:
        record = storage.get_portfolio(
            experiment_name,
            strategy,
        )

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No portfolio analysis found for experiment "
                    f"'{experiment_name}' and strategy '{strategy}'."
                ),
            )

        return _portfolio_response(record)

    records = storage.list_portfolios(
        experiment_name,
    )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No portfolio analysis found for experiment "
                f"'{experiment_name}'."
            ),
        )

    if len(records) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple portfolio strategies exist for "
                f"experiment '{experiment_name}'. Supply the "
                f"'strategy' query parameter."
            ),
        )

    return _portfolio_response(records[0])


@router.post(
    "/compare",
    response_model=PortfolioComparisonResponse,
)
def compare_portfolios(
    request: PortfolioRequest,
) -> PortfolioComparisonResponse:
    """
    Compare registered portfolio strategies.

    Actual portfolio construction and performance calculations
    belong to the portfolio research layer.
    """

    _validate_request(request)

    experiment_results = storage.list_portfolios(
        request.experiment_name,
    )

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
        if result["positions"]
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