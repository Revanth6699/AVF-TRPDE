from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.app.portfolio.engine import PortfolioExecutionError, execute_portfolio
from backend.app.schemas.portfolio import (
    PortfolioComparisonItem,
    PortfolioComparisonResponse,
    PortfolioRequest,
    PortfolioResponse,
)
from backend.app.storage import RecordExistsError, storage


router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


def _normalized_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise HTTPException(status_code=422, detail="Date boundaries must be valid timestamps.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _validate_request(request: PortfolioRequest) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    start = _normalized_timestamp(request.start_timestamp)
    end = _normalized_timestamp(request.end_timestamp)
    if start > end:
        raise HTTPException(status_code=400, detail="start_timestamp must not be after end_timestamp.")
    assets = [str(asset).strip().upper() for asset in request.assets]
    if any(not asset for asset in assets) or len(set(assets)) != len(assets):
        raise HTTPException(status_code=422, detail="Asset identifiers must be non-empty and unique.")
    if request.strategy not in {"static", "buy_and_hold", "risk_targeted"}:
        raise HTTPException(
            status_code=422,
            detail="strategy must be one of: static, buy_and_hold, risk_targeted.",
        )
    if request.max_turnover <= 0:
        raise HTTPException(status_code=422, detail="max_turnover must be greater than zero.")
    return start, end, assets


def _response(record: dict) -> PortfolioResponse:
    return PortfolioResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        strategy=record["strategy"],
        positions=record["positions"],
        return_series=record.get("return_series", []),
        performance=record.get("performance", {}),
        observation_count=record["observation_count"],
        asset_count=record["asset_count"],
        status=record["status"],
    )


@router.post("", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
def create_portfolio_analysis(request: PortfolioRequest) -> PortfolioResponse:
    """Execute and persist a portfolio using a selected model's OOS forecasts."""
    start, end, assets = _validate_request(request)
    experiment = storage.get_experiment(request.experiment_name)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{request.experiment_name}' was not found.")
    if request.dataset_name not in {item["name"] for item in storage.list_datasets()}:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' is not registered.")

    forecast_record = storage.get_forecast(request.experiment_name, request.model)
    if forecast_record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No persisted forecasts exist for model '{request.model}'. Run POST /forecasts first.",
        )
    if forecast_record["dataset_name"] != request.dataset_name:
        raise HTTPException(status_code=409, detail="dataset_name does not match the stored forecast dataset.")

    frame = pd.DataFrame(forecast_record["forecasts"])
    required = {"timestamp", "asset_id", "actual_return", "predicted_volatility"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=409,
            detail="Stored forecast records are missing: " + ", ".join(missing),
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["asset_id"] = frame["asset_id"].astype(str).str.strip().str.upper()
    frame = frame.loc[
        frame["asset_id"].isin(assets)
        & frame["timestamp"].between(start, end, inclusive="both")
    ].copy()
    if frame.empty:
        raise HTTPException(status_code=409, detail="No forecast observations match the requested assets and date range.")
    observed_assets = set(frame["asset_id"].unique())
    if observed_assets != set(assets):
        raise HTTPException(
            status_code=409,
            detail="Stored forecasts do not cover every requested asset in the selected date range.",
        )

    try:
        execution = execute_portfolio(
            frame,
            strategy=request.strategy,
            volatility_target=request.volatility_target,
            max_position=request.max_position,
            max_turnover=request.max_turnover,
            transaction_cost=request.transaction_cost,
            covariance_window=request.covariance_window,
        )
    except (PortfolioExecutionError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "strategy": request.strategy,
        "positions": execution.positions.to_dict(orient="records"),
        "return_series": execution.returns.to_dict(orient="records"),
        "performance": execution.metrics,
        "observation_count": execution.observation_count,
        "asset_count": execution.asset_count,
        "status": "completed",
    }
    try:
        storage.save_portfolio(record)
    except RecordExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _response(record)


@router.get("/{experiment_name}", response_model=PortfolioResponse)
def get_portfolio_analysis(
    experiment_name: str,
    strategy: str | None = None,
) -> PortfolioResponse:
    if strategy is not None:
        record = storage.get_portfolio(experiment_name, strategy)
        if record is None:
            raise HTTPException(status_code=404, detail="Portfolio analysis was not found.")
        return _response(record)
    records = storage.list_portfolios(experiment_name)
    if not records:
        raise HTTPException(status_code=404, detail="No portfolio analyses were found.")
    if len(records) > 1:
        raise HTTPException(status_code=409, detail="Multiple strategies exist; supply the strategy query parameter.")
    return _response(records[0])


@router.post("/compare", response_model=PortfolioComparisonResponse)
def compare_portfolios(request: PortfolioRequest) -> PortfolioComparisonResponse:
    """Compare persisted portfolio return series on their common dates."""
    _, _, _ = _validate_request(request)
    records = storage.list_portfolios(request.experiment_name)
    if not records:
        raise HTTPException(status_code=404, detail="No completed portfolio analyses were found.")
    frames: dict[str, pd.DataFrame] = {}
    for record in records:
        series = pd.DataFrame(record.get("return_series", []))
        if series.empty or not {"timestamp", "net_return", "gross_return"} <= set(series.columns):
            continue
        series["timestamp"] = pd.to_datetime(series["timestamp"], errors="coerce")
        series = series.loc[series["timestamp"].between(
            _normalized_timestamp(request.start_timestamp),
            _normalized_timestamp(request.end_timestamp),
            inclusive="both",
        )]
        frames[record["strategy"]] = series.set_index("timestamp")
    if not frames:
        raise HTTPException(status_code=409, detail="Stored portfolios have no return series in the requested date range.")
    common = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not common:
        raise HTTPException(status_code=409, detail="Stored strategies have no common evaluation dates.")
    common_dates = sorted(common)
    items = []
    for strategy, frame in frames.items():
        aligned = frame.loc[common_dates].sort_index()
        returns = aligned["net_return"].astype(float)
        wealth = float((1.0 + returns).prod())
        losses = -returns.to_numpy(dtype=float)
        var95 = float(pd.Series(losses).quantile(0.95))
        var99 = float(pd.Series(losses).quantile(0.99))
        tail = losses[losses >= var95]
        from backend.app.portfolio.portfolio import calculate_performance
        performance = calculate_performance(returns)
        items.append(
            PortfolioComparisonItem(
                strategy=strategy,
                total_return=performance.total_return,
                volatility=performance.annualized_volatility,
                sharpe=performance.sharpe_ratio,
                sortino=performance.sortino_ratio,
                max_drawdown=performance.max_drawdown,
                calmar=performance.calmar_ratio,
                var_95=var95,
                var_99=var99,
                es_95=float(tail.mean()),
                turnover=float(aligned["turnover"].sum()),
                transaction_costs=float(aligned["transaction_cost"].sum()),
                gross_pnl=float((1.0 + aligned["gross_return"]).prod() - 1.0),
                net_pnl=wealth - 1.0,
            )
        )
    return PortfolioComparisonResponse(
        experiment_name=request.experiment_name,
        strategies=items,
        observation_count=len(common_dates),
        asset_count=max(record["asset_count"] for record in records),
    )
