from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.artifacts import (
    ExperimentArtifactError,
    load_experiment_metadata,
    load_experiment_results,
)
from backend.app.risk.christoffersen import christoffersen_test
from backend.app.risk.kupiec import kupiec_unconditional_coverage
from backend.app.risk.pipeline import calculate_fhs_risk_points
from backend.app.schemas.risk import (
    RiskBacktestResponse,
    RiskBacktestResult,
    RiskComparisonItem,
    RiskComparisonResponse,
    RiskRequest,
    RiskResponse,
)
from backend.app.storage import storage


router = APIRouter(
    prefix="/risk",
    tags=["Risk"],
)

DEFAULT_FHS_MIN_OBSERVATIONS = 100
DEFAULT_FHS_SIMULATIONS = 10_000
DEFAULT_FHS_RANDOM_STATE = 42


def _normalize_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Risk date boundaries must be valid timestamps.",
        )
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _validate_request(
    request: RiskRequest,
    *,
    require_model: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate request constraints not handled by Pydantic."""
    start = _normalize_timestamp(request.start_timestamp)
    end = _normalize_timestamp(request.end_timestamp)

    if start > end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    if not request.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one asset is required.",
        )

    normalized_assets = [str(asset).strip().upper() for asset in request.assets]
    if any(not asset for asset in normalized_assets):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Asset identifiers must not be empty.",
        )
    if len(set(normalized_assets)) != len(normalized_assets):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Asset identifiers must not contain duplicates.",
        )

    levels = [round(float(level), 8) for level in request.confidence_levels]
    if sorted(levels) != [0.95, 0.99] or len(levels) != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The locked risk workflow requires exactly confidence levels 0.95 and 0.99.",
        )

    if require_model and not request.model:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="model is required for risk generation and backtesting.",
        )

    return start, end


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


def _load_experiment_artifacts(
    request: RiskRequest,
) -> tuple[dict, pd.DataFrame]:
    try:
        metadata = load_experiment_metadata(request.experiment_name)
        predictions = load_experiment_results(request.experiment_name)
    except (ExperimentArtifactError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No complete prediction artifacts were found for experiment "
                f"'{request.experiment_name}'. Run the experiment first."
            ),
        ) from exc

    if metadata.get("dataset_name") != request.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="dataset_name does not match the experiment prediction artifact.",
        )

    if "model" not in predictions.columns:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Prediction artifact has no model identifier. Re-run the experiment.",
        )

    model_predictions = predictions.loc[
        predictions["model"].astype(str) == str(request.model)
    ].copy()
    if model_predictions.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No out-of-sample predictions were found for model "
                f"'{request.model}' in experiment '{request.experiment_name}'."
            ),
        )

    return metadata, model_predictions


def _filter_risk_points(
    record: dict,
    request: RiskRequest,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if record["dataset_name"] != request.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="dataset_name does not match the stored risk analysis.",
        )

    frame = pd.DataFrame(record["risk_points"])
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Risk analysis contains no observations.",
        )

    required = {
        "timestamp",
        "asset_id",
        "var_95",
        "var_99",
        "es_95",
        "actual_return",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Stored risk result is missing backtesting fields: "
                + ", ".join(missing)
            ),
        )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored risk result contains invalid timestamps.",
        )
    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)

    frame["asset_id"] = frame["asset_id"].astype(str).str.strip().str.upper()
    frame = frame.loc[
        frame["asset_id"].isin(
            {str(asset).strip().upper() for asset in request.assets}
        )
        & frame["timestamp"].between(start, end, inclusive="both")
    ].copy()

    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No stored risk observations match the requested assets and date range.",
        )

    return frame.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


@router.post(
    "",
    response_model=RiskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_risk_analysis(
    request: RiskRequest,
) -> RiskResponse:
    """Generate and persist expanding-window FHS VaR/ES forecasts."""
    start, end = _validate_request(request, require_model=True)
    _, predictions = _load_experiment_artifacts(request)

    try:
        risk_points = calculate_fhs_risk_points(
            predictions,
            assets=request.assets,
            start_timestamp=start,
            end_timestamp=end,
            min_observations=DEFAULT_FHS_MIN_OBSERVATIONS,
            n_simulations=DEFAULT_FHS_SIMULATIONS,
            random_state=DEFAULT_FHS_RANDOM_STATE,
        )
    except ValueError as exc:
        message = str(exc)
        code = (
            status.HTTP_409_CONFLICT
            if "No risk points were produced" in message
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=code, detail=message) from exc

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "model": request.model,
        "risk_points": risk_points,
        "observation_count": len(risk_points),
        "asset_count": len({point["asset_id"] for point in risk_points}),
        "status": "completed",
    }
    storage.upsert_risk(record)
    return _risk_response(record)


@router.get(
    "/{experiment_name}",
    response_model=RiskResponse,
)
def get_risk_analysis(
    experiment_name: str,
    model: str | None = None,
) -> RiskResponse:
    """Retrieve a completed persistent risk result."""
    if model is not None:
        record = storage.get_risk(experiment_name, model)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No risk analysis found for experiment '{experiment_name}' "
                    f"and model '{model}'."
                ),
            )
        return _risk_response(record)

    records = storage.list_risks(experiment_name)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No risk analysis found for experiment '{experiment_name}'.",
        )
    if len(records) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple risk models exist for experiment '{experiment_name}'. "
                "Supply the 'model' query parameter."
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
    """Run Kupiec and Christoffersen VaR backtests per asset."""
    start, end = _validate_request(request, require_model=True)
    record = storage.get_risk(request.experiment_name, request.model)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No risk analysis registered for experiment "
                f"'{request.experiment_name}' and model '{request.model}'."
            ),
        )
    if record["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Risk analysis is not in completed status.",
        )

    frame = _filter_risk_points(record, request, start, end)
    results: list[RiskBacktestResult] = []

    for asset_id, asset_frame in frame.groupby("asset_id", sort=True):
        if len(asset_frame) < 2:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"At least two risk observations are required for "
                    f"Christoffersen backtesting of asset '{asset_id}'."
                ),
            )
        actual_returns = asset_frame["actual_return"].to_numpy(dtype=float)

        for confidence_level, column in (
            (0.95, "var_95"),
            (0.99, "var_99"),
        ):
            var_losses = asset_frame[column].to_numpy(dtype=float)
            try:
                kupiec = kupiec_unconditional_coverage(
                    actual_returns,
                    var_losses,
                    confidence_level,
                )
                christoffersen = christoffersen_test(
                    actual_returns,
                    var_losses,
                    confidence_level,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"VaR backtesting failed for asset '{asset_id}': {exc}",
                ) from exc

            results.append(
                RiskBacktestResult(
                    model=request.model,
                    asset_id=str(asset_id),
                    confidence_level=confidence_level,
                    measure=f"VaR{int(confidence_level * 100)}",
                    observations=kupiec.observation_count,
                    violations=kupiec.exception_count,
                    violation_rate=kupiec.observed_exception_rate,
                    statistic=kupiec.lr_statistic,
                    p_value=kupiec.p_value,
                    independence_statistic=christoffersen.lr_independence,
                    independence_p_value=christoffersen.independence_p_value,
                    conditional_coverage_statistic=christoffersen.lr_conditional_coverage,
                    conditional_coverage_p_value=christoffersen.conditional_coverage_p_value,
                )
            )

    return RiskBacktestResponse(
        experiment_name=request.experiment_name,
        results=results,
        observation_count=len(frame),
        asset_count=frame["asset_id"].nunique(),
    )


@router.post(
    "/compare",
    response_model=RiskComparisonResponse,
)
def compare_risk(
    request: RiskRequest,
) -> RiskComparisonResponse:
    """Compare mean VaR/ES forecasts on common asset/date observations."""
    start, end = _validate_request(request)
    records = storage.list_risks(request.experiment_name)
    records = [
        record
        for record in records
        if record["dataset_name"] == request.dataset_name
        and record["status"] == "completed"
        and record["risk_points"]
    ]
    if len(records) < 2:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "At least two completed model risk analyses for the requested "
                "experiment and dataset are required for comparison."
            ),
        )

    aligned: pd.DataFrame | None = None
    model_names: list[str] = []
    for record in sorted(records, key=lambda item: item["model"]):
        frame = _filter_risk_points(record, request, start, end)
        if frame.duplicated(["asset_id", "timestamp"]).any():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate risk observations found for model '{record['model']}'.",
            )
        model = str(record["model"])
        if model in model_names:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate stored risk records found for model '{model}'.",
            )
        model_names.append(model)

        current = frame.loc[
            :, ["asset_id", "timestamp", "var_95", "var_99", "es_95"]
        ].rename(
            columns={
                "var_95": f"var_95__{model}",
                "var_99": f"var_99__{model}",
                "es_95": f"es_95__{model}",
            }
        )
        aligned = (
            current
            if aligned is None
            else aligned.merge(
                current,
                on=["asset_id", "timestamp"],
                how="inner",
                validate="one_to_one",
            )
        )

    if aligned is None or aligned.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected models have no common risk observations to compare.",
        )

    comparison_items = []
    for model in model_names:
        comparison_items.append(
            RiskComparisonItem(
                model=model,
                var_95=float(aligned[f"var_95__{model}"].mean()),
                var_99=float(aligned[f"var_99__{model}"].mean()),
                es_95=float(aligned[f"es_95__{model}"].mean()),
            )
        )

    return RiskComparisonResponse(
        experiment_name=request.experiment_name,
        models=comparison_items,
        observation_count=len(aligned),
        asset_count=aligned["asset_id"].nunique(),
    )
