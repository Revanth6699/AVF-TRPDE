from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.artifacts import (
    ExperimentArtifactError,
    load_experiment_metadata,
    load_experiment_results,
)
from backend.app.research.evaluation.forecast_metrics import (
    ForecastMetricError,
    calculate_forecast_metrics,
)
from backend.app.schemas.forecast import (
    ForecastComparisonItem,
    ForecastComparisonResponse,
    ForecastEvaluationResponse,
    ForecastMetric,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)
from backend.app.storage import storage


router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
)


def _normalize_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Forecast date boundaries must be valid timestamps.",
        )
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _validate_request(
    request: ForecastRequest,
    *,
    require_model: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _normalize_timestamp(request.start_timestamp)
    end = _normalize_timestamp(request.end_timestamp)

    if start > end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_timestamp must be earlier than or equal to end_timestamp.",
        )

    assets = [str(asset).strip().upper() for asset in request.assets]
    if any(not asset for asset in assets):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Asset identifiers must not be empty.",
        )
    if len(set(assets)) != len(assets):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Asset identifiers must not contain duplicates.",
        )
    if require_model and not request.model:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="model is required for forecast creation and evaluation.",
        )

    return start, end


def _load_model_predictions(
    request: ForecastRequest,
) -> pd.DataFrame:
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

    required = {
        "timestamp",
        "target_timestamp",
        "asset_id",
        "target_return",
        "actual_volatility",
        "prediction",
        "model",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Prediction artifact is missing forecast contract fields: "
                + ", ".join(missing)
                + ". Re-run the experiment with the current backend."
            ),
        )

    model_frame = predictions.loc[
        predictions["model"].astype(str) == str(request.model)
    ].copy()
    if model_frame.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No out-of-sample predictions were found for model "
                f"'{request.model}' in experiment '{request.experiment_name}'."
            ),
        )

    model_frame["timestamp"] = pd.to_datetime(
        model_frame["timestamp"],
        errors="coerce",
    )
    model_frame["target_timestamp"] = pd.to_datetime(
        model_frame["target_timestamp"],
        errors="coerce",
    )
    if model_frame[["timestamp", "target_timestamp"]].isna().any().any():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Prediction artifact contains invalid forecast timestamps.",
        )

    if model_frame["timestamp"].dt.tz is not None:
        model_frame["timestamp"] = model_frame["timestamp"].dt.tz_localize(None)
    if model_frame["target_timestamp"].dt.tz is not None:
        model_frame["target_timestamp"] = model_frame[
            "target_timestamp"
        ].dt.tz_localize(None)

    model_frame["asset_id"] = (
        model_frame["asset_id"].astype(str).str.strip().str.upper()
    )
    model_frame = model_frame.loc[
        model_frame["asset_id"].isin(
            {str(asset).strip().upper() for asset in request.assets}
        )
    ].copy()
    start, end = _normalize_timestamp(request.start_timestamp), _normalize_timestamp(
        request.end_timestamp
    )
    model_frame = model_frame.loc[
        model_frame["target_timestamp"].between(start, end, inclusive="both")
    ].copy()

    if model_frame.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No forecast observations match the requested assets and target-date range.",
        )

    numeric = model_frame[
        ["target_return", "actual_volatility", "prediction"]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Prediction artifact contains missing or non-finite forecast values.",
        )
    if (numeric["actual_volatility"] <= 0).any() or (numeric["prediction"] <= 0).any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Actual and predicted volatility must be strictly positive for QLIKE.",
        )
    model_frame[["target_return", "actual_volatility", "prediction"]] = numeric

    if model_frame.duplicated(["asset_id", "target_timestamp"]).any():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate asset/target-date forecast observations were found.",
        )

    return model_frame.sort_values(
        ["asset_id", "target_timestamp"]
    ).reset_index(drop=True)


def _forecast_response(record: dict) -> ForecastResponse:
    return ForecastResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        model=record["model"],
        forecasts=record["forecasts"],
        observation_count=record["observation_count"],
        asset_count=record["asset_count"],
        status=record["status"],
    )


def _filtered_forecasts(
    record: dict,
    request: ForecastRequest,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if record["dataset_name"] != request.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="dataset_name does not match the stored forecast.",
        )
    frame = pd.DataFrame(record["forecasts"])
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored forecast contains no observations.",
        )
    required = {
        "timestamp",
        "origin_timestamp",
        "asset_id",
        "actual_return",
        "actual_volatility",
        "predicted_volatility",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored forecast is missing required fields: " + ", ".join(missing),
        )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored forecast contains invalid target timestamps.",
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
            detail="No stored forecast observations match the requested assets and date range.",
        )
    return frame.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    """Persist a selected model's out-of-sample experiment forecasts."""
    start, end = _validate_request(request, require_model=True)
    model_frame = _load_model_predictions(request)

    forecasts = [
        {
            "timestamp": row.target_timestamp,
            "origin_timestamp": row.timestamp,
            "asset_id": row.asset_id,
            "actual_return": float(row.target_return),
            "actual_volatility": float(row.actual_volatility),
            "predicted_volatility": float(row.prediction),
        }
        for row in model_frame.itertuples(index=False)
        if start <= row.target_timestamp <= end
    ]
    if not forecasts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No forecasts remain after applying the requested target-date range.",
        )

    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "model": request.model,
        "forecasts": forecasts,
        "observation_count": len(forecasts),
        "asset_count": len({point["asset_id"] for point in forecasts}),
        "status": "completed",
    }
    storage.upsert_forecast(record)
    return _forecast_response(record)


@router.get(
    "/{experiment_name}",
    response_model=ForecastResponse,
)
def get_forecast(
    experiment_name: str,
    model: str | None = None,
) -> ForecastResponse:
    """Retrieve a completed persisted forecast."""
    if model is not None:
        record = storage.get_forecast(experiment_name, model)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No forecast found for experiment '{experiment_name}' and model '{model}'.",
            )
        return _forecast_response(record)

    records = storage.list_forecasts(experiment_name)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No forecast found for experiment '{experiment_name}'.",
        )
    if len(records) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Multiple forecast models exist for experiment '{experiment_name}'. "
                "Supply the 'model' query parameter."
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
    """Calculate MAE, RMSE, and variance-based QLIKE per asset."""
    start, end = _validate_request(request, require_model=True)
    record = storage.get_forecast(request.experiment_name, request.model)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No forecast registered for experiment '{request.experiment_name}' and model '{request.model}'.",
        )

    frame = _filtered_forecasts(record, request, start, end)
    metrics: list[ForecastMetric] = []
    for asset_id, asset_frame in frame.groupby("asset_id", sort=True):
        try:
            result = calculate_forecast_metrics(
                asset_frame["actual_volatility"].to_numpy(dtype=float),
                asset_frame["predicted_volatility"].to_numpy(dtype=float),
            )
        except ForecastMetricError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Forecast evaluation failed for asset '{asset_id}': {exc}",
            ) from exc

        for metric_name, value in result.as_dict().items():
            metrics.append(
                ForecastMetric(
                    model=request.model,
                    asset_id=str(asset_id),
                    metric=metric_name.upper(),
                    value=value,
                )
            )

    return ForecastEvaluationResponse(
        experiment_name=request.experiment_name,
        metrics=metrics,
        observation_count=len(frame),
        asset_count=frame["asset_id"].nunique(),
    )


@router.post(
    "/compare",
    response_model=ForecastComparisonResponse,
)
def compare_forecasts(
    request: ForecastRequest,
) -> ForecastComparisonResponse:
    """Compare forecast models per asset on common target-date observations."""
    start, end = _validate_request(request)
    if request.model is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Omit model when comparing all registered models for an experiment.",
        )

    records = [
        record
        for record in storage.list_forecasts(request.experiment_name)
        if record["dataset_name"] == request.dataset_name
        and record["status"] == "completed"
        and record["forecasts"]
    ]
    if len(records) < 2:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="At least two completed model forecasts are required for comparison.",
        )

    aligned: pd.DataFrame | None = None
    model_names: list[str] = []
    for record in sorted(records, key=lambda item: item["model"]):
        frame = _filtered_forecasts(record, request, start, end)
        model = str(record["model"])
        if model in model_names:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate stored forecast records found for model '{model}'.",
            )
        model_names.append(model)

        current = frame.loc[
            :, ["asset_id", "timestamp", "actual_volatility", "predicted_volatility"]
        ].rename(
            columns={
                "actual_volatility": f"actual__{model}",
                "predicted_volatility": f"prediction__{model}",
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
            detail="The selected models have no common asset/date forecast observations.",
        )

    actual_columns = [f"actual__{model}" for model in model_names]
    actual_reference = aligned[actual_columns[0]].to_numpy(dtype=float)
    for column in actual_columns[1:]:
        if not np.allclose(
            actual_reference,
            aligned[column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Models do not share identical realized-volatility labels on common dates.",
            )

    comparison_items: list[ForecastComparisonItem] = []
    for model in model_names:
        prediction_column = f"prediction__{model}"
        for asset_id, asset_frame in aligned.groupby("asset_id", sort=True):
            actual_column = f"actual__{model}"
            try:
                metrics = calculate_forecast_metrics(
                    asset_frame[actual_column].to_numpy(dtype=float),
                    asset_frame[prediction_column].to_numpy(dtype=float),
                )
            except ForecastMetricError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Forecast comparison failed for model '{model}', asset '{asset_id}': {exc}",
                ) from exc
            comparison_items.append(
                ForecastComparisonItem(
                    model=model,
                    asset_id=str(asset_id),
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    qlike=metrics.qlike,
                )
            )

    return ForecastComparisonResponse(
        experiment_name=request.experiment_name,
        models=comparison_items,
        observation_count=len(aligned),
        asset_count=aligned["asset_id"].nunique(),
    )
