from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException

from backend.app.schemas.stress import StressRunRequest, StressRunResponse
from backend.app.storage import RecordExistsError, storage
from backend.app.stress.analysis import analyze_stress_results, summarize_stress_analysis
from backend.app.stress.runner import run_stress_scenarios
from backend.app.stress.scenarios import build_stress_scenarios


router = APIRouter(prefix="/stress", tags=["Stress Testing"])


def _config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[4] / "configs" / "stress.yaml"
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=500, detail="Stress configuration could not be loaded.") from exc
    if not isinstance(config, dict) or not isinstance(config.get("scenarios"), dict):
        raise HTTPException(status_code=500, detail="Stress configuration has an invalid structure.")
    return config


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.replace({np.nan: None}).to_dict(orient="records")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
            elif isinstance(value, np.generic):
                row[key] = value.item()
    return rows


@router.post("/run", response_model=StressRunResponse)
def run_stress(request: StressRunRequest) -> StressRunResponse:
    """Build, execute, analyze, and persist stress scenarios for a completed portfolio."""
    portfolio = storage.get_portfolio(request.experiment_name, request.strategy)
    if portfolio is None:
        raise HTTPException(
            status_code=404,
            detail="Completed portfolio not found. Execute POST /portfolios first.",
        )
    series = pd.DataFrame(portfolio.get("return_series", []))
    if series.empty or not {"timestamp", "net_return"} <= set(series.columns):
        raise HTTPException(status_code=409, detail="Portfolio has no persisted net-return series.")
    series["timestamp"] = pd.to_datetime(series["timestamp"], errors="coerce")
    series["return"] = pd.to_numeric(series["net_return"], errors="coerce")
    if series[["timestamp", "return"]].isna().any().any():
        raise HTTPException(status_code=409, detail="Stored portfolio return series contains invalid values.")
    config = _config()
    minimum = int(config.get("minimum_observations", 20))
    if len(series) < minimum:
        raise HTTPException(
            status_code=422,
            detail=f"Stress analysis requires at least {minimum} portfolio observations; found {len(series)}.",
        )
    window = int(config.get("volatility_window", 20))
    series = series.sort_values("timestamp").reset_index(drop=True)
    series["volatility"] = series["return"].rolling(window, min_periods=2).std().bfill()
    if series["volatility"].isna().any() or not np.isfinite(series["volatility"]).all():
        raise HTTPException(status_code=422, detail="Could not derive finite historical portfolio volatility.")
    stress_frame = series.assign(asset_id="PORTFOLIO")[
        ["timestamp", "asset_id", "return", "volatility"]
    ]
    scenario_config = config["scenarios"]
    try:
        scenarios = build_stress_scenarios(
            stress_frame,
            normal_lower_percentile=float(scenario_config["normal_volatility"]["lower_percentile"]),
            normal_upper_percentile=float(scenario_config["normal_volatility"]["upper_percentile"]),
            high_volatility_percentile=float(scenario_config["high_volatility"]["percentile"]),
            volatility_spike_percentile=float(scenario_config["volatility_spike"]["percentile"]),
            minimum_drawdown=float(scenario_config["prolonged_drawdown"]["minimum_drawdown"]),
            minimum_drawdown_duration=int(scenario_config["prolonged_drawdown"]["minimum_duration"]),
        )
        run = run_stress_scenarios(scenarios)
        analysis = analyze_stress_results(run.results)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Stress execution failed: {exc}") from exc

    results = _records(run.results)
    summary = {
        "scenario_summary": _records(summarize_stress_analysis(analysis)),
        "valid_scenario_count": analysis.valid_scenario_count,
        "volatility_window": window,
        "annualization_periods": int(config.get("annualization_periods", 252)),
        "method": "retrospective descriptive scenario selection on the persisted portfolio net-return series",
    }
    record = {
        "experiment_name": request.experiment_name,
        "strategy": request.strategy,
        "results": results,
        "summary": summary,
        "status": "completed",
    }
    try:
        storage.save_stress_result(record)
    except RecordExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StressRunResponse(
        experiment_name=request.experiment_name,
        strategy=request.strategy,
        status="completed",
        scenario_count=run.scenario_count,
        total_observations=run.total_observations,
        results=results,
        summary=summary,
    )


@router.get("/{experiment_name}", response_model=list[StressRunResponse])
def get_stress_results(experiment_name: str, strategy: str | None = None) -> list[StressRunResponse]:
    if strategy is not None:
        records = [storage.get_stress_result(experiment_name, strategy)]
        records = [record for record in records if record is not None]
    else:
        records = storage.list_stress_results(experiment_name)
    return [
        StressRunResponse(
            experiment_name=record["experiment_name"],
            strategy=record["strategy"],
            status=record["status"],
            scenario_count=len(record["results"]),
            total_observations=sum(
                int(row.get("observation_count") or 0) for row in record["results"]
            ),
            results=record["results"],
            summary=record["summary"],
        )
        for record in records
    ]
