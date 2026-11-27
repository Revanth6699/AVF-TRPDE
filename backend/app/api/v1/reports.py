from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.artifacts import ExperimentArtifactError, load_experiment_metadata
from backend.app.research.evaluation.forecast_metrics import ForecastMetricError, calculate_forecast_metrics
from backend.app.schemas.report import ReportArtifact, ReportGenerationResult, ReportRequest, ReportResponse, ReportRunResponse, ReportSection
from backend.app.storage import RecordExistsError, storage

router = APIRouter(prefix="/reports", tags=["Reports"])


def _validate_request(request: ReportRequest) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(request.start_timestamp)
    end = pd.Timestamp(request.end_timestamp)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    if start > end:
        raise HTTPException(status_code=400, detail="start_timestamp must not be after end_timestamp.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request.report_type):
        raise HTTPException(status_code=422, detail="report_type may contain only letters, numbers, underscores, and hyphens.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request.experiment_name):
        raise HTTPException(status_code=422, detail="experiment_name contains unsupported path characters.")
    experiment = storage.get_experiment(request.experiment_name)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{request.experiment_name}' was not found.")
    if storage.get_dataset(request.dataset_name) is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' was not found.")
    configured_dataset = experiment.get("parameters", {}).get("dataset_name")
    if configured_dataset and configured_dataset != request.dataset_name:
        raise HTTPException(status_code=409, detail="dataset_name does not match the registered experiment configuration.")
    return start, end


def _report_response(record: dict[str, Any]) -> ReportResponse:
    return ReportResponse(
        experiment_name=record["experiment_name"],
        dataset_name=record["dataset_name"],
        report_type=record["report_type"],
        generated_at=record["generated_at"],
        sections=record["sections"],
        status=record["status"],
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_sections(request: ReportRequest, start: pd.Timestamp, end: pd.Timestamp) -> list[ReportSection]:
    sections: list[ReportSection] = []
    experiment = storage.get_experiment(request.experiment_name)
    sections.append(ReportSection(
        title="Research configuration",
        content=(
            f"Experiment: {request.experiment_name}\nDataset: {request.dataset_name}\n"
            f"Configured models: {', '.join(experiment['models'])}\n"
            f"Requested evaluation period: {start.isoformat()} through {end.isoformat()}\n"
            f"Experiment status: {experiment['status']}"
        ),
        metrics={},
    ))

    if request.include_forecasts:
        lines: list[str] = []
        metrics_out: dict[str, float] = {}
        records = [item for item in storage.list_forecasts(request.experiment_name)
                   if item["dataset_name"] == request.dataset_name]
        for record in records:
            frame = pd.DataFrame(record["forecasts"])
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame.loc[frame["timestamp"].between(start, end, inclusive="both")]
            if frame.empty:
                continue
            lines.append(f"- {record['model']}: {len(frame)} observations across {frame['asset_id'].nunique()} assets")
            if {"actual_volatility", "predicted_volatility"} <= set(frame.columns):
                try:
                    values = calculate_forecast_metrics(
                        pd.to_numeric(frame["actual_volatility"], errors="coerce"),
                        pd.to_numeric(frame["predicted_volatility"], errors="coerce"),
                    ).as_dict()
                    for key, value in values.items():
                        metrics_out[f"{record['model']}_{key}"] = float(value)
                    lines.append(f"  MAE={values['mae']:.8g}, RMSE={values['rmse']:.8g}, QLIKE={values['qlike']:.8g}")
                except (ForecastMetricError, ValueError, TypeError):
                    lines.append("  Forecast metrics unavailable: stored values did not satisfy metric input requirements.")
        sections.append(ReportSection(
            title="Out-of-sample forecast evidence",
            content="\n".join(lines) if lines else "No persisted forecast results matched the selected dataset and evaluation period. Run the forecast persistence workflow first.",
            metrics=metrics_out,
        ))

    if request.include_statistical_tests:
        try:
            metadata = load_experiment_metadata(request.experiment_name)
            evaluation = metadata.get("evaluation", {})
            content = (
                "The following evaluation belongs to the complete experiment artifact; "
                "it is not recomputed for the report's selected date subset.\n\n"
                + json.dumps(_json_safe(evaluation), indent=2, sort_keys=True)
            ) if evaluation else "The experiment artifact contains no evaluation or statistical-test results."
        except (ExperimentArtifactError, OSError, ValueError):
            content = "Experiment evaluation artifacts were not found. No statistical conclusion is inferred."
        sections.append(ReportSection(title="Forecast evaluation and statistical tests", content=content, metrics={}))

    if request.include_risk:
        lines = []
        for record in storage.list_risks(request.experiment_name):
            if record["dataset_name"] != request.dataset_name:
                continue
            frame = pd.DataFrame(record["risk_points"])
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame.loc[frame["timestamp"].between(start, end, inclusive="both")]
            if not frame.empty:
                lines.append(
                    f"- {record['model']}: {len(frame)} risk observations; "
                    f"mean VaR95={pd.to_numeric(frame['var_95']).mean():.8g}; "
                    f"mean VaR99={pd.to_numeric(frame['var_99']).mean():.8g}; "
                    f"mean ES95={pd.to_numeric(frame['es_95']).mean():.8g}"
                )
        sections.append(ReportSection(
            title="Tail-risk estimates",
            content="\n".join(lines) if lines else "No persisted FHS VaR/ES observations matched the requested period. No risk values were generated for this report.",
            metrics={},
        ))

    if request.include_portfolio:
        lines = []
        for record in storage.list_portfolios(request.experiment_name):
            if record["dataset_name"] != request.dataset_name:
                continue
            frame = pd.DataFrame(record.get("return_series", []))
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame.loc[frame["timestamp"].between(start, end, inclusive="both")]
            if frame.empty:
                continue
            metrics = record.get("performance", {})
            selected_return = float((1.0 + frame["net_return"].astype(float)).prod() - 1.0)
            lines.append(
                f"- {record['strategy']} ({frame['forecast_model'].iloc[0] if 'forecast_model' in frame else 'model provenance unavailable'}): "
                f"{len(frame)} observations; stored full-run net return={metrics.get('net_pnl')}; "
                f"stored full-run max drawdown={metrics.get('max_drawdown')}; "
                f"selected-period net return={selected_return:.8g}"
            )
        sections.append(ReportSection(
            title="Portfolio results",
            content="\n".join(lines) if lines else "No completed portfolio result matched the requested period. Run the portfolio execution workflow first.",
            metrics={},
        ))

    if request.include_stress:
        records = storage.list_stress_results(request.experiment_name)
        lines = [
            f"- {record['strategy']}: status={record['status']}; scenarios={len(record['results'])}; "
            f"valid scenarios={record['summary'].get('valid_scenario_count')}"
            for record in records
        ]
        if lines:
            lines.append("Stress scenario results use the complete portfolio return series and are not restricted to the report date subset.")
        sections.append(ReportSection(
            title="Stress-test results",
            content="\n".join(lines) if lines else "No persisted stress-test results were found. Run POST /stress/run after portfolio execution.",
            metrics={},
        ))

    sections.append(ReportSection(
        title="Interpretation and limitations",
        content=(
            "This report summarizes persisted outputs only. It does not infer that a model is superior, "
            "does not convert missing results into zero, and does not claim statistical or economic "
            "significance unless the corresponding experiment evidence explicitly reports it. "
            "Portfolio and stress statistics are historical sample results, not guarantees of future performance."
        ),
        metrics={},
    ))
    return sections


def _write_artifacts(request: ReportRequest, generated_at: datetime, sections: list[ReportSection]) -> list[ReportArtifact]:
    root = Path(os.environ.get("AVF_REPORT_OUTPUT_DIR", Path(__file__).resolve().parents[4])).resolve()
    directory = root / "reports" / request.experiment_name
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{request.report_type}.json"
    markdown_path = directory / f"{request.report_type}.md"
    payload = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "report_type": request.report_type,
        "generated_at": generated_at.isoformat(),
        "period": {"start": request.start_timestamp.isoformat(), "end": request.end_timestamp.isoformat()},
        "sections": [section.model_dump() for section in sections],
    }
    json_temp = json_path.with_suffix(".json.tmp")
    markdown_temp = markdown_path.with_suffix(".md.tmp")
    with json_temp.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2, allow_nan=False)
    markdown = [
        f"# {request.report_type.replace('_', ' ').title()}",
        "",
        f"- Experiment: {request.experiment_name}",
        f"- Dataset: {request.dataset_name}",
        f"- Generated: {generated_at.isoformat()}",
        f"- Period: {request.start_timestamp.isoformat()} to {request.end_timestamp.isoformat()}",
        "",
    ]
    for section in sections:
        markdown.extend([f"## {section.title}", "", section.content, ""])
        if section.metrics:
            markdown.extend(["| Metric | Value |", "|---|---:|"])
            markdown.extend([f"| {key} | {value:.10g} |" for key, value in section.metrics.items()])
            markdown.append("")
    with markdown_temp.open("w", encoding="utf-8") as file:
        file.write("\n".join(markdown))
    json_temp.replace(json_path)
    markdown_temp.replace(markdown_path)
    return [
        ReportArtifact(path=str(json_path.relative_to(root)), artifact_type="json", size_bytes=json_path.stat().st_size),
        ReportArtifact(path=str(markdown_path.relative_to(root)), artifact_type="markdown", size_bytes=markdown_path.stat().st_size),
    ]


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
def create_report(request: ReportRequest) -> ReportResponse:
    _validate_request(request)
    if storage.get_report(request.experiment_name, request.report_type) is not None:
        raise HTTPException(status_code=409, detail="A report with this experiment and report_type already exists.")
    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "report_type": request.report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": [],
        "status": "requested",
    }
    try:
        storage.save_report(record)
    except RecordExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _report_response(record)


@router.get("/{experiment_name}", response_model=ReportResponse)
def get_report(experiment_name: str, report_type: str | None = None) -> ReportResponse:
    records = storage.list_reports(experiment_name)
    if report_type is not None:
        record = storage.get_report(experiment_name, report_type)
        if record is None:
            raise HTTPException(status_code=404, detail="Report was not found.")
        return _report_response(record)
    if not records:
        raise HTTPException(status_code=404, detail="No report was found for this experiment.")
    if len(records) > 1:
        raise HTTPException(status_code=409, detail="Multiple report types exist; supply report_type.")
    return _report_response(records[0])


@router.post("/run", response_model=ReportRunResponse)
def run_report(request: ReportRequest) -> ReportRunResponse:
    start, end = _validate_request(request)
    generated_at = datetime.now(timezone.utc)
    sections = _build_sections(request, start, end)
    try:
        artifacts = _write_artifacts(request, generated_at, sections)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Report artifact could not be written.") from exc
    record = {
        "experiment_name": request.experiment_name,
        "dataset_name": request.dataset_name,
        "report_type": request.report_type,
        "generated_at": generated_at.isoformat(),
        "sections": [section.model_dump() for section in sections],
        "status": "completed",
    }
    storage.upsert_report(record)
    return ReportRunResponse(
        experiment_name=request.experiment_name,
        status="completed",
        reports=[ReportGenerationResult(report_type=request.report_type, status="completed", artifacts=artifacts)],
        conclusion=None,
    )
