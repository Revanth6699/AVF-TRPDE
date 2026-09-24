from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.configuration import parse_experiment_config
from backend.app.experiments.runner import run_experiments
from backend.app.schemas.experiment import (
    ExperimentListResponse,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExperimentRunResultResponse,
)


router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"],
)


# API-level in-memory registry.
# Persistent experiment artifacts remain the responsibility
# of the experiment/research layer.
_EXPERIMENTS: dict[str, ExperimentResponse] = {}


def _model_dump(model: Any) -> dict[str, Any]:
    """Serialize a Pydantic model across supported Pydantic versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _build_experiment_response(
    request: ExperimentRequest,
    *,
    status_value: str = "configured",
) -> ExperimentResponse:
    """Build the API representation of an experiment."""

    return ExperimentResponse(
        name=request.name,
        description=request.description,
        models=request.models,
        metrics=request.metrics,
        risk_measures=request.risk_measures,
        parameters=request.parameters,
        metadata=request.metadata,
        status=status_value,
    )


@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    request: ExperimentRequest,
) -> ExperimentResponse:
    """
    Validate and register an experiment configuration.

    The research configuration parser remains the source of truth
    for experiment configuration validation.
    """

    if request.name in _EXPERIMENTS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{request.name}' already exists.",
        )

    raw_config = _model_dump(request)

    try:
        parse_experiment_config(raw_config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    response = _build_experiment_response(request)

    _EXPERIMENTS[request.name] = response

    return response


@router.get(
    "",
    response_model=ExperimentListResponse,
)
def list_experiments() -> ExperimentListResponse:
    """Return all registered experiment configurations."""

    experiments = list(_EXPERIMENTS.values())

    return ExperimentListResponse(
        experiments=experiments,
        count=len(experiments),
    )


@router.get(
    "/{experiment_name}",
    response_model=ExperimentResponse,
)
def get_experiment(
    experiment_name: str,
) -> ExperimentResponse:
    """Return one registered experiment configuration."""

    experiment = _EXPERIMENTS.get(experiment_name)

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_name}' was not found.",
        )

    return experiment


@router.post(
    "/run",
    response_model=ExperimentRunResponse,
)
def run_experiment(
    request: ExperimentRunRequest,
) -> ExperimentRunResponse:
    """
    Execute one or more experiment configurations.

    Configuration parsing and experiment execution are delegated to
    the existing research-layer modules.
    """

    if not request.experiments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one experiment is required.",
        )

    configurations = []

    for experiment_request in request.experiments:
        raw_config = _model_dump(experiment_request)

        try:
            configuration = parse_experiment_config(raw_config)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid experiment "
                    f"'{experiment_request.name}': {exc}"
                ),
            ) from exc

        configurations.append(configuration)

    run_results = []

    for configuration in configurations:
        try:
            result = run_experiments(
                configuration,
                handlers=_build_handlers(),
            )

        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid experiment execution configuration: "
                    f"{exc}"
                ),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Experiment execution failed: {exc}",
            ) from exc

        run_results.append(result)

    response_results: list[ExperimentRunResultResponse] = []

    successful_count = 0
    failed_count = 0

    for result in run_results:
        successful_count += result.successful_count
        failed_count += result.failed_count

        for item in result.results:
            response_results.append(
                ExperimentRunResultResponse(
                    name=item.name,
                    status=item.status,
                    result=item.result,
                    error=item.error,
                )
            )

    return ExperimentRunResponse(
        experiment_count=len(run_results),
        successful_count=successful_count,
        failed_count=failed_count,
        all_successful=failed_count == 0,
        results=response_results,
    )


def _build_handlers() -> dict[str, Any]:
    """
    Return the handlers currently exposed by the existing
    experiment runner.

    The API layer does not implement forecasting/risk/portfolio
    model logic. Those implementations belong to their respective
    research modules.
    """

    return {
        "baseline": _completed_handler,
        "regime_xgboost": _completed_handler,
    }


def _completed_handler(experiment: Any) -> dict[str, Any]:
    """
    Adapter for the currently implemented experiment runner.

    This preserves the existing research-layer contract without
    introducing new model logic into the API.
    """

    return {
        "models": tuple(experiment.models),
        "status": "completed",
    }