from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.configuration import (
    parse_experiment_config,
)
from backend.app.experiments.runner import run_experiments
from backend.app.schemas.experiment import (
    ExperimentListResponse,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExperimentRunResultResponse,
)
from backend.app.storage import (
    RecordExistsError,
    storage,
)


router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"],
)


def _model_dump(model: Any) -> dict[str, Any]:
    """Serialize a Pydantic model."""

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
    Validate and persist an experiment configuration.

    The research configuration parser remains the source of truth
    for experiment configuration validation.
    """

    raw_config = {
        "experiments": [
            _model_dump(request),
        ],
    }

    try:
        parse_experiment_config(raw_config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    response = _build_experiment_response(request)

    try:
        storage.save_experiment(
            _model_dump(response),
        )
    except RecordExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return response


@router.get(
    "",
    response_model=ExperimentListResponse,
)
def list_experiments() -> ExperimentListResponse:
    """Return all persistently registered experiments."""

    records = storage.list_experiments()

    experiments = [
        ExperimentResponse(
            name=record["name"],
            description=record["description"],
            models=record["models"],
            metrics=record["metrics"],
            risk_measures=record["risk_measures"],
            parameters=record["parameters"],
            metadata=record["metadata"],
            status=record["status"],
        )
        for record in records
    ]

    return ExperimentListResponse(
        experiments=experiments,
        count=len(experiments),
    )


@router.post(
    "/run",
    response_model=ExperimentRunResponse,
)
def run_experiment(
    request: ExperimentRunRequest,
) -> ExperimentRunResponse:
    """
    Execute one or more persistently registered experiments.

    Configuration parsing and execution remain delegated to the
    existing research-layer modules.
    """

    if not request.experiment_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one experiment name is required.",
        )

    run_results = []

    for experiment_name in request.experiment_names:
        record = storage.get_experiment(experiment_name)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Experiment '{experiment_name}' "
                    "was not found."
                ),
            )

        raw_config = {
            "experiments": [
                {
                    "name": record["name"],
                    "description": record["description"],
                    "models": record["models"],
                    "metrics": record["metrics"],
                    "risk_measures": record["risk_measures"],
                    "parameters": record["parameters"],
                    "metadata": record["metadata"],
                },
            ],
        }

        try:
            configuration = parse_experiment_config(
                raw_config,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid experiment "
                    f"'{experiment_name}': {exc}"
                ),
            ) from exc

        handlers = _build_handlers(configuration)

        try:
            result = run_experiments(
                configuration,
                handlers=handlers,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Invalid experiment execution "
                    f"configuration: {exc}"
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Experiment execution failed: {exc}"
                ),
            ) from exc

        run_results.append(result)

    response_results: list[
        ExperimentRunResultResponse
    ] = []

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
        experiment_count=len(response_results),
        successful_count=successful_count,
        failed_count=failed_count,
        all_successful=failed_count == 0,
        results=response_results,
    )


@router.get(
    "/{experiment_name}",
    response_model=ExperimentResponse,
)
def get_experiment(
    experiment_name: str,
) -> ExperimentResponse:
    """Return one persistently registered experiment."""

    record = storage.get_experiment(experiment_name)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Experiment '{experiment_name}' "
                "was not found."
            ),
        )

    return ExperimentResponse(
        name=record["name"],
        description=record["description"],
        models=record["models"],
        metrics=record["metrics"],
        risk_measures=record["risk_measures"],
        parameters=record["parameters"],
        metadata=record["metadata"],
        status=record["status"],
    )


def _build_handlers(
    configuration: Any,
) -> dict[str, Any]:
    """
    Build handlers for the experiments in the supplied
    validated configuration.

    The API layer does not implement model logic.
    """

    return {
        experiment.name: _completed_handler
        for experiment in configuration.experiments
    }


def _completed_handler(
    experiment: Any,
) -> dict[str, Any]:
    """
    Adapter for the currently implemented experiment runner.
    """

    return {
        "models": tuple(experiment.models),
        "status": "completed",
    }