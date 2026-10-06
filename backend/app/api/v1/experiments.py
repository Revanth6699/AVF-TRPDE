from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.app.experiments.artifacts import (
    save_experiment_artifacts,
)
from backend.app.experiments.configuration import (
    parse_experiment_config,
)
from backend.app.experiments.runner import run_experiments
from backend.app.research.integration.dataset_resolver import (
    DatasetResolutionError,
    resolve_research_dataset,
)
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.models.xgboost import XGBoostConfig
from backend.app.research.walk_forward.runner import (
    WalkForwardRunner,
)
from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)
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


SUPPORTED_RESEARCH_MODELS = (
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "Regime-XGBoost",
)

DEFAULT_FEATURE_COLUMNS = (
    "return",
    "realized_volatility",
    "open",
    "high",
    "low",
    "close",
    "volume",
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


def _registered_experiment_config(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Convert a stored experiment record into research configuration."""

    return {
        "experiments": [
            {
                "name": record["name"],
                "description": record["description"],
                "models": record["models"],
                "metrics": record["metrics"],
                "risk_measures": record["risk_measures"],
                "parameters": record["parameters"],
                "metadata": record["metadata"],
            }
        ]
    }


def _parameter_int(
    parameters: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = parameters.get(key, default)

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter '{key}' must be an integer."
        ) from exc

    if normalized < 1:
        raise ValueError(
            f"Parameter '{key}' must be greater than zero."
        )

    return normalized


def _parameter_float(
    parameters: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = parameters.get(key, default)

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter '{key}' must be numeric."
        ) from exc

    return normalized


def _resolve_registered_dataset(
    record: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Resolve the dataset associated with an experiment.

    Preferred configuration:
        parameters.dataset_name

    For the existing single-dataset API test, a unique registered
    dataset whose asset matches the experiment metadata is accepted.
    """

    requested_name = parameters.get("dataset_name")

    if requested_name:
        dataset = storage.get_dataset(
            str(requested_name)
        )

        if dataset is None:
            raise DatasetResolutionError(
                f"Dataset '{requested_name}' was not found."
            )

        return str(requested_name), dataset

    datasets = storage.list_datasets()

    if not datasets:
        raise DatasetResolutionError(
            "No registered datasets are available."
        )

    experiment_asset = str(
        record.get("metadata", {}).get("asset", "")
    ).strip().upper()

    matching = []

    for dataset in datasets:
        assets = tuple(
            str(asset).strip().upper()
            for asset in dataset.get("assets", [])
        )

        if experiment_asset and experiment_asset in assets:
            matching.append(dataset)

    if len(matching) == 1:
        dataset = matching[0]
        return str(dataset["name"]), dataset

    if len(datasets) == 1:
        dataset = datasets[0]
        return str(dataset["name"]), dataset

    raise DatasetResolutionError(
        "Experiment does not specify parameters.dataset_name "
        "and the registered datasets cannot be resolved "
        "unambiguously."
    )


def _build_research_runner(
    parameters: dict[str, Any],
) -> WalkForwardRunner:
    """Build the locked chronological walk-forward runner."""

    train_size = _parameter_int(
        parameters,
        "train_size",
        120,
    )

    test_size = _parameter_int(
        parameters,
        "test_size",
        1,
    )

    step_size = _parameter_int(
        parameters,
        "step_size",
        1,
    )

    expanding = bool(
        parameters.get(
            "expanding",
            False,
        )
    )

    splitter = WalkForwardSplitter(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        expanding=expanding,
    )

    return WalkForwardRunner(
        splitter
    )


def _build_model_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build explicit parameters for the locked models."""

    ewma_decay = _parameter_float(
        parameters,
        "ewma_decay",
        0.94,
    )

    hmm_components = _parameter_int(
        parameters,
        "hmm_n_components",
        3,
    )

    hmm_config = HMMConfig(
        n_components=hmm_components,
        covariance_type=str(
            parameters.get(
                "hmm_covariance_type",
                "full",
            )
        ),
        n_iter=_parameter_int(
            parameters,
            "hmm_n_iter",
            300,
        ),
        tol=_parameter_float(
            parameters,
            "hmm_tol",
            1e-4,
        ),
        random_state=_parameter_int(
            parameters,
            "hmm_random_state",
            42,
        ),
        min_covar=_parameter_float(
            parameters,
            "hmm_min_covar",
            1e-6,
        ),
    )

    xgboost_config = XGBoostConfig(
        n_estimators=_parameter_int(
            parameters,
            "xgb_n_estimators",
            100,
        ),
        max_depth=_parameter_int(
            parameters,
            "xgb_max_depth",
            6,
        ),
        learning_rate=_parameter_float(
            parameters,
            "xgb_learning_rate",
            0.3,
        ),
        subsample=_parameter_float(
            parameters,
            "xgb_subsample",
            1.0,
        ),
        colsample_bytree=_parameter_float(
            parameters,
            "xgb_colsample_bytree",
            1.0,
        ),
        min_child_weight=_parameter_float(
            parameters,
            "xgb_min_child_weight",
            1.0,
        ),
        reg_alpha=_parameter_float(
            parameters,
            "xgb_reg_alpha",
            0.0,
        ),
        reg_lambda=_parameter_float(
            parameters,
            "xgb_reg_lambda",
            1.0,
        ),
        random_state=_parameter_int(
            parameters,
            "xgb_random_state",
            42,
        ),
        objective=str(
            parameters.get(
                "xgb_objective",
                "reg:squarederror",
            )
        ),
    )

    return {
        "ewma_decay": ewma_decay,
        "hmm_config": hmm_config,
        "xgboost_config": xgboost_config,
    }


def _feature_columns(
    parameters: dict[str, Any],
) -> tuple[str, ...]:
    configured = parameters.get(
        "feature_columns"
    )

    if configured is None:
        return DEFAULT_FEATURE_COLUMNS

    if not isinstance(configured, (list, tuple)):
        raise ValueError(
            "Parameter 'feature_columns' must be a list or tuple."
        )

    columns = tuple(
        str(column).strip()
        for column in configured
        if str(column).strip()
    )

    if not columns:
        raise ValueError(
            "Parameter 'feature_columns' must not be empty."
        )

    return columns


def _execute_research(
    experiment: Any,
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the real AVF-TRPDE walk-forward research pipeline.

    The execution boundary is:

        registered experiment
            -> registered dataset
            -> research-ready dataframe
            -> walk-forward runner
            -> model predictions
            -> experiment artifacts
    """

    parameters = dict(
        experiment.parameters or {}
    )

    metadata = dict(
        experiment.metadata or {}
    )

    dataset_name, dataset = (
        _resolve_registered_dataset(
            record,
            parameters,
        )
    )

    research_dataframe = (
        resolve_research_dataset(
            dataset,
            parameters=parameters,
        )
    )

    runner = _build_research_runner(
        parameters
    )

    model_parameters = _build_model_parameters(
        parameters
    )

    feature_columns = _feature_columns(
        parameters
    )

    requested_models = tuple(
        experiment.models
    )

    predictions: list[Any] = []
    model_results: dict[str, Any] = {}

    for model in requested_models:
        if model == "HMM":
            continue

        if model not in SUPPORTED_RESEARCH_MODELS:
            raise ValueError(
                f"Unsupported research model: {model}"
            )

        result = runner.run_model(
            research_dataframe,
            model=model,
            target_column="target",
            ewma_decay=model_parameters[
                "ewma_decay"
            ],
            hmm_config=model_parameters[
                "hmm_config"
            ],
            xgboost_config=model_parameters[
                "xgboost_config"
            ],
            feature_columns=feature_columns,
        )

        model_predictions = (
            result.predictions.copy()
        )

        model_predictions["model"] = model

        predictions.append(
            model_predictions
        )

        model_results[model] = {
            "fold_count": result.fold_count,
            "prediction_count": (
                result.prediction_count
            ),
        }

    if not predictions:
        raise ValueError(
            "No executable forecasting models were "
            "requested by the experiment."
        )

    import pandas as pd

    combined_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    combined_predictions = (
        combined_predictions
        .sort_values(
            [
                "timestamp",
                "asset_id",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    artifact_metadata = {
        "experiment_name": experiment.name,
        "dataset_name": dataset_name,
        "models_requested": list(
            requested_models
        ),
        "models_executed": list(
            model_results.keys()
        ),
        "research_rows": len(
            research_dataframe
        ),
        "prediction_rows": len(
            combined_predictions
        ),
        "feature_columns": list(
            feature_columns
        ),
        "parameters": parameters,
        "metadata": metadata,
        "model_results": model_results,
        "status": "completed",
    }

    artifacts = save_experiment_artifacts(
        experiment.name,
        artifact_metadata,
        combined_predictions,
    )

    return {
        "status": "completed",
        "dataset_name": dataset_name,
        "research_rows": len(
            research_dataframe
        ),
        "prediction_rows": len(
            combined_predictions
        ),
        "models_requested": list(
            requested_models
        ),
        "models_executed": list(
            model_results.keys()
        ),
        "model_results": model_results,
        "artifacts": {
            name: {
                "path": str(
                    artifact.path
                ),
                "artifact_type": (
                    artifact.artifact_type
                ),
                "size_bytes": (
                    artifact.size_bytes
                ),
            }
            for name, artifact in artifacts.items()
        },
    }


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
    """

    raw_config = {
        "experiments": [
            _model_dump(request),
        ],
    }

    try:
        parse_experiment_config(
            raw_config
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    response = _build_experiment_response(
        request
    )

    try:
        storage.save_experiment(
            _model_dump(response)
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
            risk_measures=record[
                "risk_measures"
            ],
            parameters=record[
                "parameters"
            ],
            metadata=record[
                "metadata"
            ],
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
    Execute one or more persistently registered experiments
    through the actual AVF-TRPDE research pipeline.
    """

    if not request.experiment_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "At least one experiment name "
                "is required."
            ),
        )

    run_results = []

    for experiment_name in (
        request.experiment_names
    ):
        record = storage.get_experiment(
            experiment_name
        )

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Experiment '{experiment_name}' "
                    "was not found."
                ),
            )

        raw_config = (
            _registered_experiment_config(
                record
            )
        )

        try:
            configuration = (
                parse_experiment_config(
                    raw_config
                )
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=(
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ),
                detail=(
                    f"Invalid experiment "
                    f"'{experiment_name}': {exc}"
                ),
            ) from exc

        handlers = {
            experiment.name: (
                lambda experiment,
                current_record=record:
                _execute_research(
                    experiment,
                    current_record,
                )
            )
            for experiment in (
                configuration.experiments
            )
        }

        try:
            result = run_experiments(
                configuration,
                handlers=handlers,
            )
        except (
            DatasetResolutionError,
            ValueError,
            TypeError,
        ) as exc:
            raise HTTPException(
                status_code=(
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ),
                detail=(
                    "Experiment research "
                    f"configuration failed: {exc}"
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Experiment research "
                    f"execution failed: {exc}"
                ),
            ) from exc

        run_results.append(result)

    response_results: list[
        ExperimentRunResultResponse
    ] = []

    successful_count = 0
    failed_count = 0

    for result in run_results:
        successful_count += (
            result.successful_count
        )
        failed_count += (
            result.failed_count
        )

        for item in result.results:
            response_results.append(
                ExperimentRunResultResponse(
                    name=item.name,
                    status=item.status,
                    result=(
                        item.result
                        if isinstance(item.result, dict)
                        else {}
                    ),                    
                                      
                    error=item.error,
                )
            )

    return ExperimentRunResponse(
        experiment_count=len(
            response_results
        ),
        successful_count=successful_count,
        failed_count=failed_count,
        all_successful=(
            failed_count == 0
        ),
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

    record = storage.get_experiment(
        experiment_name
    )

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
        description=record[
            "description"
        ],
        models=record["models"],
        metrics=record["metrics"],
        risk_measures=record[
            "risk_measures"
        ],
        parameters=record[
            "parameters"
        ],
        metadata=record[
            "metadata"
        ],
        status=record["status"],
    )