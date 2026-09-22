from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


LOCKED_MODELS = (
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "HMM",
    "Regime-XGBoost",
)

LOCKED_METRICS = (
    "MAE",
    "RMSE",
    "QLIKE",
)

LOCKED_RISK_MEASURES = (
    "VaR95",
    "VaR99",
    "ES95",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration for one AVF-TRPDE experiment."""

    name: str
    description: str
    models: tuple[str, ...]
    metrics: tuple[str, ...]
    risk_measures: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Collection of validated AVF-TRPDE experiment definitions."""

    experiments: tuple[ExperimentConfig, ...]
    source_path: Path | None = None

    @property
    def experiment_count(self) -> int:
        """Return the number of configured experiments."""

        return len(self.experiments)

    def get(self, name: str) -> ExperimentConfig:
        """Return an experiment by name."""

        normalized_name = name.strip()

        if not normalized_name:
            raise ExperimentConfigurationError(
                "Experiment name must not be empty."
            )

        for experiment in self.experiments:
            if experiment.name == normalized_name:
                return experiment

        raise ExperimentConfigurationError(
            f"Experiment '{normalized_name}' was not found."
        )


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfiguration:
    """
    Load and validate an experiment YAML configuration.

    Expected top-level structure:

        experiments:
          - name: baseline
            description: ...
            models:
              - EWMA
              - GJR-GARCH
            metrics:
              - MAE
              - RMSE
              - QLIKE
            risk_measures:
              - VaR95
              - VaR99
              - ES95
            parameters: {}
            metadata: {}
    """

    config_path = Path(path)

    if not config_path.exists():
        raise ExperimentConfigurationError(
            f"Experiment configuration does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ExperimentConfigurationError(
            f"Experiment configuration is not a file: {config_path}"
        )

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ExperimentConfigurationError(
            "Experiment configuration must use .yaml or .yml."
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ExperimentConfigurationError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc
    except OSError as exc:
        raise ExperimentConfigurationError(
            f"Unable to read configuration: {config_path}"
        ) from exc

    configuration = parse_experiment_config(
        raw_config,
        source_path=config_path,
    )

    return configuration


def parse_experiment_config(
    raw_config: Any,
    *,
    source_path: str | Path | None = None,
) -> ExperimentConfiguration:
    """
    Parse and validate an in-memory experiment configuration.
    """

    if not isinstance(raw_config, dict):
        raise ExperimentConfigurationError(
            "Experiment configuration must be a mapping."
        )

    raw_experiments = raw_config.get("experiments")

    if not isinstance(raw_experiments, list):
        raise ExperimentConfigurationError(
            "'experiments' must be a list."
        )

    if not raw_experiments:
        raise ExperimentConfigurationError(
            "'experiments' must contain at least one experiment."
        )

    experiments: list[ExperimentConfig] = []

    for index, raw_experiment in enumerate(raw_experiments):
        experiments.append(
            _parse_experiment(
                raw_experiment,
                index=index,
            )
        )

    _validate_unique_experiment_names(experiments)

    resolved_path = (
        Path(source_path)
        if source_path is not None
        else None
    )

    return ExperimentConfiguration(
        experiments=tuple(experiments),
        source_path=resolved_path,
    )


def save_experiment_config(
    configuration: ExperimentConfiguration,
    path: str | Path,
) -> Path:
    """
    Save a validated experiment configuration as YAML.
    """

    if not isinstance(
        configuration,
        ExperimentConfiguration,
    ):
        raise TypeError(
            "configuration must be an ExperimentConfiguration."
        )

    output_path = Path(path)

    if output_path.suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        raise ExperimentConfigurationError(
            "Experiment configuration must use .yaml or .yml."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiments": [
            _experiment_to_dict(experiment)
            for experiment in configuration.experiments
        ]
    }

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                payload,
                file,
                sort_keys=False,
                default_flow_style=False,
            )
    except OSError as exc:
        raise ExperimentConfigurationError(
            f"Unable to write configuration: {output_path}"
        ) from exc

    return output_path


def validate_experiment_config(
    configuration: ExperimentConfiguration,
) -> None:
    """
    Validate an already parsed experiment configuration.
    """

    if not isinstance(
        configuration,
        ExperimentConfiguration,
    ):
        raise TypeError(
            "configuration must be an ExperimentConfiguration."
        )

    if not configuration.experiments:
        raise ExperimentConfigurationError(
            "At least one experiment is required."
        )

    _validate_unique_experiment_names(
        list(configuration.experiments)
    )

    for experiment in configuration.experiments:
        _validate_experiment(experiment)


def list_experiment_names(
    configuration: ExperimentConfiguration,
) -> tuple[str, ...]:
    """Return experiment names in configuration order."""

    if not isinstance(
        configuration,
        ExperimentConfiguration,
    ):
        raise TypeError(
            "configuration must be an ExperimentConfiguration."
        )

    return tuple(
        experiment.name
        for experiment in configuration.experiments
    )


def _parse_experiment(
    raw_experiment: Any,
    *,
    index: int,
) -> ExperimentConfig:
    if not isinstance(raw_experiment, dict):
        raise ExperimentConfigurationError(
            f"Experiment at index {index} must be a mapping."
        )

    name = _require_string(
        raw_experiment,
        "name",
        context=f"experiment[{index}]",
    )

    description = _require_string(
        raw_experiment,
        "description",
        context=f"experiment[{index}]",
    )

    models = _parse_string_tuple(
        raw_experiment.get("models"),
        field_name="models",
        context=f"experiment '{name}'",
    )

    metrics = _parse_string_tuple(
        raw_experiment.get("metrics"),
        field_name="metrics",
        context=f"experiment '{name}'",
    )

    risk_measures = _parse_string_tuple(
        raw_experiment.get("risk_measures"),
        field_name="risk_measures",
        context=f"experiment '{name}'",
    )

    parameters = raw_experiment.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):
        raise ExperimentConfigurationError(
            f"'parameters' for experiment '{name}' "
            "must be a mapping."
        )

    metadata = raw_experiment.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        raise ExperimentConfigurationError(
            f"'metadata' for experiment '{name}' "
            "must be a mapping."
        )

    experiment = ExperimentConfig(
        name=name,
        description=description,
        models=models,
        metrics=metrics,
        risk_measures=risk_measures,
        parameters=dict(parameters),
        metadata=dict(metadata),
    )

    _validate_experiment(experiment)

    return experiment


def _validate_experiment(
    experiment: ExperimentConfig,
) -> None:
    if not experiment.name.strip():
        raise ExperimentConfigurationError(
            "Experiment name must not be empty."
        )

    if not experiment.description.strip():
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "description must not be empty."
        )

    if not experiment.models:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "must define at least one model."
        )

    invalid_models = [
        model
        for model in experiment.models
        if model not in LOCKED_MODELS
    ]

    if invalid_models:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' contains "
            f"unsupported models: {invalid_models}. "
            f"Allowed models: {LOCKED_MODELS}."
        )

    if len(set(experiment.models)) != len(
        experiment.models
    ):
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "contains duplicate models."
        )

    if not experiment.metrics:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "must define at least one forecast metric."
        )

    invalid_metrics = [
        metric
        for metric in experiment.metrics
        if metric not in LOCKED_METRICS
    ]

    if invalid_metrics:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' contains "
            f"unsupported metrics: {invalid_metrics}. "
            f"Allowed metrics: {LOCKED_METRICS}."
        )

    if len(set(experiment.metrics)) != len(
        experiment.metrics
    ):
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "contains duplicate metrics."
        )

    if not experiment.risk_measures:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "must define at least one risk measure."
        )

    invalid_risk_measures = [
        measure
        for measure in experiment.risk_measures
        if measure not in LOCKED_RISK_MEASURES
    ]

    if invalid_risk_measures:
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' contains "
            f"unsupported risk measures: "
            f"{invalid_risk_measures}. "
            f"Allowed risk measures: "
            f"{LOCKED_RISK_MEASURES}."
        )

    if len(set(experiment.risk_measures)) != len(
        experiment.risk_measures
    ):
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "contains duplicate risk measures."
        )

    if not isinstance(experiment.parameters, dict):
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "parameters must be a mapping."
        )

    if not isinstance(experiment.metadata, dict):
        raise ExperimentConfigurationError(
            f"Experiment '{experiment.name}' "
            "metadata must be a mapping."
        )


def _validate_unique_experiment_names(
    experiments: list[ExperimentConfig],
) -> None:
    names = [
        experiment.name
        for experiment in experiments
    ]

    if len(names) != len(set(names)):
        duplicates = sorted(
            {
                name
                for name in names
                if names.count(name) > 1
            }
        )

        raise ExperimentConfigurationError(
            "Experiment names must be unique. "
            f"Duplicates: {duplicates}"
        )


def _require_string(
    mapping: dict[str, Any],
    field_name: str,
    *,
    context: str,
) -> str:
    value = mapping.get(field_name)

    if not isinstance(value, str):
        raise ExperimentConfigurationError(
            f"'{field_name}' in {context} "
            "must be a string."
        )

    value = value.strip()

    if not value:
        raise ExperimentConfigurationError(
            f"'{field_name}' in {context} "
            "must not be empty."
        )

    return value


def _parse_string_tuple(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ExperimentConfigurationError(
            f"'{field_name}' in {context} "
            "must be a list."
        )

    if not value:
        raise ExperimentConfigurationError(
            f"'{field_name}' in {context} "
            "must contain at least one value."
        )

    normalized: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise ExperimentConfigurationError(
                f"All values in '{field_name}' "
                f"for {context} must be strings."
            )

        item = item.strip()

        if not item:
            raise ExperimentConfigurationError(
                f"'{field_name}' in {context} "
                "contains an empty value."
            )

        normalized.append(item)

    return tuple(normalized)


def _experiment_to_dict(
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "name": experiment.name,
        "description": experiment.description,
        "models": list(experiment.models),
        "metrics": list(experiment.metrics),
        "risk_measures": list(
            experiment.risk_measures
        ),
        "parameters": dict(experiment.parameters),
        "metadata": dict(experiment.metadata),
    }