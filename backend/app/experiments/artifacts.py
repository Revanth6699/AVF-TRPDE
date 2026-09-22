from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentArtifactError(RuntimeError):
    """Raised when an experiment artifact cannot be created or loaded."""


@dataclass(frozen=True)
class ArtifactResult:
    """Metadata describing a saved experiment artifact."""

    path: Path
    artifact_type: str
    size_bytes: int


def save_json_artifact(
    data: dict[str, Any],
    path: str | Path,
) -> ArtifactResult:
    """Save a JSON-compatible experiment artifact."""

    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    artifact_path = _prepare_path(path, ".json")

    try:
        with artifact_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
    except (OSError, TypeError, ValueError) as exc:
        raise ExperimentArtifactError(
            f"Failed to save JSON artifact: {artifact_path}"
        ) from exc

    return _build_result(
        artifact_path,
        "json",
    )


def load_json_artifact(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON experiment artifact."""

    artifact_path = _validate_existing_file(
        path,
        ".json",
    )

    try:
        with artifact_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentArtifactError(
            f"Failed to load JSON artifact: {artifact_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ExperimentArtifactError(
            "JSON experiment artifact must contain an object."
        )

    return data


def save_dataframe_artifact(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> ArtifactResult:
    """Save a tabular experiment artifact as Parquet."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise ExperimentArtifactError(
            "Cannot save an empty dataframe as an experiment artifact."
        )

    artifact_path = _prepare_path(
        path,
        ".parquet",
    )

    try:
        dataframe.to_parquet(
            artifact_path,
            index=False,
        )
    except (OSError, ValueError, ImportError) as exc:
        raise ExperimentArtifactError(
            f"Failed to save dataframe artifact: {artifact_path}"
        ) from exc

    return _build_result(
        artifact_path,
        "parquet",
    )


def load_dataframe_artifact(
    path: str | Path,
) -> pd.DataFrame:
    """Load a tabular experiment artifact from Parquet."""

    artifact_path = _validate_existing_file(
        path,
        ".parquet",
    )

    try:
        dataframe = pd.read_parquet(
            artifact_path,
        )
    except (OSError, ValueError, ImportError) as exc:
        raise ExperimentArtifactError(
            f"Failed to load dataframe artifact: {artifact_path}"
        ) from exc

    if not isinstance(dataframe, pd.DataFrame):
        raise ExperimentArtifactError(
            "Loaded artifact is not a pandas.DataFrame."
        )

    return dataframe


def save_experiment_artifacts(
    experiment_name: str,
    metadata: dict[str, Any],
    results: pd.DataFrame | None = None,
    *,
    output_directory: str | Path = "experiments",
) -> dict[str, ArtifactResult]:
    """
    Save experiment metadata and optional tabular results.

    Artifacts are stored under:

        <output_directory>/<experiment_name>/
            metadata.json
            results.parquet
    """

    normalized_name = _normalize_experiment_name(
        experiment_name,
    )

    experiment_directory = (
        Path(output_directory)
        / normalized_name
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts: dict[str, ArtifactResult] = {}

    artifacts["metadata"] = save_json_artifact(
        metadata,
        experiment_directory / "metadata.json",
    )

    if results is not None:
        artifacts["results"] = save_dataframe_artifact(
            results,
            experiment_directory / "results.parquet",
        )

    return artifacts


def load_experiment_metadata(
    experiment_name: str,
    *,
    output_directory: str | Path = "experiments",
) -> dict[str, Any]:
    """Load metadata for a named experiment."""

    normalized_name = _normalize_experiment_name(
        experiment_name,
    )

    path = (
        Path(output_directory)
        / normalized_name
        / "metadata.json"
    )

    return load_json_artifact(path)


def load_experiment_results(
    experiment_name: str,
    *,
    output_directory: str | Path = "experiments",
) -> pd.DataFrame:
    """Load tabular results for a named experiment."""

    normalized_name = _normalize_experiment_name(
        experiment_name,
    )

    path = (
        Path(output_directory)
        / normalized_name
        / "results.parquet"
    )

    return load_dataframe_artifact(path)


def _prepare_path(
    path: str | Path,
    expected_suffix: str,
) -> Path:
    artifact_path = Path(path)

    if artifact_path.suffix.lower() != expected_suffix:
        raise ExperimentArtifactError(
            f"Artifact path must use '{expected_suffix}' suffix."
        )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return artifact_path


def _validate_existing_file(
    path: str | Path,
    expected_suffix: str,
) -> Path:
    artifact_path = Path(path)

    if artifact_path.suffix.lower() != expected_suffix:
        raise ExperimentArtifactError(
            f"Artifact path must use '{expected_suffix}' suffix."
        )

    if not artifact_path.exists():
        raise ExperimentArtifactError(
            f"Artifact does not exist: {artifact_path}"
        )

    if not artifact_path.is_file():
        raise ExperimentArtifactError(
            f"Artifact path is not a file: {artifact_path}"
        )

    return artifact_path


def _normalize_experiment_name(
    experiment_name: str,
) -> str:
    if not isinstance(experiment_name, str):
        raise TypeError(
            "experiment_name must be a string."
        )

    normalized = experiment_name.strip()

    if not normalized:
        raise ExperimentArtifactError(
            "experiment_name must not be empty."
        )

    if normalized in {".", ".."}:
        raise ExperimentArtifactError(
            "Invalid experiment name."
        )

    if any(
        separator in normalized
        for separator in (
            "/",
            "\\",
        )
    ):
        raise ExperimentArtifactError(
            "experiment_name must not contain path separators."
        )

    return normalized


def _build_result(
    path: Path,
    artifact_type: str,
) -> ArtifactResult:
    return ArtifactResult(
        path=path,
        artifact_type=artifact_type,
        size_bytes=path.stat().st_size,
    )


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values for JSON serialization."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )