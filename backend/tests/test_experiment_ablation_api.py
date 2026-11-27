from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import backend.app.api.v1.experiments as experiment_api
from backend.app.schemas.experiment import ExperimentRequest, ExperimentRunRequest


def _research_fixture() -> pd.DataFrame:
    # Reuse the repository's established controlled market fixture rather
    # than independent noise that can create pathological HMM covariance.
    from test_walk_forward_controlled import make_controlled_data

    return make_controlled_data().iloc[:125].copy().reset_index(drop=True)


def test_experiment_executor_runs_locked_a0_to_a3_ablations(monkeypatch) -> None:
    frame = _research_fixture()
    monkeypatch.setattr(
        experiment_api,
        "_resolve_registered_dataset",
        lambda record, parameters: ("synthetic-fixture", {"name": "synthetic-fixture"}),
    )
    monkeypatch.setattr(
        experiment_api,
        "resolve_research_dataset",
        lambda dataset, parameters: frame.copy(),
    )

    def fake_save_artifacts(name, metadata, predictions):
        return {
            "metadata": SimpleNamespace(
                path=Path("metadata.json"),
                artifact_type="json",
                size_bytes=10,
            ),
            "predictions": SimpleNamespace(
                path=Path("predictions.parquet"),
                artifact_type="parquet",
                size_bytes=20,
            ),
        }

    monkeypatch.setattr(
        experiment_api,
        "save_experiment_artifacts",
        fake_save_artifacts,
    )

    records = {}

    def fake_save_experiment(record):
        records[record["name"]] = record

    monkeypatch.setattr(
        experiment_api.storage,
        "save_experiment",
        fake_save_experiment,
    )
    monkeypatch.setattr(
        experiment_api.storage,
        "get_experiment",
        lambda name: records.get(name),
    )

    created = experiment_api.create_experiment(
        ExperimentRequest(
            name="ablation-smoke",
            description="Synthetic A0-A3 API regression",
            models=["XGBoost"],
            metrics=["MAE", "RMSE", "QLIKE"],
            risk_measures=[],
            parameters={
                "train_size": 120,
                "test_size": 1,
                "step_size": 1,
                "ablations": ["A0", "A1", "A2", "A3"],
                "xgb_n_estimators": 4,
                "xgb_max_depth": 2,
                "hmm_n_iter": 20,
                "bootstrap_samples": 100,
            },
            metadata={"fixture": True},
        )
    )
    assert created.status == "configured"

    run_response = experiment_api.run_experiment(
        ExperimentRunRequest(experiment_names=["ablation-smoke"])
    )

    assert run_response.all_successful is True
    assert run_response.successful_count == 1
    assert run_response.failed_count == 0
    assert len(run_response.results) == 1

    result = run_response.results[0].result
    assert result["status"] == "completed"
    assert result["ablations_executed"] == ["A0", "A1", "A2", "A3"]
    assert result["models_executed"] == ["A0", "A1", "A2", "A3"]
    assert result["prediction_rows"] == 20
    assert set(result["model_results"]) == {"A0", "A1", "A2", "A3"}
    assert set(result["evaluation"]["assets"]) == {"SYNTH"}
    assert result["artifacts"]["metadata"]["size_bytes"] == 10
