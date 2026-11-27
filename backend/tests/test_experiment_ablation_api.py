from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import backend.app.api.v1.experiments as experiment_api


def _research_fixture(rows: int = 130) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    returns = rng.normal(0.0002, 0.012, size=rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    previous_close = np.concatenate(([close[0]], close[:-1]))
    realized_volatility = np.maximum(
        rng.normal(0.014, 0.002, size=rows),
        0.003,
    )
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=rows, freq="B"),
            "asset_id": "TEST_ASSET",
            "return": returns,
            "realized_volatility": realized_volatility,
            "target": np.roll(realized_volatility, -1),
            "open": previous_close,
            "high": np.maximum(previous_close, close) * 1.003,
            "low": np.minimum(previous_close, close) * 0.997,
            "close": close,
            "volume": rng.integers(100_000, 500_000, size=rows),
        }
    )


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

    experiment = SimpleNamespace(
        name="ablation-smoke",
        models=["XGBoost"],
        metrics=["MAE", "RMSE", "QLIKE"],
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

    result = experiment_api._execute_research(
        experiment,
        {"metadata": {"asset": "TEST_ASSET"}},
    )

    assert result["status"] == "completed"
    assert result["ablations_executed"] == ["A0", "A1", "A2", "A3"]
    assert result["models_executed"] == ["A0", "A1", "A2", "A3"]
    assert result["prediction_rows"] == 40
    assert set(result["model_results"]) == {"A0", "A1", "A2", "A3"}
    assert set(result["evaluation"]["assets"]) == {"TEST_ASSET"}
    assert result["artifacts"]["metadata"]["size_bytes"] == 10
