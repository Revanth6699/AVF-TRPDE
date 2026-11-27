from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import backend.app.api.v1.risk as risk_api
from backend.app.schemas.risk import RiskRequest
from backend.app.storage import DuckDBStorage


def _forecast_artifact(rows: int = 105) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02", periods=rows, freq="B")
    frames = []
    for asset_index, asset_id in enumerate(("AAA", "BBB")):
        rng = np.random.default_rng(700 + asset_index)
        actual_returns = rng.normal(
            loc=0.0001,
            scale=0.012 + asset_index * 0.002,
            size=rows,
        )
        for model_index, model in enumerate(("A0", "A1")):
            frames.append(
                pd.DataFrame(
                    {
                        "target_timestamp": timestamps,
                        "asset_id": asset_id,
                        "target_return": actual_returns,
                        "prediction": rng.uniform(
                            0.008 + model_index * 0.002,
                            0.015 + model_index * 0.002,
                            size=rows,
                        ),
                        "model": model,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _request(model: str | None = "A0") -> RiskRequest:
    return RiskRequest(
        experiment_name="risk-integration",
        dataset_name="synthetic-dataset",
        model=model,
        assets=["AAA", "BBB"],
        confidence_levels=[0.95, 0.99],
        start_timestamp=pd.Timestamp("2024-01-01"),
        end_timestamp=pd.Timestamp("2025-01-01"),
    )


def test_risk_api_generates_backtests_and_common_sample_comparison(monkeypatch) -> None:
    artifact = _forecast_artifact()
    stored: dict[str, dict] = {}

    monkeypatch.setattr(
        risk_api,
        "load_experiment_metadata",
        lambda experiment_name: {"dataset_name": "synthetic-dataset"},
    )
    monkeypatch.setattr(
        risk_api,
        "load_experiment_results",
        lambda experiment_name: artifact.copy(),
    )
    monkeypatch.setattr(risk_api, "DEFAULT_FHS_MIN_OBSERVATIONS", 100)
    monkeypatch.setattr(risk_api, "DEFAULT_FHS_SIMULATIONS", 500)
    monkeypatch.setattr(
        risk_api.storage,
        "upsert_risk",
        lambda record: stored.__setitem__(record["model"], record),
    )
    monkeypatch.setattr(
        risk_api.storage,
        "get_risk",
        lambda experiment_name, model: stored.get(model),
    )
    monkeypatch.setattr(
        risk_api.storage,
        "list_risks",
        lambda experiment_name: list(stored.values()),
    )

    result_a0 = risk_api.create_risk_analysis(_request("A0"))
    result_a1 = risk_api.create_risk_analysis(_request("A1"))

    assert result_a0.status == "completed"
    assert result_a1.status == "completed"
    assert result_a0.observation_count == 10
    assert result_a0.asset_count == 2
    assert all(point.history_count >= 100 for point in result_a0.risk_points)

    backtest = risk_api.backtest_risk(_request("A0"))
    assert backtest.observation_count == 10
    assert backtest.asset_count == 2
    assert len(backtest.results) == 4
    assert backtest.warnings
    assert any("low statistical power" in warning for warning in backtest.warnings)
    assert {result.measure for result in backtest.results} == {"VaR95", "VaR99"}
    assert all(0.0 <= result.p_value <= 1.0 for result in backtest.results)
    assert all(
        0.0 <= result.independence_p_value <= 1.0
        for result in backtest.results
    )
    assert all(
        0.0 <= result.conditional_coverage_p_value <= 1.0
        for result in backtest.results
    )

    comparison = risk_api.compare_risk(_request(None))
    assert comparison.observation_count == 10
    assert comparison.asset_count == 2
    assert {item.model for item in comparison.models} == {"A0", "A1"}
    assert all(np.isfinite(item.var_95) for item in comparison.models)
    assert all(np.isfinite(item.var_99) for item in comparison.models)
    assert all(np.isfinite(item.es_95) for item in comparison.models)


def test_risk_api_rejects_legacy_forecast_artifacts_without_next_day_returns(monkeypatch) -> None:
    artifact = _forecast_artifact().drop(
        columns=["target_timestamp", "target_return"]
    )
    monkeypatch.setattr(
        risk_api,
        "load_experiment_metadata",
        lambda experiment_name: {"dataset_name": "synthetic-dataset"},
    )
    monkeypatch.setattr(
        risk_api,
        "load_experiment_results",
        lambda experiment_name: artifact.copy(),
    )

    with pytest.raises(HTTPException) as error:
        risk_api.create_risk_analysis(_request("A0"))
    assert error.value.status_code == 422
    assert "risk-alignment columns" in error.value.detail



def test_risk_storage_upsert_replaces_existing_model_result(tmp_path) -> None:
    repository = DuckDBStorage(tmp_path / "risk-test.duckdb")
    original = {
        "experiment_name": "risk-storage",
        "dataset_name": "fixture",
        "model": "A0",
        "risk_points": [],
        "observation_count": 0,
        "asset_count": 1,
        "status": "accepted",
    }
    repository.save_risk(original)

    completed = {
        **original,
        "risk_points": [
            {
                "timestamp": "2024-01-02T00:00:00",
                "asset_id": "AAA",
                "var_95": 0.02,
                "var_99": 0.03,
                "es_95": 0.025,
                "actual_return": -0.01,
                "forecast_volatility": 0.01,
                "history_count": 100,
            }
        ],
        "observation_count": 1,
        "status": "completed",
    }
    repository.upsert_risk(completed)

    stored = repository.get_risk("risk-storage", "A0")
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["observation_count"] == 1
    assert stored["risk_points"][0]["actual_return"] == -0.01
