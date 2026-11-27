from __future__ import annotations

import numpy as np
import pandas as pd

import backend.app.api.v1.forecasts as forecast_api
from backend.app.schemas.forecast import ForecastRequest
from backend.app.storage import DuckDBStorage


def _forecast_artifact(rows: int = 105) -> pd.DataFrame:
    origins = pd.date_range("2024-01-01", periods=rows, freq="B")
    targets = origins + pd.offsets.BDay(1)
    frames = []
    for asset_index, asset_id in enumerate(("AAA", "BBB")):
        rng = np.random.default_rng(900 + asset_index)
        actual_volatility = np.maximum(
            rng.normal(0.014 + 0.002 * asset_index, 0.002, size=rows),
            0.003,
        )
        actual_returns = rng.normal(
            0.0001,
            0.012 + 0.002 * asset_index,
            size=rows,
        )
        for model_index, model in enumerate(("A0", "A1")):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": origins,
                        "target_timestamp": targets,
                        "asset_id": asset_id,
                        "target_return": actual_returns,
                        "actual_volatility": actual_volatility,
                        "prediction": np.maximum(
                            actual_volatility
                            * (1.0 + (model_index - 0.25) * 0.08)
                            + rng.normal(0.0, 0.0005, size=rows),
                            0.002,
                        ),
                        "model": model,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _request(model: str | None = "A0") -> ForecastRequest:
    return ForecastRequest(
        experiment_name="forecast-integration",
        dataset_name="synthetic-dataset",
        model=model,
        assets=["AAA", "BBB"],
        start_timestamp=pd.Timestamp("2024-01-01"),
        end_timestamp=pd.Timestamp("2025-01-01"),
    )


def test_forecast_api_persists_evaluates_and_compares_common_samples(monkeypatch) -> None:
    artifact = _forecast_artifact()
    stored: dict[str, dict] = {}

    monkeypatch.setattr(
        forecast_api,
        "load_experiment_metadata",
        lambda experiment_name: {"dataset_name": "synthetic-dataset"},
    )
    monkeypatch.setattr(
        forecast_api,
        "load_experiment_results",
        lambda experiment_name: artifact.copy(),
    )
    monkeypatch.setattr(
        forecast_api.storage,
        "upsert_forecast",
        lambda record: stored.__setitem__(record["model"], record),
    )
    monkeypatch.setattr(
        forecast_api.storage,
        "get_forecast",
        lambda experiment_name, model: stored.get(model),
    )
    monkeypatch.setattr(
        forecast_api.storage,
        "list_forecasts",
        lambda experiment_name: list(stored.values()),
    )

    created_a0 = forecast_api.create_forecast(_request("A0"))
    created_a1 = forecast_api.create_forecast(_request("A1"))

    assert created_a0.status == "completed"
    assert created_a1.status == "completed"
    assert created_a0.observation_count == 210
    assert created_a0.asset_count == 2
    assert created_a0.forecasts[0].timestamp > created_a0.forecasts[0].origin_timestamp

    evaluation = forecast_api.evaluate_forecast(_request("A0"))
    assert evaluation.observation_count == 210
    assert evaluation.asset_count == 2
    assert len(evaluation.metrics) == 6
    assert {metric.metric for metric in evaluation.metrics} == {"MAE", "RMSE", "QLIKE"}
    assert {metric.asset_id for metric in evaluation.metrics} == {"AAA", "BBB"}

    comparison = forecast_api.compare_forecasts(_request(None))
    assert comparison.observation_count == 210
    assert comparison.asset_count == 2
    assert comparison.aggregation == "per_asset_on_common_asset_date_observations"
    assert len(comparison.models) == 4
    assert {(item.model, item.asset_id) for item in comparison.models} == {
        ("A0", "AAA"),
        ("A0", "BBB"),
        ("A1", "AAA"),
        ("A1", "BBB"),
    }


def test_forecast_storage_upsert_replaces_existing_model_result(tmp_path) -> None:
    repository = DuckDBStorage(tmp_path / "forecast-test.duckdb")
    original = {
        "experiment_name": "forecast-storage",
        "dataset_name": "fixture",
        "model": "A0",
        "forecasts": [],
        "observation_count": 0,
        "asset_count": 1,
        "status": "accepted",
    }
    repository.save_forecast(original)

    completed = {
        **original,
        "forecasts": [
            {
                "timestamp": "2024-01-02T00:00:00",
                "origin_timestamp": "2024-01-01T00:00:00",
                "asset_id": "AAA",
                "actual_return": -0.01,
                "actual_volatility": 0.015,
                "predicted_volatility": 0.014,
            }
        ],
        "observation_count": 1,
        "status": "completed",
    }
    repository.upsert_forecast(completed)

    stored = repository.get_forecast("forecast-storage", "A0")
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["observation_count"] == 1
    assert stored["forecasts"][0]["actual_return"] == -0.01
