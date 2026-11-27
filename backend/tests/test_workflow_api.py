from __future__ import annotations

import pandas as pd

from backend.app.api.v1 import portfolios, stress
from backend.app.schemas.dataset import DatasetRequest
from backend.app.schemas.experiment import ExperimentRequest
from backend.app.schemas.portfolio import PortfolioRequest
from backend.app.schemas.stress import StressRunRequest
from backend.app.storage import DuckDBStorage


def _seed_store(store: DuckDBStorage) -> None:
    store.save_dataset(
        {
            "name": "dataset-1",
            "source": "parquet",
            "frequency": "daily",
            "assets": ["AAA", "BBB"],
            "start_timestamp": "2024-01-01",
            "end_timestamp": "2024-03-01",
            "row_count": 60,
            "fingerprint": "test-fingerprint",
            "status": "registered",
        }
    )
    store.save_experiment(
        {
            "name": "exp-1",
            "description": "",
            "models": ["EWMA"],
            "metrics": ["MAE", "RMSE", "QLIKE"],
            "risk_measures": [],
            "parameters": {"dataset_name": "dataset-1"},
            "metadata": {},
            "status": "configured",
        }
    )
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for index, timestamp in enumerate(dates):
        for asset, offset in (("AAA", 0.0), ("BBB", 0.4)):
            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "origin_timestamp": (timestamp - pd.offsets.BDay(1)).isoformat(),
                    "asset_id": asset,
                    "actual_return": float(0.01 * __import__("math").sin(index + offset)),
                    "actual_volatility": 0.01,
                    "predicted_volatility": 0.012,
                }
            )
    store.upsert_forecast(
        {
            "experiment_name": "exp-1",
            "dataset_name": "dataset-1",
            "model": "EWMA",
            "forecasts": records,
            "observation_count": len(records),
            "asset_count": 2,
            "status": "completed",
        }
    )


def test_portfolio_and_stress_api_workflow(tmp_path, monkeypatch) -> None:
    store = DuckDBStorage(tmp_path / "api-workflow.duckdb")
    _seed_store(store)
    monkeypatch.setattr(portfolios, "storage", store)
    monkeypatch.setattr(stress, "storage", store)

    request = PortfolioRequest(
        experiment_name="exp-1",
        dataset_name="dataset-1",
        strategy="static",
        model="EWMA",
        assets=["AAA", "BBB"],
        start_timestamp="2024-01-01T00:00:00",
        end_timestamp="2024-03-01T00:00:00",
        volatility_target=0.01,
        max_position=0.8,
        max_turnover=2.0,
        transaction_cost=0.0004,
    )
    result = portfolios.create_portfolio_analysis(request)
    assert result.status == "completed"
    assert result.model == "EWMA"
    assert result.observation_count == 30
    assert result.performance["net_pnl"] is not None
    assert len(store.list_portfolios("exp-1")) == 1

    buy_and_hold_request = request.model_copy(update={"strategy": "buy_and_hold"})
    buy_and_hold_result = portfolios.create_portfolio_analysis(buy_and_hold_request)
    assert buy_and_hold_result.status == "completed"
    comparison = portfolios.compare_portfolios(request)
    assert len(comparison.strategies) == 2
    assert {item.model for item in comparison.strategies} == {"EWMA"}

    stress_result = stress.run_stress(
        StressRunRequest(experiment_name="exp-1", strategy="static")
    )
    assert stress_result.status == "completed"
    assert stress_result.scenario_count == 5
    assert store.get_stress_result("exp-1", "static") is not None
