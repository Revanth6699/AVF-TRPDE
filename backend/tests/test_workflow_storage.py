from __future__ import annotations

from backend.app.storage import DuckDBStorage


def test_portfolio_and_stress_records_persist_in_duckdb(tmp_path) -> None:
    store = DuckDBStorage(tmp_path / "workflow.duckdb")
    portfolio = {
        "experiment_name": "exp-1",
        "dataset_name": "dataset-1",
        "strategy": "static",
        "positions": [{"timestamp": "2024-01-01", "asset_id": "AAA", "weight": 1.0}],
        "return_series": [{"timestamp": "2024-01-01", "gross_return": 0.01, "net_return": 0.009, "turnover": 1.0, "transaction_cost": 0.001}],
        "performance": {"net_pnl": 0.009, "sharpe_ratio": None},
        "observation_count": 1,
        "asset_count": 1,
        "status": "completed",
    }
    store.save_portfolio(portfolio)
    stored_portfolio = store.get_portfolio("exp-1", "static")
    assert stored_portfolio is not None
    assert stored_portfolio["return_series"][0]["net_return"] == 0.009
    assert stored_portfolio["performance"]["sharpe_ratio"] is None

    stress = {
        "experiment_name": "exp-1",
        "strategy": "static",
        "results": [{"scenario": "normal_volatility", "observation_count": 1}],
        "summary": {"valid_scenario_count": 1},
        "status": "completed",
    }
    store.save_stress_result(stress)
    stored_stress = store.get_stress_result("exp-1", "static")
    assert stored_stress is not None
    assert stored_stress["summary"]["valid_scenario_count"] == 1


def test_report_upsert_persists_generated_sections(tmp_path) -> None:
    store = DuckDBStorage(tmp_path / "reports.duckdb")
    record = {
        "experiment_name": "exp-2",
        "dataset_name": "dataset-2",
        "report_type": "summary",
        "generated_at": "2026-11-27T00:00:00+00:00",
        "sections": [],
        "status": "requested",
    }
    store.save_report(record)
    record["sections"] = [{"title": "Evidence", "content": "Persisted evidence", "metrics": {}}]
    record["status"] = "completed"
    store.upsert_report(record)
    stored = store.get_report("exp-2", "summary")
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["sections"][0]["content"] == "Persisted evidence"
