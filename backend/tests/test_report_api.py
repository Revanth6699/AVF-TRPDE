from __future__ import annotations

from backend.app.api.v1 import reports
from backend.app.schemas.report import ReportRequest
from backend.app.storage import DuckDBStorage


def test_report_api_generates_and_persists_markdown_and_json(tmp_path, monkeypatch) -> None:
    store = DuckDBStorage(tmp_path / "report-api.duckdb")
    store.save_dataset(
        {
            "name": "dataset-r",
            "source": "parquet",
            "frequency": "daily",
            "assets": ["AAA"],
            "start_timestamp": "2024-01-01",
            "end_timestamp": "2024-02-01",
            "row_count": 20,
            "fingerprint": "fingerprint-r",
            "status": "registered",
        }
    )
    store.save_experiment(
        {
            "name": "experiment-r",
            "description": "",
            "models": ["EWMA"],
            "metrics": ["MAE", "RMSE", "QLIKE"],
            "risk_measures": [],
            "parameters": {"dataset_name": "dataset-r"},
            "metadata": {},
            "status": "completed",
        }
    )
    monkeypatch.setattr(reports, "storage", store)
    monkeypatch.setenv("AVF_REPORT_OUTPUT_DIR", str(tmp_path))

    request = ReportRequest(
        experiment_name="experiment-r",
        dataset_name="dataset-r",
        report_type="research_summary",
        start_timestamp="2024-01-01T00:00:00",
        end_timestamp="2024-02-01T00:00:00",
    )
    registered = reports.create_report(request)
    assert registered.status == "requested"

    result = reports.run_report(request)
    assert result.status == "completed"
    assert {artifact.artifact_type for artifact in result.reports[0].artifacts} == {"json", "markdown"}
    assert all((tmp_path / artifact.path).exists() for artifact in result.reports[0].artifacts)

    stored = store.get_report("experiment-r", "research_summary")
    assert stored is not None
    assert stored["status"] == "completed"
    assert any(section["title"] == "Interpretation and limitations" for section in stored["sections"])
