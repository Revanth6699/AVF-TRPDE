from __future__ import annotations
import gzip, hashlib, json, os, sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import pandas as pd
import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "real_market"
OUT = ROOT / "validation_output"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["AVF_REPORT_OUTPUT_DIR"] = str(ROOT / "reports" / "real_market_validation")
COMMIT = "b57f460fab1afb82705eb813cee2ef4afb1f6848"
BASE = f"https://raw.githubusercontent.com/alessiostomeo/btc-cycle-data/{COMMIT}/bitstamp"
DAILY_URL = f"{BASE}/1day/BTCUSD_bitstamp_daily_2012-2026.csv"
HOURLY_URL = f"{BASE}/1h/BTCUSD_bitstamp_hourly_2012-2026.csv.gz"

def fetch(url):
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content

def canonical(frame):
    frame = frame.copy()
    time_col = next((c for c in ("datetime_utc", "timestamp", "datetime", "date") if c in frame.columns), None)
    if time_col is None:
        raise RuntimeError(f"No recognized timestamp column: {list(frame.columns)}")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing OHLCV columns: {sorted(required - set(frame.columns))}")
    frame["timestamp"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise RuntimeError("Invalid source timestamps detected.")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(None)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise RuntimeError("Missing or non-numeric OHLCV values detected.")
    frame["asset_id"] = "BTCUSD"
    frame = frame[["timestamp", "asset_id", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep=False).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No valid observations after canonicalization.")
    return frame

def call(client, path, payload):
    response = client.post(path, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned {response.status_code}: {response.text[:3000]}")
    return response.json()

def main():
    daily_bytes = fetch(DAILY_URL)
    hourly_gz = fetch(HOURLY_URL)
    daily = canonical(pd.read_csv(BytesIO(daily_bytes)))
    hourly = canonical(pd.read_csv(BytesIO(gzip.decompress(hourly_gz))))
    end = min(daily.timestamp.max(), hourly.timestamp.max()).normalize()
    start = end - pd.DateOffset(years=8)
    upper = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    daily = daily.loc[daily.timestamp.between(start, upper)].reset_index(drop=True)
    hourly = hourly.loc[hourly.timestamp.between(start, upper)].reset_index(drop=True)
    if len(daily) < 1500 or len(hourly) < 40000:
        raise RuntimeError(f"Insufficient overlap: {len(daily)} daily and {len(hourly)} hourly rows.")
    daily_path = DATA / "BTCUSD_bitstamp_daily_8y.csv"
    hourly_path = DATA / "BTCUSD_bitstamp_hourly_8y.csv"
    daily.to_csv(daily_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    digest = hashlib.sha256(daily_bytes + hourly_gz).hexdigest()
    start_iso, end_iso = daily.timestamp.min().isoformat(), daily.timestamp.max().isoformat()
    intraday_start, intraday_end = hourly.timestamp.min().isoformat(), hourly.timestamp.max().isoformat()
    manifest = {
        "dataset": "Bitstamp BTC/USD spot OHLCV",
        "asset": "BTCUSD",
        "source_repository": "https://github.com/alessiostomeo/btc-cycle-data",
        "source_commit": COMMIT,
        "daily_source_url": DAILY_URL,
        "hourly_source_url": HOURLY_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest,
        "daily": {"file": daily_path.name, "rows": len(daily), "start": start_iso, "end": end_iso, "frequency": "1day"},
        "intraday": {"file": hourly_path.name, "rows": len(hourly), "start": intraday_start, "end": intraday_end, "frequency": "1h"},
        "note": "Real historical BTC/USD exchange data. Single-asset validation only."
    }
    (DATA / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA / "SOURCE_AND_ATTRIBUTION.md").write_text(
        "Source: community-maintained btc-cycle-data repository, pinned to commit " + COMMIT +
        ". The repository documents Bitstamp BTC/USD daily and hourly OHLCV and UTC timestamps. "
        "Its README does not state a separate data license for this mirror; confirm terms with the maintainer before redistribution or commercial use. "
        "Files use AVF-TRPDE canonical columns: timestamp, asset_id, open, high, low, close, volume.\\n",
        encoding="utf-8",
    )

    from backend.app.main import app
    client = TestClient(app)
    dataset_name = "bitstamp_btcusd_8y_real"
    experiment_name = "real_btcusd_walk_forward_8y"
    call(client, "/datasets", {
        "name": dataset_name, "source": "csv", "frequency": "daily", "assets": ["BTCUSD"],
        "start_timestamp": start_iso, "end_timestamp": end_iso, "row_count": len(daily), "fingerprint": digest
    })
    params = {
        "dataset_name": dataset_name, "path": str(daily_path), "intraday_path": str(hourly_path),
        "intraday_start_timestamp": intraday_start, "intraday_end_timestamp": intraday_end,
        "train_size": 2550, "test_size": 1, "step_size": 1, "expanding": True,
        "ewma_decay": 0.94, "hmm_n_components": 3, "hmm_n_iter": 150, "hmm_random_state": 42,
        "xgb_n_estimators": 25, "xgb_max_depth": 3, "xgb_learning_rate": 0.05,
        "xgb_subsample": 0.8, "xgb_colsample_bytree": 0.8,
        "bootstrap_samples": 200, "bootstrap_random_state": 42, "confidence_level": 0.95
    }
    models = ["EWMA", "GJR-GARCH", "XGBoost", "Regime-XGBoost"]
    call(client, "/experiments", {
        "name": experiment_name,
        "description": "Real-data walk-forward validation using Bitstamp BTC/USD daily and hourly OHLCV.",
        "models": models, "metrics": ["MAE", "RMSE", "QLIKE"],
        "risk_measures": ["VaR95", "VaR99", "ES95"], "parameters": params,
        "metadata": {"asset": "BTCUSD", "market": "Bitstamp BTC/USD spot", "data_frequency": ["1day", "1h"]}
    })
    experiment = call(client, "/experiments/run", {"experiment_names": [experiment_name]})
    (OUT / "experiment_run.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    if not experiment.get("all_successful"):
        raise RuntimeError(f"Model execution failed: {experiment}")
    forecasts = {}
    for model in models:
        forecasts[model] = call(client, "/forecasts", {
            "experiment_name": experiment_name, "dataset_name": dataset_name, "model": model,
            "assets": ["BTCUSD"], "start_timestamp": start_iso, "end_timestamp": end_iso
        })
        pd.DataFrame(forecasts[model]["forecasts"]).to_csv(OUT / f"forecasts_{model.lower().replace('-', '_')}.csv", index=False)
    (OUT / "forecasts.json").write_text(json.dumps(forecasts, indent=2), encoding="utf-8")
    risk = call(client, "/risk", {
        "experiment_name": experiment_name, "dataset_name": dataset_name, "model": "EWMA",
        "assets": ["BTCUSD"], "confidence_levels": [0.95, 0.99],
        "start_timestamp": start_iso, "end_timestamp": end_iso
    })
    (OUT / "risk.json").write_text(json.dumps(risk, indent=2), encoding="utf-8")
    pd.DataFrame(risk["risk_points"]).to_csv(OUT / "risk_points_ewma.csv", index=False)
    # Persist formal out-of-sample VaR coverage and independence tests.
    # The risk API generates the FHS VaR/ES series; this endpoint evaluates
    # realized returns against the stored VaR thresholds using Kupiec and
    # Christoffersen tests. Expected Shortfall is reported as a risk estimate,
    # not described as statistically backtested here.
    risk_backtest = call(client, "/risk/backtest", {
        "experiment_name": experiment_name, "dataset_name": dataset_name, "model": "EWMA",
        "assets": ["BTCUSD"], "confidence_levels": [0.95, 0.99],
        "start_timestamp": start_iso, "end_timestamp": end_iso
    })
    (OUT / "risk_backtest.json").write_text(
        json.dumps(risk_backtest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(risk_backtest["results"]).to_csv(
        OUT / "risk_backtest_ewma.csv", index=False
    )
    portfolio = call(client, "/portfolios", {
        "experiment_name": experiment_name, "dataset_name": dataset_name, "strategy": "static", "model": "EWMA",
        "assets": ["BTCUSD"], "start_timestamp": start_iso, "end_timestamp": end_iso,
        "volatility_target": 0.01, "max_position": 1.0, "max_turnover": 1.0,
        "transaction_cost": 0.0005, "covariance_window": 60
    })
    (OUT / "portfolio.json").write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    pd.DataFrame(portfolio["return_series"]).to_csv(OUT / "portfolio_returns_static_ewma.csv", index=False)
    stress = call(client, "/stress/run", {"experiment_name": experiment_name, "strategy": "static"})
    (OUT / "stress.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")
    report = call(client, "/reports/run", {
        "experiment_name": experiment_name, "dataset_name": dataset_name, "report_type": "real_market_validation",
        "start_timestamp": start_iso, "end_timestamp": end_iso,
        "include_forecasts": True, "include_risk": True, "include_portfolio": True,
        "include_stress": True, "include_statistical_tests": True
    })
    (OUT / "report_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "status": "completed", "dataset": manifest, "experiment": experiment,
        "forecast_observations_by_model": {k: v["observation_count"] for k, v in forecasts.items()},
        "risk_observations": risk["observation_count"],
        "risk_backtest_observations": risk_backtest["observation_count"],
        "risk_backtest_results": risk_backtest["results"],
        "risk_backtest_warnings": risk_backtest["warnings"],
        "portfolio_observations": portfolio["observation_count"],
        "portfolio_performance": portfolio["performance"], "stress_scenario_count": stress["scenario_count"],
        "stress_summary": stress["summary"], "report": report,
        "caveats": [
            "Single-asset BTC/USD test; multi-asset covariance and cross-asset behavior are not validated.",
            "Eight years of hourly observations are used so realized-volatility targets exist across the daily training history.",
            "Historical metrics do not imply future performance or investment suitability."
        ]
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": "completed", "daily_rows": len(daily), "hourly_rows": len(hourly),
        "date_start": start_iso, "date_end": end_iso,
        "forecast_observations": summary["forecast_observations_by_model"],
        "risk_observations": risk["observation_count"],
        "risk_backtest_observations": risk_backtest["observation_count"],
        "risk_backtest_result_count": len(risk_backtest["results"]),
        "risk_backtest_warnings": risk_backtest["warnings"],
        "portfolio_observations": portfolio["observation_count"],
        "stress_scenario_count": stress["scenario_count"], "portfolio_performance": portfolio["performance"]
    }, indent=2, default=str))

if __name__ == "__main__":
    main()
