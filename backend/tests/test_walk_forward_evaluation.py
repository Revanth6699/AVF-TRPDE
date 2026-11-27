from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.research.evaluation.walk_forward_evaluation import (
    WalkForwardEvaluationError,
    evaluate_walk_forward_predictions,
)


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=16, freq="B")
    for asset in ("ASSET_A", "ASSET_B"):
        actual = 0.02 + 0.004 * np.sin(np.arange(16) / 2.0)
        for model, noise_scale in (("XGBoost", 0.002), ("Regime-XGBoost", 0.0015)):
            forecasts = actual + rng.normal(0.0, noise_scale, size=16)
            forecasts = np.maximum(forecasts, 0.0001)
            for timestamp, y, yhat in zip(timestamps, actual, forecasts):
                rows.append({
                    "timestamp": timestamp,
                    "asset_id": asset,
                    "model": model,
                    "prediction": yhat,
                    "actual_volatility": y,
                })
    return pd.DataFrame(rows)


def test_evaluation_returns_per_asset_metrics_intervals_dm_and_mcs() -> None:
    result = evaluate_walk_forward_predictions(
        _predictions(),
        metrics=("MAE", "RMSE", "QLIKE"),
        n_bootstrap=100,
        random_state=11,
    )

    assert result["method"] == "per_asset_walk_forward_oos"
    assert set(result["assets"]) == {"ASSET_A", "ASSET_B"}

    for asset in result["assets"].values():
        assert set(asset["model_metrics"]) == {"XGBoost", "Regime-XGBoost"}
        assert asset["model_metrics"]["XGBoost"]["MAE"]["value"] >= 0
        interval = asset["model_metrics"]["XGBoost"]["MAE"]["confidence_interval"]
        assert interval["lower"] <= interval["upper"]
        assert "Regime-XGBoost" in asset["pairwise_diebold_mariano"]["XGBoost"]
        assert asset["model_confidence_set"]["observations"] == 16


def test_evaluation_rejects_duplicate_model_timestamp_rows() -> None:
    frame = _predictions()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(WalkForwardEvaluationError, match="Duplicate"):
        evaluate_walk_forward_predictions(frame, n_bootstrap=100)


def test_evaluation_requires_positive_realized_volatility_and_forecasts() -> None:
    frame = _predictions()
    frame.loc[0, "prediction"] = 0.0

    with pytest.raises(WalkForwardEvaluationError, match="strictly positive"):
        evaluate_walk_forward_predictions(frame, n_bootstrap=100)
