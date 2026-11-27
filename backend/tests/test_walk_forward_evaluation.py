from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.research.evaluation.walk_forward_evaluation import (
    WalkForwardEvaluationError,
    evaluate_walk_forward_predictions,
)
from backend.app.research.evaluation.forecast_metrics import calculate_qlike
from backend.app.research.evaluation.diebold_mariano import diebold_mariano_test
from backend.app.research.evaluation.bootstrap import bootstrap_metric
from backend.app.research.evaluation.model_confidence_set import build_loss_matrix


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



def test_qlike_uses_variance_ratio_for_volatility_inputs() -> None:
    actual_volatility = np.array([0.10, 0.20])
    forecast_volatility = np.array([0.20, 0.10])
    variance_ratio = np.square(actual_volatility / forecast_volatility)
    expected = float(np.mean(variance_ratio - np.log(variance_ratio) - 1.0))

    result = calculate_qlike(actual_volatility, forecast_volatility)

    assert result == pytest.approx(expected)



def test_qlike_definition_is_consistent_across_comparison_tools() -> None:
    actual = np.array([0.10, 0.20, 0.15, 0.18])
    forecast_a = np.array([0.12, 0.17, 0.14, 0.20])
    forecast_b = np.array([0.09, 0.22, 0.16, 0.17])

    def pointwise_qlike(forecast: np.ndarray) -> np.ndarray:
        ratio = np.square(actual / forecast)
        return ratio - np.log(ratio) - 1.0

    expected_a = pointwise_qlike(forecast_a)
    expected_b = pointwise_qlike(forecast_b)

    dm = diebold_mariano_test(
        actual,
        forecast_a,
        forecast_b,
        loss="qlike",
    )
    assert dm.mean_loss_difference == pytest.approx(
        float(np.mean(expected_a - expected_b))
    )

    loss_matrix = build_loss_matrix(
        actual,
        {"model_a": forecast_a, "model_b": forecast_b},
        loss="qlike",
    )
    np.testing.assert_allclose(loss_matrix["model_a"], expected_a)
    np.testing.assert_allclose(loss_matrix["model_b"], expected_b)

    bootstrap = bootstrap_metric(
        actual,
        forecast_a,
        metric="qlike",
        n_bootstrap=100,
        random_state=9,
    )
    assert bootstrap.estimate == pytest.approx(float(np.mean(expected_a)))
