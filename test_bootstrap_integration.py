from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter

from backend.app.research.evaluation.bootstrap import (
    bootstrap_metric,
    bootstrap_difference,
)


def _run_model(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    runner = WalkForwardRunner(
        WalkForwardSplitter(
            train_size=120,
            test_size=1,
            step_size=1,
            expanding=False,
        )
    )

    result = runner.run_model(
        df,
        model=model,
        target_column="target",
        hmm_config=HMMConfig(
            n_components=3,
            random_state=42,
        ),
        feature_columns=(
            "return",
            "realized_volatility",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ),
    )

    predictions = result.predictions[
        [
            "timestamp",
            "asset_id",
            "prediction",
            "actual_volatility",
        ]
    ].copy()

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

    predictions["asset_id"] = (
        predictions["asset_id"]
        .astype(str)
        .str.upper()
    )

    predictions["prediction"] = (
        predictions["prediction"]
        .astype(float)
    )

    predictions["actual_volatility"] = (
        predictions["actual_volatility"]
        .astype(float)
    )

    predictions = predictions.rename(
        columns={
            "prediction": "forecast",
            "actual_volatility": "actual",
        }
    )

    return predictions


def _print_bootstrap_result(
    name: str,
    result,
) -> None:
    print(f"\n--- {name} ---")
    print(
        "ESTIMATE:",
        result.estimate,
    )
    print(
        "STANDARD ERROR:",
        result.standard_error,
    )
    print(
        "CONFIDENCE LEVEL:",
        result.confidence_level,
    )
    print(
        "LOWER 95%:",
        result.lower_bound,
    )
    print(
        "UPPER 95%:",
        result.upper_bound,
    )
    print(
        "BOOTSTRAP SAMPLES:",
        result.bootstrap_samples,
    )

    assert np.isfinite(result.estimate)
    assert np.isfinite(result.standard_error)
    assert np.isfinite(result.lower_bound)
    assert np.isfinite(result.upper_bound)

    assert result.lower_bound <= result.estimate
    assert result.estimate <= result.upper_bound


def main() -> None:
    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["asset_id"] = (
        df["asset_id"]
        .astype(str)
        .str.upper()
    )

    print(
        "\n========== "
        "PHASE 2 / BOOTSTRAP INTEGRATION "
        "=========="
    )

    print(
        "INPUT ROWS:",
        len(df),
    )

    xgb = _run_model(
        df,
        "XGBoost",
    )

    regime_xgb = _run_model(
        df,
        "Regime-XGBoost",
    )

    assert len(xgb) == len(regime_xgb)
    assert len(xgb) > 1

    assert np.isfinite(
        xgb["actual"].to_numpy()
    ).all()

    assert np.isfinite(
        xgb["forecast"].to_numpy()
    ).all()

    assert np.isfinite(
        regime_xgb["forecast"].to_numpy()
    ).all()

    assert (
        xgb["timestamp"].to_numpy()
        == regime_xgb["timestamp"].to_numpy()
    ).all()

    print(
        "OOS OBSERVATIONS:",
        len(xgb),
    )

    print(
        "TIMESTAMPS MATCH:",
        (
            xgb["timestamp"].to_numpy()
            == regime_xgb["timestamp"].to_numpy()
        ).all(),
    )

    actual = xgb["actual"].to_numpy(
        dtype=float
    )

    xgb_forecast = xgb["forecast"].to_numpy(
        dtype=float
    )

    regime_forecast = (
        regime_xgb["forecast"]
        .to_numpy(dtype=float)
    )

    assert np.all(actual > 0)
    assert np.all(xgb_forecast > 0)
    assert np.all(regime_forecast > 0)

    print(
        "ACTUAL POSITIVE:",
        bool(np.all(actual > 0)),
    )

    print(
        "XGBOOST FORECAST POSITIVE:",
        bool(np.all(xgb_forecast > 0)),
    )

    print(
        "REGIME-XGBOOST FORECAST POSITIVE:",
        bool(np.all(regime_forecast > 0)),
    )

    # ---------------------------------------------------------
    # Bootstrap confidence intervals for XGBoost
    # ---------------------------------------------------------

    xgb_mae = bootstrap_metric(
        actual,
        xgb_forecast,
        metric="mae",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    xgb_rmse = bootstrap_metric(
        actual,
        xgb_forecast,
        metric="rmse",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    xgb_qlike = bootstrap_metric(
        actual,
        xgb_forecast,
        metric="qlike",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    _print_bootstrap_result(
        "XGBoost / MAE",
        xgb_mae,
    )

    _print_bootstrap_result(
        "XGBoost / RMSE",
        xgb_rmse,
    )

    _print_bootstrap_result(
        "XGBoost / QLIKE",
        xgb_qlike,
    )

    # ---------------------------------------------------------
    # Bootstrap confidence intervals for Regime-XGBoost
    # ---------------------------------------------------------

    regime_mae = bootstrap_metric(
        actual,
        regime_forecast,
        metric="mae",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    regime_rmse = bootstrap_metric(
        actual,
        regime_forecast,
        metric="rmse",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    regime_qlike = bootstrap_metric(
        actual,
        regime_forecast,
        metric="qlike",
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    _print_bootstrap_result(
        "Regime-XGBoost / MAE",
        regime_mae,
    )

    _print_bootstrap_result(
        "Regime-XGBoost / RMSE",
        regime_rmse,
    )

    _print_bootstrap_result(
        "Regime-XGBoost / QLIKE",
        regime_qlike,
    )

    # ---------------------------------------------------------
    # Paired bootstrap differences
    #
    # Positive difference means:
    #   XGBoost metric - Regime-XGBoost metric
    #
    # For error metrics, negative means Regime-XGBoost
    # has lower error on the observed sample.
    # ---------------------------------------------------------

    xgb_abs_error = np.abs(
        actual - xgb_forecast
    )

    regime_abs_error = np.abs(
        actual - regime_forecast
    )

    xgb_squared_error = np.square(
        actual - xgb_forecast
    )

    regime_squared_error = np.square(
        actual - regime_forecast
    )

    xgb_qlike_loss = (
        actual / xgb_forecast
        - np.log(actual / xgb_forecast)
        - 1.0
    )

    regime_qlike_loss = (
        actual / regime_forecast
        - np.log(actual / regime_forecast)
        - 1.0
    )

    mae_difference = bootstrap_difference(
        xgb_abs_error,
        regime_abs_error,
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    rmse_difference = bootstrap_difference(
        xgb_squared_error,
        regime_squared_error,
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    qlike_difference = bootstrap_difference(
        xgb_qlike_loss,
        regime_qlike_loss,
        confidence_level=0.95,
        n_bootstrap=500,
        random_state=42,
    )

    _print_bootstrap_result(
        "PAIRED DIFFERENCE / MAE LOSS",
        mae_difference,
    )

    _print_bootstrap_result(
        "PAIRED DIFFERENCE / SQUARED LOSS",
        rmse_difference,
    )

    _print_bootstrap_result(
        "PAIRED DIFFERENCE / QLIKE LOSS",
        qlike_difference,
    )

    # ---------------------------------------------------------
    # Independent metric consistency checks
    # ---------------------------------------------------------

    expected_xgb_mae = float(
        np.mean(xgb_abs_error)
    )

    expected_xgb_rmse = float(
        np.sqrt(
            np.mean(xgb_squared_error)
        )
    )

    expected_xgb_qlike = float(
        np.mean(xgb_qlike_loss)
    )

    expected_regime_mae = float(
        np.mean(regime_abs_error)
    )

    expected_regime_rmse = float(
        np.sqrt(
            np.mean(regime_squared_error)
        )
    )

    expected_regime_qlike = float(
        np.mean(regime_qlike_loss)
    )

    print(
        "\n--- Independent Metric Cross-Checks ---"
    )

    print(
        "XGBoost MAE MATCH:",
        np.isclose(
            xgb_mae.estimate,
            expected_xgb_mae,
        ),
    )

    print(
        "XGBoost RMSE MATCH:",
        np.isclose(
            xgb_rmse.estimate,
            expected_xgb_rmse,
        ),
    )

    print(
        "XGBoost QLIKE MATCH:",
        np.isclose(
            xgb_qlike.estimate,
            expected_xgb_qlike,
        ),
    )

    print(
        "Regime-XGBoost MAE MATCH:",
        np.isclose(
            regime_mae.estimate,
            expected_regime_mae,
        ),
    )

    print(
        "Regime-XGBoost RMSE MATCH:",
        np.isclose(
            regime_rmse.estimate,
            expected_regime_rmse,
        ),
    )

    print(
        "Regime-XGBoost QLIKE MATCH:",
        np.isclose(
            regime_qlike.estimate,
            expected_regime_qlike,
        ),
    )

    assert np.isclose(
        xgb_mae.estimate,
        expected_xgb_mae,
    )

    assert np.isclose(
        xgb_rmse.estimate,
        expected_xgb_rmse,
    )

    assert np.isclose(
        xgb_qlike.estimate,
        expected_xgb_qlike,
    )

    assert np.isclose(
        regime_mae.estimate,
        expected_regime_mae,
    )

    assert np.isclose(
        regime_rmse.estimate,
        expected_regime_rmse,
    )

    assert np.isclose(
        regime_qlike.estimate,
        expected_regime_qlike,
    )

    # ---------------------------------------------------------
    # Paired-difference consistency checks
    # ---------------------------------------------------------

    expected_mae_difference = float(
        np.mean(
            xgb_abs_error
            - regime_abs_error
        )
    )

    expected_rmse_difference = float(
        np.mean(
            xgb_squared_error
            - regime_squared_error
        )
    )

    expected_qlike_difference = float(
        np.mean(
            xgb_qlike_loss
            - regime_qlike_loss
        )
    )

    print(
        "\n--- Paired Difference Cross-Checks ---"
    )

    print(
        "MAE DIFFERENCE MATCH:",
        np.isclose(
            mae_difference.estimate,
            expected_mae_difference,
        ),
    )

    print(
        "SQUARED LOSS DIFFERENCE MATCH:",
        np.isclose(
            rmse_difference.estimate,
            expected_rmse_difference,
        ),
    )

    print(
        "QLIKE DIFFERENCE MATCH:",
        np.isclose(
            qlike_difference.estimate,
            expected_qlike_difference,
        ),
    )

    assert np.isclose(
        mae_difference.estimate,
        expected_mae_difference,
    )

    assert np.isclose(
        rmse_difference.estimate,
        expected_rmse_difference,
    )

    assert np.isclose(
        qlike_difference.estimate,
        expected_qlike_difference,
    )

    print(
        "\nBOOTSTRAP INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()