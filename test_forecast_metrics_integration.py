from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter
from backend.app.research.regimes.hmm import HMMConfig

from backend.app.research.evaluation.forecast_metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_qlike,
)


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
        model="Regime-XGBoost",
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

    predictions = result.predictions.copy()

    actual = pd.to_numeric(
        predictions["actual_volatility"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    forecast = pd.to_numeric(
        predictions["prediction"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    print(
        "\n========== "
        "PHASE 2 / FORECAST METRICS "
        "=========="
    )

    print(
        "OOS OBSERVATIONS:",
        len(predictions),
    )

    print(
        "ACTUAL FINITE:",
        np.isfinite(actual).all(),
    )

    print(
        "FORECAST FINITE:",
        np.isfinite(forecast).all(),
    )

    print(
        "ACTUAL POSITIVE:",
        (actual > 0.0).all(),
    )

    print(
        "FORECAST POSITIVE:",
        (forecast > 0.0).all(),
    )

    mae = calculate_mae(
        actual,
        forecast,
    )

    rmse = calculate_rmse(
        actual,
        forecast,
    )

    qlike = calculate_qlike(
        actual,
        forecast,
    )

    print(
        "\n--- Forecast Metrics ---"
    )

    print(
        "MAE:",
        mae,
    )

    print(
        "RMSE:",
        rmse,
    )

    print(
        "QLIKE:",
        qlike,
    )

    print(
        "\n--- Metric Validation ---"
    )

    print(
        "MAE FINITE:",
        np.isfinite(mae),
    )

    print(
        "RMSE FINITE:",
        np.isfinite(rmse),
    )

    print(
        "QLIKE FINITE:",
        np.isfinite(qlike),
    )

    print(
        "MAE NONNEGATIVE:",
        mae >= 0.0,
    )

    print(
        "RMSE NONNEGATIVE:",
        rmse >= 0.0,
    )

    print(
        "QLIKE FINITE AND NONNEGATIVE:",
        np.isfinite(qlike)
        and qlike >= 0.0,
    )

    # Independent calculation used only to verify
    # the existing metric implementation.
    expected_mae = float(
        np.mean(
            np.abs(
                actual - forecast
            )
        )
    )

    expected_rmse = float(
        np.sqrt(
            np.mean(
                (
                    actual - forecast
                ) ** 2
            )
        )
    )

    expected_qlike = float(
        np.mean(
            (
                actual / forecast
            )
            -
            np.log(
                actual / forecast
            )
            -
            1.0
        )
    )

    print(
        "\n--- Independent Cross-Check ---"
    )

    print(
        "EXPECTED MAE:",
        expected_mae,
    )

    print(
        "EXPECTED RMSE:",
        expected_rmse,
    )

    print(
        "EXPECTED QLIKE:",
        expected_qlike,
    )

    print(
        "MAE MATCH:",
        np.isclose(
            mae,
            expected_mae,
            rtol=1e-10,
            atol=1e-12,
        ),
    )

    print(
        "RMSE MATCH:",
        np.isclose(
            rmse,
            expected_rmse,
            rtol=1e-10,
            atol=1e-12,
        ),
    )

    print(
        "QLIKE MATCH:",
        np.isclose(
            qlike,
            expected_qlike,
            rtol=1e-10,
            atol=1e-12,
        ),
    )

    if not np.isfinite(actual).all():
        raise AssertionError(
            "Actual volatility contains non-finite values."
        )

    if not np.isfinite(forecast).all():
        raise AssertionError(
            "Forecast contains non-finite values."
        )

    if not (actual > 0.0).all():
        raise AssertionError(
            "Actual volatility must be strictly positive."
        )

    if not (forecast > 0.0).all():
        raise AssertionError(
            "Forecast must be strictly positive."
        )

    if not np.isclose(
        mae,
        expected_mae,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "MAE implementation mismatch."
        )

    if not np.isclose(
        rmse,
        expected_rmse,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "RMSE implementation mismatch."
        )

    if not np.isclose(
        qlike,
        expected_qlike,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "QLIKE implementation mismatch."
        )

    print(
        "\nFORECAST METRICS INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()