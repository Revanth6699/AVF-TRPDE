from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.evaluation.diebold_mariano import (
    diebold_mariano_test,
    calculate_loss_difference,
)
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import (
    WalkForwardRunner,
)
from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)


def main() -> None:
    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    runner = WalkForwardRunner(
        WalkForwardSplitter(
            train_size=120,
            test_size=1,
            step_size=1,
            expanding=False,
        )
    )

    result_a = runner.run_model(
        df,
        model="XGBoost",
        target_column="target",
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

    result_b = runner.run_model(
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

    predictions_a = result_a.predictions[
        [
            "timestamp",
            "asset_id",
            "prediction",
            "actual_volatility",
        ]
    ].copy()

    predictions_b = result_b.predictions[
        [
            "timestamp",
            "asset_id",
            "prediction",
            "actual_volatility",
        ]
    ].copy()

    predictions_a = predictions_a.rename(
        columns={
            "prediction": "forecast_a",
            "actual_volatility": "actual_a",
        }
    )

    predictions_b = predictions_b.rename(
        columns={
            "prediction": "forecast_b",
            "actual_volatility": "actual_b",
        }
    )

    merged = predictions_a.merge(
        predictions_b,
        on=[
            "timestamp",
            "asset_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if not np.allclose(
        merged["actual_a"].to_numpy(dtype=float),
        merged["actual_b"].to_numpy(dtype=float),
    ):
        raise AssertionError(
            "Actual volatility values do not match "
            "between model outputs."
        )

    actual = merged[
        "actual_a"
    ].to_numpy(dtype=float)

    forecast_a = merged[
        "forecast_a"
    ].to_numpy(dtype=float)

    forecast_b = merged[
        "forecast_b"
    ].to_numpy(dtype=float)

    print(
        "\n========== "
        "PHASE 2 / DIEBOLD-MARIANO INTEGRATION "
        "=========="
    )

    print(
        "MODEL A: XGBoost"
    )

    print(
        "MODEL B: Regime-XGBoost"
    )

    print(
        "OOS OBSERVATIONS:",
        len(merged),
    )

    print(
        "ACTUAL FINITE:",
        np.isfinite(actual).all(),
    )

    print(
        "FORECAST A FINITE:",
        np.isfinite(forecast_a).all(),
    )

    print(
        "FORECAST B FINITE:",
        np.isfinite(forecast_b).all(),
    )

    print(
        "TIMESTAMPS MATCH:",
        len(merged)
        == len(predictions_a)
        == len(predictions_b),
    )

    for loss in (
        "absolute",
        "squared",
        "qlike",
    ):
        result = diebold_mariano_test(
            actual,
            forecast_a,
            forecast_b,
            loss=loss,
            alternative="two_sided",
        )

        loss_difference = (
            calculate_loss_difference(
                actual,
                forecast_a,
                forecast_b,
                loss=loss,
            )
        )

        print(
            f"\n--- DM / {loss.upper()} ---"
        )

        print(
            "STATISTIC:",
            result.statistic,
        )

        print(
            "P-VALUE:",
            result.p_value,
        )

        print(
            "MEAN LOSS DIFFERENCE:",
            result.mean_loss_difference,
        )

        print(
            "OBSERVATIONS:",
            result.observations,
        )

        print(
            "LOSS FUNCTION:",
            result.loss_function,
        )

        print(
            "ALTERNATIVE:",
            result.alternative,
        )

        print(
            "LOSS DIFFERENCE FINITE:",
            np.isfinite(
                loss_difference
            ).all(),
        )

        print(
            "LOSS DIFFERENCE MEAN MATCH:",
            np.isclose(
                result.mean_loss_difference,
                np.mean(loss_difference),
            ),
        )

        print(
            "P-VALUE VALID:",
            0.0 <= result.p_value <= 1.0,
        )

    squared_result = diebold_mariano_test(
        actual,
        forecast_a,
        forecast_b,
        loss="squared",
        alternative="two_sided",
    )

    print(
        "\n--- DM Alternative Hypotheses ---"
    )

    less_result = diebold_mariano_test(
        actual,
        forecast_a,
        forecast_b,
        loss="squared",
        alternative="less",
    )

    greater_result = diebold_mariano_test(
        actual,
        forecast_a,
        forecast_b,
        loss="squared",
        alternative="greater",
    )

    print(
        "TWO-SIDED P-VALUE:",
        squared_result.p_value,
    )

    print(
        "LESS P-VALUE:",
        less_result.p_value,
    )

    print(
        "GREATER P-VALUE:",
        greater_result.p_value,
    )

    print(
        "ALTERNATIVE P-VALUES VALID:",
        all(
            0.0 <= value <= 1.0
            for value in (
                less_result.p_value,
                greater_result.p_value,
            )
        ),
    )

    print(
        "\nDIEBOLD-MARIANO "
        "INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()