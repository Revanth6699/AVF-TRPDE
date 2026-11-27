from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter

from backend.app.research.evaluation.model_confidence_set import (
    build_loss_matrix,
    model_confidence_set,
)


FEATURE_COLUMNS = (
    "return",
    "realized_volatility",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

ABLATIONS = (
    "A0",
    "A1",
    "A2",
    "A3",
)


def calculate_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    if len(actual) != len(forecast):
        raise ValueError(
            "Actual and forecast lengths must match."
        )

    if len(actual) == 0:
        raise ValueError(
            "Actual and forecast arrays must not be empty."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Actual values contain non-finite values."
        )

    if not np.isfinite(forecast).all():
        raise ValueError(
            "Forecast values contain non-finite values."
        )

    if np.any(actual <= 0):
        raise ValueError(
            "QLIKE requires strictly positive "
            "actual values."
        )

    if np.any(forecast <= 0):
        raise ValueError(
            "QLIKE requires strictly positive "
            "forecast values."
        )

    errors = actual - forecast

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(errors)
            )
        )
    )

    ratio = actual / forecast

    qlike = float(
        np.mean(
            ratio
            - np.log(ratio)
            - 1.0
        )
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "QLIKE": qlike,
    }


def validate_prediction_frame(
    name: str,
    predictions: pd.DataFrame,
) -> None:
    required_columns = {
        "timestamp",
        "asset_id",
        "prediction",
        "actual_volatility",
        "fold_id",
        "ablation",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise AssertionError(
            f"{name} missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(predictions) != 119:
        raise AssertionError(
            f"{name} expected 119 predictions, "
            f"got {len(predictions)}"
        )

    timestamps = pd.to_datetime(
        predictions["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise AssertionError(
            f"{name} contains invalid timestamps."
        )

    if predictions["asset_id"].isna().any():
        raise AssertionError(
            f"{name} contains missing asset IDs."
        )

    prediction_values = predictions[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    actual_values = predictions[
        "actual_volatility"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        prediction_values
    ).all():
        raise AssertionError(
            f"{name} contains non-finite predictions."
        )

    if not np.isfinite(
        actual_values
    ).all():
        raise AssertionError(
            f"{name} contains non-finite actuals."
        )

    if (
        prediction_values <= 0
    ).any():
        raise AssertionError(
            f"{name} contains non-positive predictions."
        )

    if (
        actual_values <= 0
    ).any():
        raise AssertionError(
            f"{name} contains non-positive actuals."
        )

    if not predictions[
        "timestamp"
    ].is_unique:
        raise AssertionError(
            f"{name} contains duplicate timestamps."
        )

    if predictions[
        "fold_id"
    ].nunique() != 119:
        raise AssertionError(
            f"{name} does not contain "
            "119 unique folds."
        )

    if (
        predictions["ablation"]
        .astype(str)
        .ne(name)
        .any()
    ):
        raise AssertionError(
            f"{name} contains incorrect "
            "ablation labels."
        )


def main() -> None:
    dataframe = make_controlled_data().copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe["asset_id"] = (
        dataframe["asset_id"]
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

    results = {}

    print(
        "\n========== "
        "PHASE 2 / A0-A3 MODEL COMPARISON "
        "=========="
    )

    print(
        "INPUT ROWS:",
        len(dataframe),
    )

    print(
        "ABLATIONS:",
        ABLATIONS,
    )

    for ablation in ABLATIONS:
        print(
            f"\n========== "
            f"RUNNING {ablation} "
            f"=========="
        )

        result = runner.run_ablation(
            dataframe,
            ablation=ablation,
            target_column="target",
            hmm_config=HMMConfig(
                n_components=3,
                random_state=42,
            ),
            feature_columns=FEATURE_COLUMNS,
        )

        results[ablation] = result

        print(
            f"{ablation} COMPLETE"
        )

        print(
            "DESCRIPTION:",
            result.description,
        )

        print(
            "FOLDS:",
            result.fold_count,
        )

        print(
            "PREDICTIONS:",
            len(result.predictions),
        )

    assert tuple(
        results.keys()
    ) == ABLATIONS

    metric_rows = []

    forecast_series = {}

    common_timestamps = None

    actual_reference = None

    for ablation in ABLATIONS:
        result = results[ablation]

        predictions = (
            result.predictions
            .copy()
        )

        predictions["timestamp"] = (
            pd.to_datetime(
                predictions["timestamp"]
            )
        )

        validate_prediction_frame(
            ablation,
            predictions,
        )

        if common_timestamps is None:
            common_timestamps = (
                predictions[
                    "timestamp"
                ]
                .reset_index(drop=True)
            )
        else:
            current_timestamps = (
                predictions[
                    "timestamp"
                ]
                .reset_index(drop=True)
            )

            if not current_timestamps.equals(
                common_timestamps
            ):
                raise AssertionError(
                    f"{ablation} timestamps "
                    "do not match the common "
                    "OOS evaluation sample."
                )

        actual = predictions[
            "actual_volatility"
        ].to_numpy(
            dtype=float
        )

        forecast = predictions[
            "prediction"
        ].to_numpy(
            dtype=float
        )

        if actual_reference is None:
            actual_reference = actual.copy()
        else:
            if not np.array_equal(
                actual,
                actual_reference,
            ):
                raise AssertionError(
                    f"{ablation} actual target "
                    "values do not match the "
                    "common OOS sample."
                )

        metrics = calculate_metrics(
            actual,
            forecast,
        )

        forecast_series[ablation] = (
            forecast
        )

        metric_rows.append(
            {
                "ablation": ablation,
                "description": result.description,
                "observations": len(
                    predictions
                ),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "QLIKE": metrics["QLIKE"],
            }
        )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    assert len(metrics_table) == 4

    assert (
        metrics_table[
            "observations"
        ]
        == 119
    ).all()

    metric_values = metrics_table[
        [
            "MAE",
            "RMSE",
            "QLIKE",
        ]
    ].to_numpy(
        dtype=float
    )

    assert np.isfinite(
        metric_values
    ).all()

    assert (
        metric_values >= 0
    ).all()

    print(
        "\n========== "
        "A0-A3 METRIC TABLE "
        "=========="
    )

    print(
        metrics_table.to_string(
            index=False
        )
    )

    loss_matrix = build_loss_matrix(
        actual_reference,
        forecast_series,
        loss="qlike",
    )

    assert loss_matrix.shape == (
        119,
        4,
    )

    assert list(
        loss_matrix.columns
    ) == list(
        ABLATIONS
    )

    assert np.isfinite(
        loss_matrix.to_numpy(
            dtype=float
        )
    ).all()

    print(
        "\n========== "
        "A0-A3 QLIKE LOSS MATRIX "
        "=========="
    )

    print(
        loss_matrix.describe()
        .to_string()
    )

    mcs = model_confidence_set(
        loss_matrix,
        confidence_level=0.90,
        n_bootstrap=500,
        block_length=5,
        random_state=42,
    )

    assert (
        mcs.observations
        == 119
    )

    assert (
        mcs.bootstrap_samples
        == 500
    )

    assert len(
        mcs.included_models
    ) >= 1

    combined_models = (
        set(mcs.included_models)
        | set(mcs.eliminated_models)
    )

    assert combined_models == set(
        ABLATIONS
    )

    assert (
        set(mcs.included_models)
        & set(mcs.eliminated_models)
    ) == set()

    assert np.isfinite(
        mcs.statistic
    )

    assert (
        0.0
        <= mcs.p_value
        <= 1.0
    )

    print(
        "\n========== "
        "A0-A3 MODEL CONFIDENCE SET "
        "=========="
    )

    print(
        "CONFIDENCE LEVEL:",
        mcs.confidence_level,
    )

    print(
        "OBSERVATIONS:",
        mcs.observations,
    )

    print(
        "BOOTSTRAP SAMPLES:",
        mcs.bootstrap_samples,
    )

    print(
        "STATISTIC:",
        mcs.statistic,
    )

    print(
        "P-VALUE:",
        mcs.p_value,
    )

    print(
        "INCLUDED MODELS:",
        mcs.included_models,
    )

    print(
        "ELIMINATED MODELS:",
        mcs.eliminated_models,
    )

    print(
        "\n========== "
        "A0-A3 MODEL COMPARISON "
        "VALIDATION "
        "=========="
    )

    print(
        "COMMON OOS TIMESTAMPS:",
        len(common_timestamps),
    )

    print(
        "COMMON OOS SAMPLE:",
        len(actual_reference),
    )

    print(
        "ALL ABLATIONS FINITE: True"
    )

    print(
        "TIMESTAMPS ALIGNED: True"
    )

    print(
        "ACTUALS ALIGNED: True"
    )

    print(
        "MAE/RMSE/QLIKE VALID: True"
    )

    print(
        "QLIKE LOSS MATRIX VALID: True"
    )

    print(
        "\nA0-A3 MODEL COMPARISON "
        "INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()