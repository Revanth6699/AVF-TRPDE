import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.walk_forward.runner import (
    WalkForwardRunner,
)

from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)

from backend.app.research.regimes.hmm import (
    HMMConfig,
)


# =========================================================
# Controlled research dataset
# =========================================================

dataframe = make_controlled_data()

# Keep the test deliberately smaller than the full
# 239-row controlled walk-forward run.
dataframe = dataframe.iloc[:160].reset_index(drop=True)


splitter = WalkForwardSplitter(
    train_size=120,
    test_size=1,
    step_size=1,
    expanding=False,
)

runner = WalkForwardRunner(splitter)


feature_columns = (
    "return",
    "realized_volatility",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

hmm_config = HMMConfig(
    n_components=3,
    random_state=42,
)


# =========================================================
# Model execution
# =========================================================

models = (
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "Regime-XGBoost",
)

results = {}


for model in models:

    print(
        f"\n========== RUNNING {model} =========="
    )

    result = runner.run_model(
        dataframe,
        model=model,
        target_column="target",
        feature_columns=feature_columns,
        hmm_config=hmm_config,
	ewma_decay=0.94,
    )

    predictions = result.predictions

    results[model] = result

    print(
        "FOLD COUNT:",
        result.fold_count,
    )

    print(
        "PREDICTION COUNT:",
        result.prediction_count,
    )

    print(
        "PREDICTION COLUMNS:",
        list(predictions.columns),
    )

    assert result.fold_count > 0
    assert result.prediction_count > 0

    assert len(predictions) == (
        result.prediction_count
    )

    assert predictions[
        "timestamp"
    ].notna().all()

    assert predictions[
        "asset_id"
    ].notna().all()

    assert np.isfinite(
        predictions[
            "prediction"
        ].to_numpy(dtype=float)
    ).all()

    assert np.isfinite(
        predictions[
            "actual_volatility"
        ].to_numpy(dtype=float)
    ).all()

    assert (
        predictions["fold_id"]
        .is_unique
    )

    # Every test timestamp must be after
    # the corresponding training window.
    for fold in result.folds:

        assert (
            fold.train_end
            < fold.test_start
        )

        

    print(
        "FINITE PREDICTIONS: True"
    )

    print(
        "FINITE ACTUALS: True"
    )

    print(
        "CHRONOLOGICAL FOLDS: True"
    )


# =========================================================
# Verify model outputs
# =========================================================

print(
    "\n========== MODEL EXECUTION SUMMARY =========="
)

for model, result in results.items():

    predictions = result.predictions

    print(
        f"{model}: "
        f"folds={result.fold_count}, "
        f"predictions={result.prediction_count}"
    )

    assert (
        predictions["prediction"]
        .notna()
        .all()
    )


# =========================================================
# Verify Regime-XGBoost regime features
# =========================================================

regime_predictions = (
    results[
        "Regime-XGBoost"
    ].predictions
)

regime_probability_columns = [
    column
    for column in regime_predictions.columns
    if column.startswith(
        "regime_probability_"
    )
]

print(
    "\n========== REGIME FEATURES =========="
)

print(
    "REGIME PROBABILITY COLUMNS:",
    regime_probability_columns,
)

assert len(
    regime_probability_columns
) == 3

regime_probabilities = (
    regime_predictions[
        regime_probability_columns
    ]
    .to_numpy(dtype=float)
)

assert np.isfinite(
    regime_probabilities
).all()

probability_sums = (
    regime_probabilities.sum(
        axis=1
    )
)

assert np.allclose(
    probability_sums,
    1.0,
    atol=1e-6,
)

print(
    "REGIME PROBABILITIES FINITE: True"
)

print(
    "REGIME PROBABILITIES SUM TO 1: True"
)


# =========================================================
# Verify OOS ordering
# =========================================================

for model, result in results.items():

    predictions = result.predictions

    timestamps = pd.to_datetime(
        predictions["timestamp"]
    )

    assert timestamps.is_monotonic_increasing

    assert (
        predictions[
            "fold_id"
        ].is_monotonic_increasing
    )


print(
    "\n========== WALK-FORWARD VALIDATION =========="
)

print(
    "ALL MODEL OUTPUTS CHRONOLOGICAL: True"
)

print(
    "ALL MODEL OUTPUTS FINITE: True"
)


# =========================================================
# Final validation
# =========================================================

assert set(results.keys()) == {
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "Regime-XGBoost",
}

print(
    "\nRESEARCH MODEL EXECUTION STACK OK"
)