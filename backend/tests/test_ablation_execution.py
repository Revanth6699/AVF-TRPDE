import numpy as np

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
# Controlled data
# =========================================================

dataframe = make_controlled_data()
dataframe = dataframe.iloc[:160].reset_index(drop=True)


# =========================================================
# Walk-forward configuration
# =========================================================

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
# A0-A3 ablation
# =========================================================

print("========== RUNNING A0-A3 ABLATION ==========")

results = {}

for ablation_name in (
    "A0",
    "A1",
    "A2",
    "A3",
):
    print(
        f"\n========== RUNNING {ablation_name} =========="
    )

    results[ablation_name] = runner.run_ablation(
        dataframe,
        ablation=ablation_name,
        target_column="target",
        feature_columns=feature_columns,
        hmm_config=hmm_config,
    )

# =========================================================
# Expected ablation definitions
# =========================================================

expected = {
    "A0": "XGBoost",
    "A1": "XGBoost + GJR-GARCH",
    "A2": "XGBoost + HMM",
    "A3": "XGBoost + GJR-GARCH + HMM",
}


print("\n========== ABLATION RESULTS ==========")

print(results)


# =========================================================
# Verify result structure
# =========================================================

assert results is not None

for ablation_name in expected:

    assert ablation_name in results

    ablation = results[ablation_name]

    print(
        f"{ablation_name}:",
        expected[ablation_name],
    )

    print(
        "  FOLDS:",
        ablation.fold_count,
    )

    print(
        "  PREDICTIONS:",
        ablation.prediction_count,
    )

    assert ablation.fold_count > 0
    assert ablation.prediction_count > 0

    predictions = ablation.predictions

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
        predictions["timestamp"]
        .is_monotonic_increasing
    )


# =========================================================
# Verify A0
# =========================================================

a0 = results["A0"].predictions

assert "prediction" in a0.columns

assert "actual_volatility" in a0.columns


# =========================================================
# Verify A1 has GARCH feature
# =========================================================

a1 = results["A1"].predictions

assert "garch_forecast" in a1.columns

assert np.isfinite(
    a1["garch_forecast"]
    .to_numpy(dtype=float)
).all()


# =========================================================
# Verify A2 has HMM probabilities
# =========================================================

a2 = results["A2"].predictions

a2_regime_columns = [
    column
    for column in a2.columns
    if column.startswith(
        "regime_probability_"
    )
]

print(
    "\nA2 REGIME COLUMNS:",
    a2_regime_columns,
)

assert len(a2_regime_columns) == 3

a2_probabilities = (
    a2[a2_regime_columns]
    .to_numpy(dtype=float)
)

assert np.isfinite(
    a2_probabilities
).all()

assert np.allclose(
    a2_probabilities.sum(axis=1),
    1.0,
    atol=1e-6,
)


# =========================================================
# Verify A3 has both GARCH + HMM
# =========================================================

a3 = results["A3"].predictions

assert "garch_forecast" in a3.columns

a3_regime_columns = [
    column
    for column in a3.columns
    if column.startswith(
        "regime_probability_"
    )
]

assert len(a3_regime_columns) == 3

assert np.isfinite(
    a3["garch_forecast"]
    .to_numpy(dtype=float)
).all()

a3_probabilities = (
    a3[a3_regime_columns]
    .to_numpy(dtype=float)
)

assert np.isfinite(
    a3_probabilities
).all()

assert np.allclose(
    a3_probabilities.sum(axis=1),
    1.0,
    atol=1e-6,
)


# =========================================================
# Final
# =========================================================

print("\n========== ABLATION VALIDATION ==========")

print("A0 XGBoost: True")
print("A1 XGBoost + GJR-GARCH: True")
print("A2 XGBoost + HMM: True")
print("A3 XGBoost + GJR-GARCH + HMM: True")

print("\nABLATION EXECUTION STACK OK")