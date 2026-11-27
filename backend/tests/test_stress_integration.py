import numpy as np
import pandas as pd

from backend.app.stress.scenarios import (
    HistoricalWindow,
    build_stress_scenarios,
    summarize_scenarios,
)

from backend.app.stress.runner import (
    run_stress_scenarios,
)

from backend.app.stress.analysis import (
    analyze_stress_results,
    summarize_stress_analysis,
    compare_stress_scenarios,
    calculate_stress_changes,
    build_stress_analysis_table,
)


# ---------------------------------------------------------
# Controlled historical dataset
# ---------------------------------------------------------

dates = pd.date_range(
    "2026-01-01",
    periods=40,
    freq="D",
)

returns = np.array([
    0.010,
    0.008,
    -0.006,
    0.012,
    -0.004,
    0.009,
    0.007,
    -0.005,
    0.011,
    -0.007,

    0.006,
    -0.008,
    0.005,
    -0.009,
    0.004,
    -0.010,
    0.003,
    -0.012,
    0.002,
    -0.015,

    -0.020,
    -0.025,
    -0.030,
    -0.020,
    -0.018,
    0.010,
    0.015,
    0.012,
    0.008,
    0.006,

    -0.005,
    0.004,
    -0.006,
    0.005,
    -0.007,
    0.006,
    -0.008,
    0.007,
    -0.009,
    0.010,
])

volatility = np.array([
    0.010,
    0.011,
    0.012,
    0.011,
    0.010,
    0.012,
    0.011,
    0.013,
    0.012,
    0.011,

    0.014,
    0.015,
    0.016,
    0.017,
    0.018,
    0.020,
    0.021,
    0.022,
    0.024,
    0.025,

    0.040,
    0.045,
    0.050,
    0.055,
    0.060,
    0.035,
    0.030,
    0.025,
    0.020,
    0.018,

    0.016,
    0.015,
    0.017,
    0.016,
    0.018,
    0.019,
    0.020,
    0.021,
    0.022,
    0.023,
])

regime = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
]

df = pd.DataFrame(
    {
        "timestamp": dates,
        "asset_id": ["AAPL"] * len(dates),
        "return": returns,
        "volatility": volatility,
        "regime": regime,
    }
)


# ---------------------------------------------------------
# Build stress scenarios
# ---------------------------------------------------------

historical_windows = (
    HistoricalWindow(
        "historical_test_window",
        pd.Timestamp("2026-01-15").date(),
        pd.Timestamp("2026-01-25").date(),
    ),
)

scenarios = build_stress_scenarios(
    df,
    normal_lower_percentile=0.0,
    normal_upper_percentile=50.0,
    high_volatility_percentile=90.0,
    volatility_spike_percentile=95.0,
    minimum_drawdown=-0.05,
    minimum_drawdown_duration=3,
    regime_column="regime",
    historical_windows=historical_windows,
)

print("========== STRESS SCENARIO BUILD ==========")
print("SCENARIOS:", list(scenarios.keys()))
print(summarize_scenarios(scenarios))


# ---------------------------------------------------------
# Run stress scenarios
# ---------------------------------------------------------

run = run_stress_scenarios(scenarios)

print("\n========== STRESS RUNNER ==========")
print("SCENARIO COUNT:", run.scenario_count)
print("TOTAL OBSERVATIONS:", run.total_observations)
print(run.results)


# ---------------------------------------------------------
# Analyze stress results
# ---------------------------------------------------------

analysis = analyze_stress_results(
    run.results
)

print("\n========== STRESS ANALYSIS ==========")
print("ANALYSIS SCENARIOS:", analysis.scenario_count)
print("VALID SCENARIOS:", analysis.valid_scenario_count)

print("\nSUMMARY:")
print(
    summarize_stress_analysis(
        analysis
    )
)

print("\nCOMPARISON:")
print(
    compare_stress_scenarios(
        analysis
    )
)


# ---------------------------------------------------------
# Reference scenario comparison
# ---------------------------------------------------------

changes = calculate_stress_changes(
    run.results,
    reference_scenario="normal_volatility",
)

print("\n========== CHANGES VS NORMAL ==========")
print(changes)


# ---------------------------------------------------------
# Analytical table
# ---------------------------------------------------------

table = build_stress_analysis_table(
    run.results
)

print("\n========== STRESS ANALYSIS TABLE ==========")
print(table)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert run.scenario_count == len(scenarios)
assert run.total_observations >= 0

assert analysis.scenario_count == len(run.results)
assert analysis.valid_scenario_count > 0

assert not changes.empty
assert "reference_scenario" in changes.columns
assert set(changes["reference_scenario"]) == {
    "normal_volatility"
}

assert not table.empty
assert "has_observations" in table.columns

numeric_columns = [
    "total_return",
    "mean_return",
    "volatility",
    "downside_deviation",
    "max_drawdown",
    "cumulative_return",
]

for column in numeric_columns:
    observed = run.results.loc[
        run.results["observation_count"] > 0,
        column,
    ]

    assert np.isfinite(
        observed.to_numpy(dtype=float)
    ).all(), column


print("\nSTRESS INTEGRATION STACK OK")