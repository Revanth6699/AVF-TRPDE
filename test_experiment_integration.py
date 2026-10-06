import tempfile
from pathlib import Path

import pandas as pd

from backend.app.experiments.configuration import (
    ExperimentConfiguration,
    ExperimentConfig,
    load_experiment_config,
    parse_experiment_config,
    save_experiment_config,
    validate_experiment_config,
    list_experiment_names,
    ExperimentConfigurationError,
)

from backend.app.experiments.runner import (
    ExperimentRunner,
    run_experiments,
)

from backend.app.experiments.artifacts import (
    save_json_artifact,
    load_json_artifact,
    save_dataframe_artifact,
    load_dataframe_artifact,
    save_experiment_artifacts,
    load_experiment_metadata,
    load_experiment_results,
)


# =========================================================
# 1. Build locked experiment configuration
# =========================================================

raw_config = {
    "experiments": [
        {
            "name": "baseline",
            "description": "Baseline volatility model comparison.",
            "models": [
                "EWMA",
                "GJR-GARCH",
                "XGBoost",
            ],
            "metrics": [
                "MAE",
                "RMSE",
                "QLIKE",
            ],
            "risk_measures": [
                "VaR95",
                "VaR99",
                "ES95",
            ],
            "parameters": {
                "train_size": 120,
                "test_size": 1,
                "step_size": 1,
            },
            "metadata": {
                "research_stage": "baseline",
            },
        },
        {
            "name": "regime_xgboost",
            "description": "Regime-aware volatility experiment.",
            "models": [
                "XGBoost",
                "HMM",
                "Regime-XGBoost",
            ],
            "metrics": [
                "MAE",
                "RMSE",
                "QLIKE",
            ],
            "risk_measures": [
                "VaR95",
                "VaR99",
                "ES95",
            ],
            "parameters": {
                "train_size": 120,
                "test_size": 1,
                "step_size": 1,
            },
            "metadata": {
                "research_stage": "regime",
            },
        },
    ]
}


configuration = parse_experiment_config(
    raw_config
)

validate_experiment_config(
    configuration
)

print("========== CONFIGURATION ==========")
print(
    "EXPERIMENT COUNT:",
    configuration.experiment_count,
)

print(
    "EXPERIMENT NAMES:",
    list_experiment_names(configuration)
)

assert configuration.experiment_count == 2
assert list_experiment_names(configuration) == (
    "baseline",
    "regime_xgboost",
)

print("CONFIGURATION VALID: True")


# =========================================================
# 2. YAML save/load round trip
# =========================================================

with tempfile.TemporaryDirectory() as temp_dir:

    temp_path = Path(temp_dir)

    yaml_path = (
        temp_path / "experiments.yaml"
    )

    saved_path = save_experiment_config(
        configuration,
        yaml_path,
    )

    loaded_configuration = (
        load_experiment_config(
            saved_path
        )
    )

    assert (
        loaded_configuration.experiments
        == configuration.experiments
    )

    assert loaded_configuration.source_path == saved_path

    print("\n========== YAML ROUND TRIP ==========")
    print(
        "CONFIG FILE:",
        saved_path,
    )
    print(
        "ROUND TRIP: True"
    )


    # =====================================================
    # 3. Runner with registered handlers
    # =====================================================

    def baseline_handler(
        experiment,
    ):
        return {
            "experiment": experiment.name,
            "models": list(
                experiment.models
            ),
            "status": "completed",
        }


    def regime_handler(
        experiment,
    ):
        return {
            "experiment": experiment.name,
            "models": list(
                experiment.models
            ),
            "status": "completed",
        }


    handlers = {
        "baseline": baseline_handler,
        "regime_xgboost": regime_handler,
    }

    run = run_experiments(
        configuration,
        handlers=handlers,
    )

    print(
        "\n========== EXPERIMENT RUNNER =========="
    )
    print(
        "EXPERIMENT COUNT:",
        run.experiment_count,
    )
    print(
        "SUCCESSFUL:",
        run.successful_count,
    )
    print(
        "FAILED:",
        run.failed_count,
    )
    print(
        "ALL SUCCESSFUL:",
        run.all_successful,
    )

    for result in run.results:
        print(
            result.name,
            result.status,
            result.result,
        )

    assert run.experiment_count == 2
    assert run.successful_count == 2
    assert run.failed_count == 0
    assert run.all_successful is True


    # =====================================================
    # 4. Runner failure isolation
    # =====================================================

    failure_handlers = {
        "baseline": baseline_handler,
    }

    failure_run = run_experiments(
        configuration,
        handlers=failure_handlers,
    )

    print(
        "\n========== HANDLER FAILURE TEST =========="
    )
    print(
        "SUCCESSFUL:",
        failure_run.successful_count,
    )
    print(
        "FAILED:",
        failure_run.failed_count,
    )

    assert failure_run.experiment_count == 2
    assert failure_run.successful_count == 1
    assert failure_run.failed_count == 1

    failed_results = [
        result
        for result in failure_run.results
        if result.status == "failed"
    ]

    assert len(failed_results) == 1
    assert (
        "No handler registered"
        in failed_results[0].error
    )


    # =====================================================
    # 5. JSON artifact round trip
    # =====================================================

    metadata = {
        "experiment": "baseline",
        "models": [
            "EWMA",
            "GJR-GARCH",
            "XGBoost",
        ],
        "metrics": [
            "MAE",
            "RMSE",
            "QLIKE",
        ],
        "successful": True,
    }

    json_path = (
        temp_path / "metadata.json"
    )

    json_artifact = save_json_artifact(
        metadata,
        json_path,
    )

    loaded_metadata = load_json_artifact(
        json_path
    )

    assert loaded_metadata == metadata
    assert json_artifact.size_bytes > 0

    print(
        "\n========== JSON ARTIFACT =========="
    )
    print(
        "TYPE:",
        json_artifact.artifact_type,
    )
    print(
        "SIZE:",
        json_artifact.size_bytes,
    )
    print(
        "ROUND TRIP: True"
    )


    # =====================================================
    # 6. Parquet artifact round trip
    # =====================================================

    results_dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
            "asset_id": [
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
            ],
            "actual_volatility": [
                0.020,
                0.021,
                0.019,
                0.022,
                0.018,
            ],
            "forecast_volatility": [
                0.021,
                0.020,
                0.020,
                0.021,
                0.019,
            ],
        }
    )

    parquet_path = (
        temp_path / "results.parquet"
    )

    parquet_artifact = (
        save_dataframe_artifact(
            results_dataframe,
            parquet_path,
        )
    )

    loaded_results = (
        load_dataframe_artifact(
            parquet_path
        )
    )

    pd.testing.assert_frame_equal(
        loaded_results,
        results_dataframe,
    )

    assert parquet_artifact.size_bytes > 0

    print(
        "\n========== PARQUET ARTIFACT =========="
    )
    print(
        "TYPE:",
        parquet_artifact.artifact_type,
    )
    print(
        "SIZE:",
        parquet_artifact.size_bytes,
    )
    print(
        "ROWS:",
        len(loaded_results),
    )
    print(
        "ROUND TRIP: True"
    )


    # =====================================================
    # 7. Complete experiment artifact package
    # =====================================================

    artifact_directory = (
        temp_path / "experiment_artifacts"
    )

    artifacts = save_experiment_artifacts(
        experiment_name="baseline",
        metadata=metadata,
        results=results_dataframe,
        output_directory=artifact_directory,
    )

    assert "metadata" in artifacts
    assert "results" in artifacts

    loaded_experiment_metadata = (
        load_experiment_metadata(
            "baseline",
            output_directory=artifact_directory,
        )
    )

    loaded_experiment_results = (
        load_experiment_results(
            "baseline",
            output_directory=artifact_directory,
        )
    )

    assert (
        loaded_experiment_metadata
        == metadata
    )

    pd.testing.assert_frame_equal(
        loaded_experiment_results,
        results_dataframe,
    )

    metadata_path = (
        artifact_directory
        / "baseline"
        / "metadata.json"
    )

    results_path = (
        artifact_directory
        / "baseline"
        / "results.parquet"
    )

    assert metadata_path.exists()
    assert results_path.exists()

    print(
        "\n========== COMPLETE ARTIFACT PACKAGE =========="
    )
    print(
        "METADATA EXISTS:",
        metadata_path.exists(),
    )
    print(
        "RESULTS EXISTS:",
        results_path.exists(),
    )
    print(
        "ARTIFACT ROUND TRIP: True"
    )


# =========================================================
# 8. Validation failure checks
# =========================================================

try:
    parse_experiment_config(
        {
            "experiments": [
                {
                    "name": "invalid",
                    "description": "Invalid model test.",
                    "models": [
                        "NOT-A-LOCKED-MODEL"
                    ],
                    "metrics": ["MAE"],
                    "risk_measures": ["VaR95"],
                }
            ]
        }
    )
    raise AssertionError(
        "Unsupported model was not rejected."
    )

except ExperimentConfigurationError:
    print(
        "\nINVALID MODEL REJECTION: True"
    )


try:
    parse_experiment_config(
        {
            "experiments": [
                {
                    "name": "duplicate_test",
                    "description": "Duplicate model test.",
                    "models": [
                        "EWMA",
                        "EWMA",
                    ],
                    "metrics": ["MAE"],
                    "risk_measures": ["VaR95"],
                }
            ]
        }
    )
    raise AssertionError(
        "Duplicate model was not rejected."
    )

except ExperimentConfigurationError:
    print(
        "DUPLICATE MODEL REJECTION: True"
    )


print(
    "\nEXPERIMENT INTEGRATION STACK OK"
)