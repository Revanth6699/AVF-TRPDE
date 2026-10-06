from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


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

    feature_columns = (
        "return",
        "realized_volatility",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    models = (
        "EWMA",
        "GJR-GARCH",
        "XGBoost",
        "Regime-XGBoost",
    )

    results: dict[str, pd.DataFrame] = {}

    print(
        "\n========== "
        "PHASE 2 / FULL MODEL PREDICTION COMPARISON "
        "=========="
    )

    for model in models:
        print(
            f"\n---------- RUNNING {model} ----------"
        )

        if model == "EWMA":
            result = runner.run_model(
                dataframe,
                model=model,
                target_column="target",
                ewma_decay=0.94,
            )

        elif model == "Regime-XGBoost":
            result = runner.run_model(
                dataframe,
                model=model,
                target_column="target",
                hmm_config=HMMConfig(
                    n_components=3,
                    random_state=42,
                ),
                feature_columns=feature_columns,
            )

        elif model == "XGBoost":
            result = runner.run_model(
                dataframe,
                model=model,
                target_column="target",
                feature_columns=feature_columns,
            )

        else:
            result = runner.run_model(
                dataframe,
                model=model,
                target_column="target",
            )

        predictions = (
            result.predictions[
                [
                    "timestamp",
                    "asset_id",
                    "prediction",
                    "actual_volatility",
                ]
            ]
            .sort_values(
                ["timestamp", "asset_id"]
            )
            .reset_index(drop=True)
        )

        results[model] = predictions

        prediction_values = predictions[
            "prediction"
        ].to_numpy(dtype=float)

        actual_values = predictions[
            "actual_volatility"
        ].to_numpy(dtype=float)

        print(
            "FOLDS:",
            result.fold_count,
        )

        print(
            "PREDICTIONS:",
            result.prediction_count,
        )

        print(
            "PREDICTIONS FINITE:",
            np.isfinite(
                prediction_values
            ).all(),
        )

        print(
            "ACTUALS FINITE:",
            np.isfinite(
                actual_values
            ).all(),
        )

        print(
            "PREDICTIONS POSITIVE:",
            (
                prediction_values > 0
            ).all(),
        )

    print(
        "\n========== "
        "ALIGNMENT CHECK "
        "=========="
    )

    reference = results["EWMA"]

    reference_keys = reference[
        ["timestamp", "asset_id"]
    ]

    reference_actuals = reference[
        "actual_volatility"
    ].to_numpy(dtype=float)

    all_aligned = True

    for model in models:
        current = results[model]

        keys_match = current[
            ["timestamp", "asset_id"]
        ].equals(reference_keys)

        actuals_match = np.array_equal(
            current[
                "actual_volatility"
            ].to_numpy(dtype=float),
            reference_actuals,
        )

        print(
            f"{model} TIMESTAMPS/ASSETS MATCH:",
            keys_match,
        )

        print(
            f"{model} ACTUALS MATCH:",
            actuals_match,
        )

        if not keys_match or not actuals_match:
            all_aligned = False

    print(
        "ALL MODELS ALIGNED:",
        all_aligned,
    )

    if not all_aligned:
        raise RuntimeError(
            "Full model OOS predictions are not aligned."
        )

    print(
        "\n========== "
        "COMBINED OOS PREDICTIONS "
        "=========="
    )

    combined = reference[
        [
            "timestamp",
            "asset_id",
            "actual_volatility",
        ]
    ].copy()

    for model in models:
        combined[model] = results[
            model
        ]["prediction"].to_numpy(
            dtype=float
        )

    print(
        "ROWS:",
        len(combined),
    )

    print(
        "COLUMNS:",
        combined.columns.tolist(),
    )

    print(
        "\nFIRST 20 ROWS:"
    )

    print(
        combined.head(20).to_string(
            index=False
        )
    )

    print(
        "\n========== "
        "MODEL PREDICTION VALIDATION "
        "=========="
    )

    for model in models:
        values = combined[
            model
        ].to_numpy(dtype=float)

        print(
            f"{model}: "
            f"finite={np.isfinite(values).all()} "
            f"unique={len(np.unique(values))} "
            f"min={values.min():.12g} "
            f"max={values.max():.12g}"
        )

    print(
        "\n========== "
        "FULL MODEL PREDICTION COMPARISON COMPLETE "
        "=========="
    )


if __name__ == "__main__":
    main()