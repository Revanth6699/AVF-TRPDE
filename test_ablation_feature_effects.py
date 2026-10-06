from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"]
    )

    result["asset_id"] = (
        result["asset_id"]
        .astype(str)
        .str.upper()
    )

    return result.sort_values(
        ["timestamp", "asset_id", "fold_id"]
    ).reset_index(drop=True)


def prediction_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[float, float, int]:
    merged = left[
        [
            "timestamp",
            "asset_id",
            "prediction",
        ]
    ].merge(
        right[
            [
                "timestamp",
                "asset_id",
                "prediction",
            ]
        ],
        on=["timestamp", "asset_id"],
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )

    difference = (
        merged["prediction_left"].to_numpy(dtype=float)
        - merged["prediction_right"].to_numpy(dtype=float)
    )

    return (
        float(np.max(np.abs(difference))),
        float(np.mean(np.abs(difference))),
        int(np.count_nonzero(np.abs(difference) > 1e-12)),
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

    common_kwargs = {
        "target_column": "target",
        "feature_columns": (
            "return",
            "realized_volatility",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ),
    }

    results = {}

    print(
        "\n========== "
        "ABLATION FEATURE EFFECT TEST "
        "=========="
    )

    for ablation in ("A0", "A1", "A2", "A3"):
        print(
            f"\n---------- RUNNING {ablation} ----------"
        )

        result = runner.run_ablation(
            df,
            ablation=ablation,
            hmm_config=HMMConfig(
                n_components=3,
                random_state=42,
            ),
            **common_kwargs,
        )

        predictions = normalize_predictions(
            result.predictions
        )

        results[ablation] = predictions

        print(
            "DESCRIPTION:",
            result.description,
        )

        print(
            "PREDICTIONS:",
            len(predictions),
        )

        feature_columns = [
            column
            for column in predictions.columns
            if (
                column.startswith("regime_probability_")
                or "garch" in column.lower()
                or "volatility" in column.lower()
            )
        ]

        print(
            "RELEVANT OUTPUT COLUMNS:",
            feature_columns,
        )

        if feature_columns:
            print(
                "\nRELEVANT COLUMN SUMMARY:"
            )

            for column in feature_columns:
                values = pd.to_numeric(
                    predictions[column],
                    errors="coerce",
                )

                print(
                    f"{column}: "
                    f"finite={np.isfinite(values).all()} "
                    f"unique={values.nunique()} "
                    f"min={values.min()} "
                    f"max={values.max()}"
                )

    print(
        "\n========== "
        "ABLATION PREDICTION DIFFERENCES "
        "=========="
    )

    comparisons = (
        ("A0", "A1"),
        ("A0", "A2"),
        ("A0", "A3"),
        ("A1", "A3"),
        ("A2", "A3"),
    )

    for left_name, right_name in comparisons:
        max_difference, mean_difference, changed = (
            prediction_difference(
                results[left_name],
                results[right_name],
            )
        )

        print(
            f"{left_name} vs {right_name}"
        )

        print(
            "MAX ABS PREDICTION DIFFERENCE:",
            max_difference,
        )

        print(
            "MEAN ABS PREDICTION DIFFERENCE:",
            mean_difference,
        )

        print(
            "PREDICTIONS CHANGED:",
            changed,
        )

    print(
        "\n========== "
        "ABLATION OUTPUT VALIDATION "
        "=========="
    )

    for name, frame in results.items():
        prediction_values = frame[
            "prediction"
        ].to_numpy(dtype=float)

        print(
            f"{name} PREDICTIONS FINITE:",
            np.isfinite(
                prediction_values
            ).all(),
        )

    print(
        "\nABLATION FEATURE EFFECT TEST COMPLETE"
    )


if __name__ == "__main__":
    main()