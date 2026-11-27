from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


def main() -> None:
    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

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

    base_features = (
        "return",
        "realized_volatility",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    results = {}

    print(
        "\n========== "
        "ABLATION OOS PREDICTION TRACE "
        "=========="
    )

    for ablation in (
        "A0",
        "A1",
        "A2",
        "A3",
    ):
        print(
            f"\n---------- RUNNING {ablation} ----------"
        )

        result = runner.run_ablation(
            df,
            ablation=ablation,
            target_column="target",
            hmm_config=HMMConfig(
                n_components=3,
                random_state=42,
            ),
            feature_columns=base_features,
        )

        predictions = (
            result.predictions[
                [
                    "timestamp",
                    "asset_id",
                    "prediction",
                ]
            ]
            .sort_values(
                ["timestamp", "asset_id"]
            )
            .reset_index(drop=True)
        )

        results[ablation] = predictions

        print(
            "FOLDS:",
            result.fold_count,
        )

        print(
            "PREDICTIONS:",
            result.prediction_count,
        )

        values = predictions[
            "prediction"
        ].to_numpy(dtype=float)

        print(
            "FINITE:",
            np.isfinite(values).all(),
        )

        print(
            "UNIQUE PREDICTIONS:",
            len(np.unique(values)),
        )

        print(
            "MIN:",
            values.min(),
        )

        print(
            "MAX:",
            values.max(),
        )

    print(
        "\n========== "
        "PAIRWISE OOS DIFFERENCES "
        "=========="
    )

    comparisons = (
        ("A0", "A1"),
        ("A0", "A2"),
        ("A0", "A3"),
        ("A1", "A2"),
        ("A1", "A3"),
        ("A2", "A3"),
    )

    for left, right in comparisons:
        left_df = results[left]
        right_df = results[right]

        if not left_df[
            ["timestamp", "asset_id"]
        ].equals(
            right_df[
                ["timestamp", "asset_id"]
            ]
        ):
            print(
                f"{left} vs {right}: "
                "TIMESTAMP/ASSET ALIGNMENT FAILED"
            )
            continue

        left_values = left_df[
            "prediction"
        ].to_numpy(dtype=float)

        right_values = right_df[
            "prediction"
        ].to_numpy(dtype=float)

        difference = (
            left_values - right_values
        )

        print(
            f"\n{left} vs {right}"
        )

        print(
            "MAX ABS DIFF:",
            np.max(np.abs(difference)),
        )

        print(
            "MEAN ABS DIFF:",
            np.mean(np.abs(difference)),
        )

        print(
            "NONZERO DIFF COUNT:",
            np.count_nonzero(difference),
        )

        print(
            "ALL EXACTLY EQUAL:",
            np.array_equal(
                left_values,
                right_values,
            ),
        )

    print(
        "\n========== "
        "FIRST 20 OOS PREDICTIONS "
        "=========="
    )

    comparison = results["A0"].copy()

    comparison = comparison.rename(
        columns={
            "prediction": "A0"
        }
    )

    for ablation in (
        "A1",
        "A2",
        "A3",
    ):
        comparison[ablation] = (
            results[ablation][
                "prediction"
            ].to_numpy(dtype=float)
        )

    print(
        comparison.head(20).to_string(
            index=False
        )
    )

    print(
        "\n========== "
        "OOS PREDICTION TRACE COMPLETE "
        "=========="
    )


if __name__ == "__main__":
    main()