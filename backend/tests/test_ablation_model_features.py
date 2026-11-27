from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.models.xgboost import XGBoostVolatility
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


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

    base_features = (
        "return",
        "realized_volatility",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    original_fit = XGBoostVolatility.fit

    captured: dict[str, dict] = {}

    current_ablation = {"name": None}

    def traced_fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_columns=None,
    ):
        name = current_ablation["name"]

        result = original_fit(
            self,
            features,
            target,
            feature_columns=feature_columns,
        )

        if name is not None and name not in captured:
            columns = tuple(
                self.feature_columns
            )

            importance = (
                self.feature_importance()
            )

            captured[name] = {
                "feature_columns": columns,
                "feature_matrix": features.loc[
                    :,
                    list(columns),
                ].copy(),
                "importance": importance.copy(),
            }

        return result

    print(
        "\n========== "
        "ABLATION MODEL FEATURE INSPECTION "
        "=========="
    )

    for ablation in (
        "A0",
        "A1",
        "A2",
        "A3",
    ):
        current_ablation["name"] = ablation

        print(
            f"\n---------- RUNNING {ablation} ----------"
        )

        result = None

        with patch.object(
            XGBoostVolatility,
            "fit",
            new=traced_fit,
        ):
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

        print(
            "DESCRIPTION:",
            result.description,
        )

        print(
            "PREDICTIONS:",
            len(result.predictions),
        )

    print(
        "\n========== "
        "ACTUAL XGBOOST FEATURE COLUMNS "
        "=========="
    )

    for ablation in (
        "A0",
        "A1",
        "A2",
        "A3",
    ):
        info = captured[ablation]

        print(
            f"\n{ablation}:"
        )

        print(
            "FEATURE COLUMNS:"
        )

        for column in info[
            "feature_columns"
        ]:
            print(
                "  -",
                column,
            )

    print(
        "\n========== "
        "FEATURE MATRIX EFFECT "
        "=========="
    )

    for ablation in (
        "A0",
        "A1",
        "A2",
        "A3",
    ):
        info = captured[ablation]
        matrix = info["feature_matrix"]

        print(
            f"\n{ablation}:"
        )

        for column in info[
            "feature_columns"
        ]:
            values = pd.to_numeric(
                matrix[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            print(
                f"{column}: "
                f"finite={np.isfinite(values).all()} "
                f"unique={len(np.unique(values))} "
                f"std={np.std(values):.12g}"
            )

    print(
        "\n========== "
        "XGBOOST FEATURE IMPORTANCE "
        "=========="
    )

    for ablation in (
        "A0",
        "A1",
        "A2",
        "A3",
    ):
        print(
            f"\n{ablation}:"
        )

        print(
            captured[ablation][
                "importance"
            ].to_string(
                index=False
            )
        )

    print(
        "\n========== "
        "FEATURE SET COMPARISON "
        "=========="
    )

    feature_sets = {
        name: captured[name][
            "feature_columns"
        ]
        for name in (
            "A0",
            "A1",
            "A2",
            "A3",
        )
    }

    for name, columns in feature_sets.items():
        print(
            f"{name}: {columns}"
        )

    print(
        "\nABLATION MODEL FEATURE "
        "INSPECTION COMPLETE"
    )


if __name__ == "__main__":
    main()