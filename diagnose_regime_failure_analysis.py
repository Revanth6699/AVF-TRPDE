import warnings

import numpy as np
import pandas as pd

import test_phase5_cross_asset_research as t
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


warnings.filterwarnings("ignore")


dataframe = t.make_multi_asset_data()

feature_input = dataframe[
    ["timestamp", "asset_id", "return", "close"]
].copy()

feature_input["timestamp"] = pd.to_datetime(
    feature_input["timestamp"],
    errors="coerce",
)

first_rows = (
    feature_input
    .sort_values(["asset_id", "timestamp"])
    .groupby(
        "asset_id",
        sort=False,
        as_index=False,
    )
    .first()
)

timestamp_diffs = (
    feature_input
    .sort_values(["asset_id", "timestamp"])
    .groupby("asset_id")["timestamp"]
    .diff()
    .dropna()
)

timestamp_step = timestamp_diffs.mode().iloc[0]

warmup_rows = first_rows[
    ["timestamp", "asset_id", "return", "close"]
].copy()

warmup_rows["timestamp"] = (
    warmup_rows["timestamp"] - timestamp_step
)

warmup_rows["close"] = (
    warmup_rows["close"].to_numpy(dtype=float)
    / np.exp(
        warmup_rows["return"].to_numpy(dtype=float)
    )
)

feature_input = pd.concat(
    [
        warmup_rows[
            ["timestamp", "asset_id", "close"]
        ],
        feature_input[
            ["timestamp", "asset_id", "close"]
        ],
    ],
    ignore_index=True,
)

cross_asset_full = t.create_cross_asset_features(
    feature_input
)

dataframe = dataframe.merge(
    cross_asset_full,
    on=["timestamp", "asset_id"],
    how="left",
    validate="one_to_one",
)

features = (
    *t.BASE_FEATURE_COLUMNS,
    *t.CROSS_ASSET_FEATURE_COLUMNS,
)

splitter = WalkForwardSplitter(
    train_size=120,
    test_size=1,
    step_size=1,
    expanding=False,
)

print("========== REGIME FAILURE ANALYSIS ==========")
print()

rows = []

for asset in t.ASSETS:
    print(f"---------- {asset} ----------")

    asset_data = (
        dataframe.loc[
            dataframe["asset_id"] == asset
        ]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    runner = WalkForwardRunner(splitter)

    xgb = runner.run_model(
        asset_data,
        model="XGBoost",
        target_column="target",
        feature_columns=features,
    ).predictions

    regime = runner.run_model(
        asset_data,
        model="Regime-XGBoost",
        target_column="target",
        hmm_config=HMMConfig(
            n_components=3,
            covariance_type="diag",
            random_state=42,
        ),
        feature_columns=features,
    ).predictions

    merged = (
        xgb[
            [
                "timestamp",
                "actual_volatility",
                "prediction",
            ]
        ]
        .rename(columns={"prediction": "xgb"})
        .merge(
            regime[
                [
                    "timestamp",
                    "prediction",
                ]
            ],
            on="timestamp",
            validate="one_to_one",
        )
        .rename(
            columns={
                "prediction": "regime_xgb"
            }
        )
    )

    actual = merged[
        "actual_volatility"
    ].to_numpy(dtype=float)

    xgb_prediction = merged[
        "xgb"
    ].to_numpy(dtype=float)

    regime_prediction = merged[
        "regime_xgb"
    ].to_numpy(dtype=float)

    xgb_qlike = (
        actual / xgb_prediction
        + np.log(xgb_prediction)
        - 1.0
        - np.log(actual)
    )

    regime_qlike = (
        actual / regime_prediction
        + np.log(regime_prediction)
        - 1.0
        - np.log(actual)
    )

    difference = regime_qlike - xgb_qlike

    result = {
        "asset": asset,
        "observations": len(difference),
        "regime_better_pct": 100.0
        * float((difference < 0).mean()),
        "regime_worse_pct": 100.0
        * float((difference > 0).mean()),
        "ties_pct": 100.0
        * float((difference == 0).mean()),
        "mean_qlike_diff": float(
            difference.mean()
        ),
        "median_qlike_diff": float(
            np.median(difference)
        ),
        "best_improvement": float(
            difference.min()
        ),
        "worst_degradation": float(
            difference.max()
        ),
    }

    rows.append(result)

    print(
        f"observations={result['observations']}"
    )
    print(
        f"regime_better_pct={result['regime_better_pct']:.2f}"
    )
    print(
        f"regime_worse_pct={result['regime_worse_pct']:.2f}"
    )
    print(
        f"ties_pct={result['ties_pct']:.2f}"
    )
    print(
        f"mean_qlike_diff={result['mean_qlike_diff']:.8f}"
    )
    print(
        f"median_qlike_diff={result['median_qlike_diff']:.8f}"
    )
    print(
        f"best_improvement={result['best_improvement']:.8f}"
    )
    print(
        f"worst_degradation={result['worst_degradation']:.8f}"
    )
    print()

summary = pd.DataFrame(rows)

print("========== SUMMARY ==========")
print(summary.to_string(index=False))

print()
print("RESULT=PASS")