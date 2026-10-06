from __future__ import annotations

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.evaluation.forecast_metrics import (
    calculate_mae,
    calculate_qlike,
    calculate_rmse,
)
from backend.app.research.features.cross_asset import (
    create_cross_asset_features,
)
from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import (
    WalkForwardRunner,
)
from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)


ASSETS = (
    "SYNTH_A",
    "SYNTH_B",
    "SYNTH_C",
)

MODELS = (
    "EWMA",
    "GJR-GARCH",
    "XGBoost",
    "Regime-XGBoost",
)

BASE_FEATURE_COLUMNS = (
    "return",
    "realized_volatility",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

CROSS_ASSET_FEATURE_COLUMNS = (
    "cross_asset_mean_return",
    "cross_asset_return_dispersion",
    "cross_asset_mean_abs_return",
)


def make_multi_asset_data() -> pd.DataFrame:
    """
    Build the deterministic multi-asset research fixture.

    make_controlled_data() contains 238 observations. Because the research
    target is next-observation realized volatility, one additional terminal
    observation is required before applying shift(-1). This gives:

        240 source observations
        -> 238 valid next-step targets
        -> 120 train + 119 one-step test observations

    The terminal row is constructed deterministically from the final source
    row and is used only to provide the next observation required to define
    the final target. Its own target remains NaN and is removed.
    """
    base = make_controlled_data().copy()

    if base.empty:
        raise AssertionError(
            "Controlled base dataset must not be empty."
        )

    required_base_columns = {
        "timestamp",
        "return",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing_base_columns = (
        required_base_columns - set(base.columns)
    )
    if missing_base_columns:
        raise AssertionError(
            "Controlled base dataset is missing required columns: "
            + ", ".join(sorted(missing_base_columns))
        )

    base = (
        base
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # The locked Phase 5 protocol requires 238 valid observations per asset.
    # Since target = next-day realized volatility, construct one terminal
    # source observation before calculating the shifted target.
    if len(base) != 238:
        raise AssertionError(
            "Controlled base dataset must contain 238 observations "
            f"before target construction; got {len(base)}."
        )

    last = base.iloc[-1].copy()

    timestamps = pd.to_datetime(
        base["timestamp"],
        errors="coerce",
    )
    if timestamps.isna().any():
        raise AssertionError(
            "Controlled base dataset contains invalid timestamps."
        )

    if len(timestamps) < 2:
        raise AssertionError(
            "Controlled base dataset requires at least two timestamps."
        )

    inferred_step = timestamps.iloc[-1] - timestamps.iloc[-2]
    if inferred_step <= pd.Timedelta(0):
        raise AssertionError(
            "Controlled base timestamps must be strictly increasing."
        )

    terminal = last.copy()
    terminal["timestamp"] = (
        timestamps.iloc[-1] + inferred_step
    )

    # The terminal observation is deterministic. Its return is carried
    # forward from the final source observation so that the final valid
    # realized-volatility target is well-defined.
    terminal["return"] = float(
        pd.to_numeric(
            last["return"],
            errors="coerce",
        )
    )

    terminal["close"] = float(last["close"]) * np.exp(
        float(terminal["return"])
    )
    terminal["open"] = (
        float(terminal["close"])
        * (1.0 + float(terminal["return"]) * 0.25)
    )
    terminal["high"] = float(terminal["close"]) * 1.01
    terminal["low"] = float(terminal["close"]) * 0.99
    terminal["volume"] = float(last["volume"])

    base = pd.concat(
        [
            base,
            pd.DataFrame([terminal]),
        ],
        ignore_index=True,
    )

    frames: list[pd.DataFrame] = []

    for index, asset_id in enumerate(ASSETS):
        frame = base.copy()

        frame["asset_id"] = asset_id

        if index == 0:
            scale = 1.0
            drift = 0.0
        elif index == 1:
            scale = 1.10
            drift = 0.0002
        else:
            scale = 0.90
            drift = -0.0001

        frame["return"] = (
            pd.to_numeric(
                frame["return"],
                errors="coerce",
            )
            * scale
            + drift
        )

        frame["close"] = (
            100.0
            * np.exp(
                frame["return"].cumsum()
            )
        )

        frame["open"] = (
            frame["close"]
            * (
                1.0
                + frame["return"] * 0.25
            )
        )

        frame["high"] = (
            frame["close"] * 1.01
        )

        frame["low"] = (
            frame["close"] * 0.99
        )

        frame["realized_volatility"] = (
            frame["return"]
            .rolling(5)
            .std()
            .bfill()
        )

        frame["target"] = (
            frame["realized_volatility"]
            .shift(-1)
        )

        frames.append(frame)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.dropna(
        subset=["target"]
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["asset_id"] = (
        result["asset_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if result["timestamp"].isna().any():
        raise AssertionError(
            "Generated dataset contains invalid timestamps."
        )

    if result["asset_id"].eq("").any():
        raise AssertionError(
            "Generated dataset contains empty asset IDs."
        )

    if result["target"].isna().any():
        raise AssertionError(
            "Generated dataset contains missing targets."
        )

    if not np.isfinite(
        result["target"]
        .to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Generated dataset contains non-finite targets."
        )

    if (
        result["target"]
        .to_numpy(dtype=float)
        < 0
    ).any():
        raise AssertionError(
            "Generated dataset contains negative targets."
        )

    counts = (
        result.groupby("asset_id")
        .size()
    )

    if not (
        len(counts) == len(ASSETS)
        and counts.nunique() == 1
        and int(counts.iloc[0]) == 238
    ):
        raise AssertionError(
            "Generated Phase 5 dataset must contain exactly "
            "238 observations per asset; got "
            + counts.to_dict().__str__()
        )

    return (
        result
        .sort_values(
            ["timestamp", "asset_id"]
        )
        .reset_index(drop=True)
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    actual = predictions[
        "actual_volatility"
    ].to_numpy(dtype=float)

    forecast = predictions[
        "prediction"
    ].to_numpy(dtype=float)

    if len(actual) == 0:
        raise AssertionError(
            "Prediction dataset is empty."
        )

    if len(actual) != len(forecast):
        raise AssertionError(
            "Actual and forecast lengths do not match."
        )

    if not np.isfinite(actual).all():
        raise AssertionError(
            "Actual volatility contains non-finite values."
        )

    if not np.isfinite(forecast).all():
        raise AssertionError(
            "Forecast contains non-finite values."
        )

    if (actual < 0).any():
        raise AssertionError(
            "Actual volatility contains negative values."
        )

    if (forecast < 0).any():
        raise AssertionError(
            "Forecast contains negative values."
        )

    return {
        "MAE": float(
            calculate_mae(
                actual,
                forecast,
            )
        ),
        "RMSE": float(
            calculate_rmse(
                actual,
                forecast,
            )
        ),
        "QLIKE": float(
            calculate_qlike(
                actual,
                forecast,
            )
        ),
    }


def validate_multi_asset_dataset(
    dataframe: pd.DataFrame,
) -> None:
    expected_assets = tuple(
        sorted(ASSETS)
    )

    actual_assets = tuple(
        sorted(
            dataframe["asset_id"]
            .unique()
        )
    )

    assert actual_assets == expected_assets, (
        "Generated asset set does not match "
        f"expected assets: {expected_assets}"
    )

    counts = (
        dataframe
        .groupby("asset_id")
        .size()
    )

    assert (
        len(counts) == len(ASSETS)
    )

    assert (
        counts.nunique() == 1
    ), (
        "Assets do not contain equal "
        "numbers of observations."
    )

    assert (
        dataframe["timestamp"]
        .nunique()
        == counts.iloc[0]
    ), (
        "Each asset must have one observation "
        "per timestamp."
    )


def validate_cross_asset_features(
    dataframe: pd.DataFrame,
    cross_asset: pd.DataFrame,
) -> None:
    expected_columns = (
        "timestamp",
        "asset_id",
        *CROSS_ASSET_FEATURE_COLUMNS,
    )

    assert tuple(
        cross_asset.columns
    ) == expected_columns

    expected_rows = len(dataframe)

    assert len(cross_asset) == expected_rows

    assert not (
        cross_asset[
            ["timestamp", "asset_id"]
        ]
        .duplicated()
        .any()
    )

    for column in CROSS_ASSET_FEATURE_COLUMNS:
        values = pd.to_numeric(
            cross_asset[column],
            errors="coerce",
        )

        non_null = values.dropna()

        assert np.isfinite(
            non_null.to_numpy(
                dtype=float
            )
        ).all(), (
            f"{column} contains "
            "non-finite non-null values."
        )

    # Normalize key columns before comparison.
    #
    # create_cross_asset_features() intentionally
    # returns asset_id as pandas StringDtype.
    # The research dataframe may contain object/string
    # dtype. DataFrame.equals() treats these dtype
    # differences as unequal even when the key values
    # are identical.
    merged_keys = (
        dataframe[
            ["timestamp", "asset_id"]
        ]
        .assign(
            timestamp=lambda frame: pd.to_datetime(
                frame["timestamp"],
                errors="coerce",
            ),
            asset_id=lambda frame: (
                frame["asset_id"]
                .astype("string")
                .str.strip()
                .str.upper()
            ),
        )
        .drop_duplicates()
        .sort_values(
            ["timestamp", "asset_id"]
        )
        .reset_index(drop=True)
    )

    feature_keys = (
        cross_asset[
            ["timestamp", "asset_id"]
        ]
        .assign(
            timestamp=lambda frame: pd.to_datetime(
                frame["timestamp"],
                errors="coerce",
            ),
            asset_id=lambda frame: (
                frame["asset_id"]
                .astype("string")
                .str.strip()
                .str.upper()
            ),
        )
        .drop_duplicates()
        .sort_values(
            ["timestamp", "asset_id"]
        )
        .reset_index(drop=True)
    )

    assert merged_keys.equals(
        feature_keys
    ), (
        "Cross-asset feature keys do not "
        "match the research dataset."
	)



def validate_target_asset_exclusion(
    dataframe: pd.DataFrame,
) -> None:
    ordered = (
        dataframe
        .sort_values(
            ["asset_id", "timestamp"]
        )
        .copy()
    )

    ordered["log_return"] = (
        ordered
        .groupby(
            "asset_id",
            sort=False,
        )["close"]
        .transform(
            lambda series: np.log(
                series / series.shift(1)
            )
        )
    )

    valid_timestamps = (
        ordered.loc[
            ordered["log_return"].notna(),
            "timestamp",
        ]
        .drop_duplicates()
        .sort_values()
    )

    if valid_timestamps.empty:
        raise AssertionError(
            "No valid timestamps available "
            "for target-exclusion validation."
        )

    check_timestamp = (
        valid_timestamps.iloc[
            min(10, len(valid_timestamps) - 1)
        ]
    )

    rows = ordered.loc[
        ordered["timestamp"]
        == check_timestamp
    ].copy()

    assert len(rows) == len(ASSETS), (
        "Target-exclusion check requires "
        "all assets at the selected timestamp."
    )

    for _, row in rows.iterrows():
        own_asset = row["asset_id"]

        others = rows.loc[
            rows["asset_id"]
            != own_asset
        ]

        expected_mean = (
            others["log_return"]
            .mean()
        )

        actual_mean = row[
            "cross_asset_mean_return"
        ]

        if not np.isclose(
            actual_mean,
            expected_mean,
            equal_nan=True,
        ):
            raise AssertionError(
                "Cross-asset target exclusion "
                f"failed for asset '{own_asset}'. "
                f"Expected {expected_mean}, "
                f"got {actual_mean}."
            )

        expected_abs_mean = (
            others["log_return"]
            .abs()
            .mean()
        )

        actual_abs_mean = row[
            "cross_asset_mean_abs_return"
        ]

        if not np.isclose(
            actual_abs_mean,
            expected_abs_mean,
            equal_nan=True,
        ):
            raise AssertionError(
                "Cross-asset absolute-return "
                f"feature failed for asset '{own_asset}'. "
                f"Expected {expected_abs_mean}, "
                f"got {actual_abs_mean}."
            )

        expected_dispersion = float(
            np.sqrt(
                max(
                    (
                        others["log_return"] ** 2
                    ).mean()
                    - expected_mean**2,
                    0.0,
                )
            )
        )

        actual_dispersion = float(
            row[
                "cross_asset_return_dispersion"
            ]
        )

        if not np.isclose(
            actual_dispersion,
            expected_dispersion,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise AssertionError(
                "Cross-asset return dispersion "
                f"failed for asset '{own_asset}'. "
                f"Expected {expected_dispersion}, "
                f"got {actual_dispersion}."
            )


def run_asset_models(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    dict[str, dict[str, float]],
]:
    splitter = WalkForwardSplitter(
        train_size=120,
        test_size=1,
        step_size=1,
        expanding=False,
    )

    asset_results: dict[
        str,
        dict[str, dict[str, float]],
    ] = {}

    for asset_id in ASSETS:
        print(
            f"\n---------- "
            f"ASSET: {asset_id} "
            f"----------"
        )

        asset_data = (
            dataframe.loc[
                dataframe["asset_id"]
                == asset_id
            ]
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(asset_data) != 238:
            raise AssertionError(
                f"{asset_id} contains "
                f"{len(asset_data)} observations; "
                "expected 238."
            )

        runner = WalkForwardRunner(
            splitter
        )

        asset_results[
            asset_id
        ] = {}

        for model in MODELS:
            print(
                f"\n{model}"
            )

            if model == "EWMA":
                result = runner.run_model(
                    asset_data,
                    model=model,
                    target_column="target",
                    ewma_decay=0.94,
                )

            elif model == "GJR-GARCH":
                result = runner.run_model(
                    asset_data,
                    model=model,
                    target_column="target",
                )

            elif model == "XGBoost":
                result = runner.run_model(
                    asset_data,
                    model=model,
                    target_column="target",
                    feature_columns=(
                        *BASE_FEATURE_COLUMNS,
                        *CROSS_ASSET_FEATURE_COLUMNS,
                    ),
                )

            else:
                result = runner.run_model(
                    asset_data,
                    model=model,
                    target_column="target",
                    hmm_config=HMMConfig(
                        n_components=3,
                        random_state=42,
                    ),
                    feature_columns=(
                        *BASE_FEATURE_COLUMNS,
                        *CROSS_ASSET_FEATURE_COLUMNS,
                    ),
                )

            predictions = result.predictions

            assert len(predictions) == 119, (
                f"{asset_id} / {model}: "
                f"expected 119 predictions, "
                f"got {len(predictions)}."
            )

            prediction_values = (
                predictions[
                    "prediction"
                ]
                .to_numpy(dtype=float)
            )

            actual_values = (
                predictions[
                    "actual_volatility"
                ]
                .to_numpy(dtype=float)
            )

            assert np.isfinite(
                prediction_values
            ).all(), (
                f"{asset_id} / {model}: "
                "predictions contain "
                "non-finite values."
            )

            assert np.isfinite(
                actual_values
            ).all(), (
                f"{asset_id} / {model}: "
                "actual volatility contains "
                "non-finite values."
            )

            metrics = evaluate_predictions(
                predictions
            )

            asset_results[
                asset_id
            ][model] = metrics

            print(
                "PREDICTIONS:",
                len(predictions),
            )

            print(
                "MAE:",
                metrics["MAE"],
            )

            print(
                "RMSE:",
                metrics["RMSE"],
            )

            print(
                "QLIKE:",
                metrics["QLIKE"],
            )

    return asset_results


def main() -> None:
    print(
        "\n========== "
        "PHASE 5 CROSS-ASSET RESEARCH "
        "=========="
    )

    dataframe = make_multi_asset_data()

    print(
        "\n========== DATASET =========="
    )

    print(
        "ROWS:",
        len(dataframe),
    )

    print(
        "ASSETS:",
        tuple(
            dataframe["asset_id"]
            .unique()
        ),
    )

    print(
        "ASSET COUNTS:"
    )

    print(
        dataframe["asset_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    validate_multi_asset_dataset(
        dataframe
    )

    print(
        "MULTI-ASSET DATASET: PASS"
    )

    print(
        "\n========== "
        "CROSS-ASSET FEATURES "
        "=========="
    )

    cross_asset = (
        create_cross_asset_features(
            dataframe[
                [
                    "timestamp",
                    "asset_id",
                    "close",
                ]
            ]
        )
    )

    print(
        "FEATURE COLUMNS:",
        CROSS_ASSET_FEATURE_COLUMNS,
    )

    print(
        "ROWS:",
        len(cross_asset),
    )

    print(
        "ASSETS:",
        tuple(
            cross_asset["asset_id"]
            .unique()
        ),
    )

    validate_cross_asset_features(
        dataframe,
        cross_asset,
    )

    print(
        "CROSS-ASSET FEATURES: PASS"
    )

    dataframe = dataframe.merge(
        cross_asset,
        on=[
            "timestamp",
            "asset_id",
        ],
        how="left",
        validate="one_to_one",
    )

    for column in CROSS_ASSET_FEATURE_COLUMNS:
        assert column in dataframe.columns

    print(
        "CROSS-ASSET FEATURE MERGE: PASS"
    )

    print(
        "\n========== "
        "TARGET-ASSET EXCLUSION CHECK "
        "=========="
    )

    validate_target_asset_exclusion(
        dataframe
    )

    print(
        "TARGET-ASSET EXCLUSION: PASS"
    )

    print(
        "\n========== "
        "ASSET-BY-ASSET WALK-FORWARD "
        "=========="
    )

    asset_results = run_asset_models(
        dataframe
    )

    print(
        "\n========== "
        "CROSS-ASSET AGGREGATION "
        "=========="
    )

    rows: list[dict[str, object]] = []

    for asset_id in ASSETS:
        for model in MODELS:
            metrics = asset_results[
                asset_id
            ][model]

            rows.append(
                {
                    "asset_id": asset_id,
                    "model": model,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "QLIKE": metrics["QLIKE"],
                }
            )

    results = pd.DataFrame(
        rows
    )

    print(
        results.to_string(
            index=False
        )
    )

    expected_result_count = (
        len(ASSETS)
        * len(MODELS)
    )

    assert len(results) == (
        expected_result_count
    )

    metric_values = results[
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

    aggregated = (
        results
        .groupby("model")[
            [
                "MAE",
                "RMSE",
                "QLIKE",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        "\nAGGREGATED ASSET-LEVEL RESULTS:"
    )

    print(
        aggregated.to_string(
            index=False
        )
    )

    assert len(aggregated) == len(
        MODELS
    )

    print(
        "\n========== "
        "PHASE 5 CROSS-ASSET RESEARCH "
        "TEST: PASS "
        "=========="
    )


if __name__ == "__main__":
    main()