from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.research.regimes.hmm import HMMConfig
from backend.app.research.walk_forward.runner import WalkForwardRunner
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


def make_controlled_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    n = 240
    timestamps = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D",
    )

    returns = rng.normal(
        loc=0.0005,
        scale=0.01,
        size=n,
    )

    close = 100.0 * np.exp(np.cumsum(returns))

    high = close * (1.0 + rng.uniform(0.001, 0.02, n))
    low = close * (1.0 - rng.uniform(0.001, 0.02, n))
    open_price = close * (1.0 + rng.normal(0.0, 0.003, n))

    realized_volatility = (
        pd.Series(returns)
        .rolling(5)
        .std()
        .bfill()
        .to_numpy()
    )

    # Next-period realized-volatility target.
    target = pd.Series(realized_volatility).shift(-1)

    volume = rng.integers(
        1_000_000,
        5_000_000,
        size=n,
    )

    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": "SYNTH",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "return": returns,
            "realized_volatility": realized_volatility,
            "target": target,
        }
    )

    # Remove the final row because its next-day target
    # does not exist.
    return dataframe.iloc[:-1].reset_index(drop=True)


def main() -> None:
    dataframe = make_controlled_data()

    splitter = WalkForwardSplitter(
        train_size=120,
        test_size=1,
        step_size=1,
        expanding=False,
    )

    runner = WalkForwardRunner(splitter)

    result = runner.run_model(
    	dataframe,
    	model="Regime-XGBoost",
    	target_column="target",
    	hmm_config=HMMConfig(
        	n_components=3,
        	random_state=42,
    	),
    	feature_columns=(
        	"return",
        	"realized_volatility",
        	"open",
        	"high",
        	"low",
        	"close",
        	"volume",
    	),
    )

    print("\n========== CONTROLLED WALK-FORWARD TEST ==========")
    print(f"Input rows       : {len(dataframe)}")
    print(f"Result type      : {type(result).__name__}")
    print(f"Prediction rows  : {len(result.predictions)}")
    print(f"Fold count       : {result.fold_count}")

    print("\n--- Predictions ---")
    print(result.predictions.head(10).to_string(index=False))

    print("\n--- Prediction columns ---")
    print(list(result.predictions.columns))

    print("\n--- Basic validation ---")

    predictions = result.predictions

    print(
        "Finite predictions:",
        np.isfinite(
            predictions["prediction"].to_numpy(dtype=float)
        ).all(),
    )

    print(
        "Finite actuals:",
        np.isfinite(
            predictions["actual_volatility"].to_numpy(dtype=float)
        ).all(),
    )

    print(
        "Unique timestamps:",
        predictions["timestamp"].is_unique,
    )

    print(
        "Asset IDs:",
        predictions["asset_id"].unique().tolist(),
    )

    print("\n--- Fold information ---")

    for fold in result.folds[:5]:
        print(
            f"Fold {fold.fold_id}: "
            f"train={fold.train_start} -> {fold.train_end}, "
            f"test={fold.test_start} -> {fold.test_end}"
        )

    print("\n========== TEST COMPLETE ==========")


if __name__ == "__main__":
    main()