import numpy as np
import pandas as pd

from test_phase5_cross_asset_research import make_multi_asset_data
from backend.app.research.regimes.hmm import HMMRegimeDetector, HMMConfig
from backend.app.research.walk_forward.splitter import WalkForwardSplitter


df = make_multi_asset_data()

df = (
    df.loc[df["asset_id"] == "SYNTH_A"]
    .copy()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

df["range"] = (
    (
        pd.to_numeric(df["high"], errors="coerce")
        - pd.to_numeric(df["low"], errors="coerce")
    )
    / pd.to_numeric(df["close"], errors="coerce").abs()
)

columns = [
    "return",
    "realized_volatility",
    "range",
]

splitter = WalkForwardSplitter(
    train_size=120,
    test_size=1,
    step_size=1,
    expanding=False,
)

folds = list(splitter.split(df))

config = HMMConfig(
    n_components=3,
    random_state=42,
)

print("========== HMM FOLD DIAGNOSTIC ==========")
print("asset=SYNTH_A")
print("folds=", len(folds))
print("------------------------------------------")

for fold in folds:
    train_mask = (
        (df["timestamp"] >= fold.train_start)
        & (df["timestamp"] <= fold.train_end)
    )

    test_mask = (
        (df["timestamp"] >= fold.test_start)
        & (df["timestamp"] <= fold.test_end)
    )

    train = (
        df.loc[train_mask, columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    test = (
        df.loc[test_mask, columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    hmm = HMMRegimeDetector(config=config)

    try:
        hmm.fit(train)

        train_probabilities = hmm.filter_probabilities(
            train
        )

        test_probabilities = hmm.filter_after_training(
            test
        )

        if not np.isfinite(train_probabilities).all():
            raise RuntimeError(
                "Training probabilities contain non-finite values."
            )

        if not np.isfinite(test_probabilities).all():
            raise RuntimeError(
                "Test probabilities contain non-finite values."
            )

    except Exception as exc:
        print()
        print("========== FIRST FAILURE ==========")
        print("fold_id=", fold.fold_id)
        print("train_rows=", len(train))
        print("test_rows=", len(test))
        print("train_start=", fold.train_start)
        print("train_end=", fold.train_end)
        print("test_start=", fold.test_start)
        print("test_end=", fold.test_end)
        print("error_type=", type(exc).__name__)
        print("error=", str(exc))
        print()

        if hmm.is_fitted:
            print("========== FITTED HMM STATE ==========")
            print("startprob=", hmm.model.startprob_)
            print(
                "startprob_sum=",
                float(np.sum(hmm.model.startprob_)),
            )
            print("transmat=")
            print(hmm.model.transmat_)
            print(
                "transmat_row_sums=",
                hmm.model.transmat_.sum(axis=1),
            )
            print("means=")
            print(hmm.model.means_)
            print("covars=")
            print(hmm.model.covars_)
            print(
                "covars_finite=",
                np.isfinite(hmm.model.covars_).all(),
            )

        print()
        print("========== TRAINING DATA ==========")
        print(train.describe().to_string())
        print()
        print("========== RESULT ==========")
        print("FIRST_FAILURE_FOLD=", fold.fold_id)
        print("RESULT=FAIL")
        break

else:
    print()
    print("RESULT=ALL_HMM_FOLDS_PASS")