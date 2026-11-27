from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import (
    HMMConfig,
    HMMRegimeDetector,
)

from backend.app.research.walk_forward.splitter import (
    WalkForwardSplitter,
)


def build_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return": pd.to_numeric(
                dataframe["return"],
                errors="coerce",
            ),
            "realized_volatility": pd.to_numeric(
                dataframe["realized_volatility"],
                errors="coerce",
            ),
            "range": (
                (
                    pd.to_numeric(
                        dataframe["high"],
                        errors="coerce",
                    )
                    -
                    pd.to_numeric(
                        dataframe["low"],
                        errors="coerce",
                    )
                )
                /
                pd.to_numeric(
                    dataframe["close"],
                    errors="coerce",
                ).abs()
            ),
        }
    ).dropna()


def assess(
    observations: pd.DataFrame,
    *,
    n_iter: int,
) -> tuple[int, float, bool, bool, np.ndarray]:
    config = HMMConfig(
        n_components=3,
        covariance_type="full",
        n_iter=n_iter,
        tol=0.0001,
        random_state=42,
    )

    detector = HMMRegimeDetector(config)

    stderr_capture = io.StringIO()
    stdout_capture = io.StringIO()

    with redirect_stderr(
        stderr_capture
    ), redirect_stdout(
        stdout_capture
    ):
        detector.fit(observations)

    monitor = detector.model.monitor_

    history = list(
        getattr(
            monitor,
            "history",
            (),
        )
    )

    iterations = int(
        getattr(
            monitor,
            "iter",
            0,
        )
    )

    if len(history) >= 2:
        final_delta = float(
            abs(
                history[-1]
                - history[-2]
            )
        )
    else:
        final_delta = np.nan

    reached_tolerance = (
        np.isfinite(final_delta)
        and final_delta < config.tol
    )

    warning_text = (
        stderr_capture.getvalue()
        + stdout_capture.getvalue()
    )

    has_warning = (
        "not converging"
        in warning_text.lower()
    )

    probabilities = (
        detector.filter_probabilities(
            observations
        )
    )

    return (
        iterations,
        final_delta,
        bool(reached_tolerance),
        bool(has_warning),
        probabilities,
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

    splitter = WalkForwardSplitter(
        train_size=120,
        test_size=1,
        step_size=1,
        expanding=False,
    )

    splits = splitter.split_indices(
        df
    )

    print(
        "\n========== "
        "PHASE 1 / EXTENDED HMM CONVERGENCE "
        "=========="
    )

    print(
        "TOTAL FOLDS:",
        len(splits),
    )

    baseline_max_iter_folds: list[int] = []

    baseline_results: dict[
        int,
        tuple[
            int,
            float,
            bool,
            bool,
            np.ndarray,
        ],
    ] = {}

    for fold_number, (
        train_indices,
        _test_indices,
    ) in enumerate(
        splits,
        start=1,
    ):
        train_data = df.loc[
            train_indices
        ].copy()

        observations = build_observations(
            train_data
        )

        result = assess(
            observations,
            n_iter=200,
        )

        baseline_results[
            fold_number
        ] = result

        iterations = result[0]

        if iterations >= 200:
            baseline_max_iter_folds.append(
                fold_number
            )

    print(
        "BASELINE MAX-ITER FOLDS:",
        baseline_max_iter_folds,
    )

    print(
        "BASELINE MAX-ITER COUNT:",
        len(
            baseline_max_iter_folds
        ),
    )

    extended_converged = 0
    extended_max_iter = 0
    extended_warnings = 0
    probability_changes: list[float] = []

    print(
        "\nEXTENDED-FOLD RESULTS:"
    )

    for fold_number in baseline_max_iter_folds:
        train_indices, _ = splits[
            fold_number - 1
        ]

        train_data = df.loc[
            train_indices
        ].copy()

        observations = build_observations(
            train_data
        )

        baseline = baseline_results[
            fold_number
        ]

        extended = assess(
            observations,
            n_iter=1000,
        )

        (
            iterations,
            final_delta,
            reached_tolerance,
            warning,
            probabilities,
        ) = extended

        if reached_tolerance:
            extended_converged += 1

        if iterations >= 1000:
            extended_max_iter += 1

        if warning:
            extended_warnings += 1

        baseline_probabilities = baseline[4]

        if (
            probabilities.shape
            == baseline_probabilities.shape
        ):
            probability_change = float(
                np.max(
                    np.abs(
                        probabilities
                        -
                        baseline_probabilities
                    )
                )
            )
        else:
            probability_change = np.nan

        probability_changes.append(
            probability_change
        )

        print(
            f"FOLD {fold_number:03d}: "
            f"iterations={iterations}, "
            f"final_delta={final_delta:.10f}, "
            f"tol_reached={reached_tolerance}, "
            f"warning={warning}, "
            f"max_probability_change="
            f"{probability_change:.10f}"
        )

    print(
        "\n========== "
        "EXTENDED HMM SUMMARY "
        "=========="
    )

    print(
        "BASELINE MAX-ITER COUNT:",
        len(
            baseline_max_iter_folds
        ),
    )

    print(
        "EXTENDED FOLDS REACHING TOLERANCE:",
        extended_converged,
    )

    print(
        "EXTENDED FOLDS HITTING 1000 ITERATIONS:",
        extended_max_iter,
    )

    print(
        "EXTENDED CONVERGENCE WARNINGS:",
        extended_warnings,
    )

    if probability_changes:
        finite_changes = np.asarray(
            probability_changes,
            dtype=float,
        )

        finite_changes = finite_changes[
            np.isfinite(
                finite_changes
            )
        ]

        print(
            "MAX PROBABILITY CHANGE:",
            float(
                np.max(
                    finite_changes
                )
            ),
        )

        print(
            "MEAN PROBABILITY CHANGE:",
            float(
                np.mean(
                    finite_changes
                )
            ),
        )

    print(
        "\nEXTENDED HMM CONVERGENCE "
        "ASSESSMENT COMPLETE"
    )


if __name__ == "__main__":
    main()