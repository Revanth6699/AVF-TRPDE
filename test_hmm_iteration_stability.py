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


PROBLEMATIC_FOLDS = (
    21,
    40,
    61,
    117,
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


def fit_hmm(
    observations: pd.DataFrame,
    *,
    n_iter: int,
) -> tuple[
    int,
    float,
    bool,
    bool,
    np.ndarray,
]:
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
        detector.fit(
            observations
        )

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
        "PHASE 1 / HMM ITERATION STABILITY "
        "=========="
    )

    print(
        "FOLDS TESTED:",
        PROBLEMATIC_FOLDS,
    )

    print(
        "BASELINE ITERATIONS:",
        200,
    )

    print(
        "COMPARISON ITERATIONS:",
        300,
    )

    print(
        "REFERENCE ITERATIONS:",
        1000,
    )

    for fold_number in PROBLEMATIC_FOLDS:
        train_indices, _test_indices = (
            splits[fold_number - 1]
        )

        train_data = df.loc[
            train_indices
        ].copy()

        observations = build_observations(
            train_data
        )

        baseline = fit_hmm(
            observations,
            n_iter=200,
        )

        comparison = fit_hmm(
            observations,
            n_iter=300,
        )

        reference = fit_hmm(
            observations,
            n_iter=1000,
        )

        (
            baseline_iterations,
            baseline_delta,
            baseline_tol,
            baseline_warning,
            baseline_probabilities,
        ) = baseline

        (
            comparison_iterations,
            comparison_delta,
            comparison_tol,
            comparison_warning,
            comparison_probabilities,
        ) = comparison

        (
            reference_iterations,
            reference_delta,
            reference_tol,
            reference_warning,
            reference_probabilities,
        ) = reference

        change_200_to_300 = float(
            np.max(
                np.abs(
                    baseline_probabilities
                    -
                    comparison_probabilities
                )
            )
        )

        change_300_to_1000 = float(
            np.max(
                np.abs(
                    comparison_probabilities
                    -
                    reference_probabilities
                )
            )
        )

        change_200_to_1000 = float(
            np.max(
                np.abs(
                    baseline_probabilities
                    -
                    reference_probabilities
                )
            )
        )

        print(
            "\n"
            f"FOLD {fold_number:03d}"
        )

        print(
            "  200 ITERATIONS:",
            f"iterations={baseline_iterations},",
            f"delta={baseline_delta:.10f},",
            f"tol={baseline_tol},",
            f"warning={baseline_warning}",
        )

        print(
            "  300 ITERATIONS:",
            f"iterations={comparison_iterations},",
            f"delta={comparison_delta:.10f},",
            f"tol={comparison_tol},",
            f"warning={comparison_warning}",
        )

        print(
            "  1000 ITERATIONS:",
            f"iterations={reference_iterations},",
            f"delta={reference_delta:.10f},",
            f"tol={reference_tol},",
            f"warning={reference_warning}",
        )

        print(
            "  MAX PROBABILITY CHANGE "
            "200 -> 300:",
            f"{change_200_to_300:.10f}",
        )

        print(
            "  MAX PROBABILITY CHANGE "
            "300 -> 1000:",
            f"{change_300_to_1000:.10f}",
        )

        print(
            "  MAX PROBABILITY CHANGE "
            "200 -> 1000:",
            f"{change_200_to_1000:.10f}",
        )

    print(
        "\n========== "
        "HMM ITERATION STABILITY COMPLETE "
        "=========="
    )


if __name__ == "__main__":
    main()