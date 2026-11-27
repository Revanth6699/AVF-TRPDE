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

    config = HMMConfig(
        n_components=3,
        random_state=42,
    )

    iteration_counts: list[int] = []
    final_deltas: list[float] = []
    converged_by_tolerance: list[bool] = []

    valid_probability_folds = 0
    warning_folds = 0

    print(
        "\n========== "
        "PHASE 1 / HMM FOLD CONVERGENCE "
        "=========="
    )

    print(
        "TOTAL FOLDS:",
        len(splits),
    )

    for fold_number, (
        train_indices,
        test_indices,
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

        detector = HMMRegimeDetector(
            config
        )

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

        model = detector.model
        monitor = model.monitor_

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

        probabilities_valid = (
            np.isfinite(
                probabilities
            ).all()
            and
            np.allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-6,
            )
        )

        if probabilities_valid:
            valid_probability_folds += 1

        if has_warning:
            warning_folds += 1

        iteration_counts.append(
            iterations
        )

        final_deltas.append(
            final_delta
        )

        converged_by_tolerance.append(
            bool(reached_tolerance)
        )

        print(
            f"FOLD {fold_number:03d}: "
            f"iterations={iterations}, "
            f"final_delta={final_delta:.8f}, "
            f"tol_reached={reached_tolerance}, "
            f"monitor_converged={monitor.converged}, "
            f"probabilities_valid={probabilities_valid}, "
            f"warning={has_warning}"
        )

    print(
        "\n========== "
        "HMM FOLD SUMMARY "
        "=========="
    )

    print(
        "FOLDS:",
        len(splits),
    )

    print(
        "FOLDS REACHING TOLERANCE:",
        sum(
            converged_by_tolerance
        ),
    )

    print(
        "FOLDS HITTING MAX ITERATIONS:",
        sum(
            count >= config.n_iter
            for count in iteration_counts
        ),
    )

    print(
        "FOLDS WITH CONVERGENCE WARNING:",
        warning_folds,
    )

    print(
        "FOLDS WITH VALID PROBABILITIES:",
        valid_probability_folds,
    )

    print(
        "MIN ITERATIONS:",
        min(iteration_counts),
    )

    print(
        "MAX ITERATIONS:",
        max(iteration_counts),
    )

    print(
        "MEAN ITERATIONS:",
        float(
            np.mean(
                iteration_counts
            )
        ),
    )

    finite_deltas = np.asarray(
        final_deltas,
        dtype=float,
    )

    finite_deltas = finite_deltas[
        np.isfinite(
            finite_deltas
        )
    ]

    if len(finite_deltas):
        print(
            "MIN FINAL DELTA:",
            float(
                np.min(
                    finite_deltas
                )
            ),
        )

        print(
            "MAX FINAL DELTA:",
            float(
                np.max(
                    finite_deltas
                )
            ),
        )

        print(
            "MEAN FINAL DELTA:",
            float(
                np.mean(
                    finite_deltas
                )
            ),
        )

    print(
        "\nHMM FOLD CONVERGENCE "
        "ASSESSMENT COMPLETE"
    )


if __name__ == "__main__":
    main()