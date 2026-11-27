from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from test_walk_forward_controlled import make_controlled_data

from backend.app.research.regimes.hmm import (
    HMMConfig,
    HMMRegimeDetector,
)


def main() -> None:
    df = make_controlled_data().copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    observations = pd.DataFrame(
        {
            "return": pd.to_numeric(
                df["return"],
                errors="coerce",
            ),
            "realized_volatility": pd.to_numeric(
                df["realized_volatility"],
                errors="coerce",
            ),
            "range": (
                (
                    pd.to_numeric(
                        df["high"],
                        errors="coerce",
                    )
                    -
                    pd.to_numeric(
                        df["low"],
                        errors="coerce",
                    )
                )
                /
                pd.to_numeric(
                    df["close"],
                    errors="coerce",
                ).abs()
            ),
        }
    ).dropna()

    config = HMMConfig(
        n_components=3,
        random_state=42,
    )

    detector = HMMRegimeDetector(config)

    convergence_warnings: list[str] = []

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        detector.fit(observations)

        for warning in captured:
            message = str(warning.message)

            if "not converging" in message.lower():
                convergence_warnings.append(
                    message
                )

    probabilities = detector.filter_probabilities(
        observations
    )

    regime_states = detector.predict_regimes(
        probabilities
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

    print(
        "\n========== "
        "PHASE 1 / HMM CONVERGENCE ASSESSMENT "
        "=========="
    )

    print(
        "OBSERVATION COUNT:",
        len(observations),
    )

    print(
        "FEATURE COUNT:",
        observations.shape[1],
    )

    print(
        "CONFIG:",
        config,
    )

    print(
        "CONVERGENCE WARNINGS:",
        len(convergence_warnings),
    )

    for index, message in enumerate(
        convergence_warnings,
        start=1,
    ):
        print(
            f"WARNING {index}:",
            message,
        )

    print(
        "MODEL IS FITTED:",
        detector.is_fitted,
    )

    print(
        "MODEL CONVERGED:",
        monitor.converged,
    )

    print(
        "MODEL ITERATIONS:",
        getattr(
            monitor,
            "iter",
            "NOT_EXPOSED",
        ),
    )

    print(
        "MODEL HISTORY LENGTH:",
        len(history),
    )

    if history:
        print(
            "INITIAL LOG-LIKELIHOOD:",
            float(history[0]),
        )

        print(
            "FINAL LOG-LIKELIHOOD:",
            float(history[-1]),
        )

        if len(history) >= 2:
            print(
                "FINAL LOG-LIKELIHOOD DELTA:",
                float(
                    history[-1]
                    - history[-2]
                ),
            )

    print(
        "PROBABILITY SHAPE:",
        probabilities.shape,
    )

    print(
        "PROBABILITIES FINITE:",
        np.isfinite(
            probabilities
        ).all(),
    )

    row_sums = probabilities.sum(
        axis=1
    )

    print(
        "PROBABILITY ROW SUMS:",
        np.round(
            row_sums[:10],
            8,
        ),
    )

    print(
        "PROBABILITIES SUM TO ONE:",
        np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ),
    )

    print(
        "REGIME STATE COUNT:",
        len(regime_states),
    )

    print(
        "REGIME STATES:",
        np.unique(
            regime_states
        ),
    )

    print(
        "TRANSITION COUNT:",
        int(
            np.sum(
                regime_states[1:]
                != regime_states[:-1]
            )
        ),
    )

    print(
        "REGIME STATES FINITE:",
        np.isfinite(
            regime_states
        ).all(),
    )

    print(
        "\nHMM CONVERGENCE ASSESSMENT COMPLETE"
    )


if __name__ == "__main__":
    main()