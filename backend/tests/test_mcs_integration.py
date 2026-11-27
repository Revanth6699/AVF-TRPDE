from __future__ import annotations

import numpy as np

from backend.app.research.evaluation.model_confidence_set import (
    build_loss_matrix,
    model_confidence_set,
)


def main() -> None:
    rng = np.random.default_rng(42)

    observations = 120

    actual = np.abs(
        rng.normal(
            loc=0.015,
            scale=0.003,
            size=observations,
        )
    )

    actual = np.maximum(actual, 1e-6)

    xgboost = (
        actual
        + rng.normal(
            loc=0.0,
            scale=0.0020,
            size=observations,
        )
    )

    regime_xgboost = (
        actual
        + rng.normal(
            loc=0.0,
            scale=0.0018,
            size=observations,
        )
    )

    gjr_garch = (
        actual
        + rng.normal(
            loc=0.0,
            scale=0.0022,
            size=observations,
        )
    )

    xgboost = np.maximum(xgboost, 1e-6)
    regime_xgboost = np.maximum(
        regime_xgboost,
        1e-6,
    )
    gjr_garch = np.maximum(
        gjr_garch,
        1e-6,
    )

    forecasts = {
        "XGBoost": xgboost,
        "Regime-XGBoost": regime_xgboost,
        "GJR-GARCH": gjr_garch,
    }

    loss_matrix = build_loss_matrix(
        actual,
        forecasts,
        loss="qlike",
    )

    result = model_confidence_set(
        loss_matrix,
        confidence_level=0.90,
        n_bootstrap=500,
        block_length=5,
        random_state=42,
    )

    result_dict = result.as_dict()

    print(
        "\n========== "
        "MCS INTEGRATION "
        "=========="
    )

    print(
        "LOSS MATRIX SHAPE:",
        loss_matrix.shape,
    )

    print(
        "LOSS MATRIX FINITE:",
        np.isfinite(
            loss_matrix.to_numpy(
                dtype=float
            )
        ).all(),
    )

    print(
        "MODELS:",
        tuple(loss_matrix.columns),
    )

    print(
        "OBSERVATIONS:",
        result.observations,
    )

    print(
        "CONFIDENCE LEVEL:",
        result.confidence_level,
    )

    print(
        "BOOTSTRAP SAMPLES:",
        result.bootstrap_samples,
    )

    print(
        "STATISTIC:",
        result.statistic,
    )

    print(
        "P-VALUE:",
        result.p_value,
    )

    print(
        "INCLUDED MODELS:",
        result.included_models,
    )

    print(
        "ELIMINATED MODELS:",
        result.eliminated_models,
    )

    print(
        "RESULT DICT:",
        result_dict,
    )

    assert loss_matrix.shape == (
        observations,
        3,
    )

    assert (
        result.observations
        == observations
    )

    assert (
        result.bootstrap_samples
        == 500
    )

    assert (
        len(result.included_models)
        >= 1
    )

    assert (
        set(result.included_models)
        | set(result.eliminated_models)
    ) == set(forecasts)

    assert (
        set(result.included_models)
        & set(result.eliminated_models)
    ) == set()

    assert np.isfinite(
        result.statistic
    )

    assert 0.0 <= result.p_value <= 1.0

    assert (
        result_dict["included_models"]
        == result.included_models
    )

    print(
        "\nMCS INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()