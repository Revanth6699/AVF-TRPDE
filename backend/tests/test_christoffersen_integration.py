from __future__ import annotations

import numpy as np

from backend.app.risk.christoffersen import (
    christoffersen_95,
    christoffersen_99,
)


def main() -> None:
    rng = np.random.default_rng(42)

    print("\n========== PHASE 3 / CHRISTOFFERSEN INTEGRATION ==========")

    observation_count = 250

    actual_returns = rng.normal(
        loc=0.0,
        scale=0.01,
        size=observation_count,
    )

    var95 = np.full(
        observation_count,
        0.016870155647452992,
        dtype=float,
    )

    var99 = np.full(
        observation_count,
        0.025850891679508457,
        dtype=float,
    )

    print("\n--- Input Validation ---")

    print(
        "ACTUAL OBSERVATIONS:",
        len(actual_returns),
    )

    print(
        "VaR95 OBSERVATIONS:",
        len(var95),
    )

    print(
        "VaR99 OBSERVATIONS:",
        len(var99),
    )

    print(
        "ACTUAL FINITE:",
        np.isfinite(actual_returns).all(),
    )

    print(
        "VaR95 FINITE:",
        np.isfinite(var95).all(),
    )

    print(
        "VaR99 FINITE:",
        np.isfinite(var99).all(),
    )

    print("\n--- Christoffersen 95% ---")

    result95 = christoffersen_95(
        actual_returns=actual_returns,
        var_losses=var95,
    )

    print(
        "CONFIDENCE LEVEL:",
        result95.confidence_level,
    )

    print(
        "OBSERVATION COUNT:",
        result95.observation_count,
    )

    print(
        "EXCEPTION COUNT:",
        result95.exception_count,
    )

    print(
        "N00:",
        result95.n00,
    )

    print(
        "N01:",
        result95.n01,
    )

    print(
        "N10:",
        result95.n10,
    )

    print(
        "N11:",
        result95.n11,
    )

    print(
        "TRANSITION PROBABILITY 0:",
        result95.transition_probability_0,
    )

    print(
        "TRANSITION PROBABILITY 1:",
        result95.transition_probability_1,
    )

    print(
        "LR INDEPENDENCE:",
        result95.lr_independence,
    )

    print(
        "INDEPENDENCE P-VALUE:",
        result95.independence_p_value,
    )

    print(
        "LR UNCONDITIONAL COVERAGE:",
        result95.lr_unconditional_coverage,
    )

    print(
        "UNCONDITIONAL COVERAGE P-VALUE:",
        result95.unconditional_coverage_p_value,
    )

    print(
        "LR CONDITIONAL COVERAGE:",
        result95.lr_conditional_coverage,
    )

    print(
        "CONDITIONAL COVERAGE P-VALUE:",
        result95.conditional_coverage_p_value,
    )

    print(
        "REJECT INDEPENDENCE NULL:",
        result95.reject_independence_null,
    )

    print(
        "REJECT CONDITIONAL COVERAGE NULL:",
        result95.reject_conditional_coverage_null,
    )

    print("\n--- Christoffersen 99% ---")

    result99 = christoffersen_99(
        actual_returns=actual_returns,
        var_losses=var99,
    )

    print(
        "CONFIDENCE LEVEL:",
        result99.confidence_level,
    )

    print(
        "OBSERVATION COUNT:",
        result99.observation_count,
    )

    print(
        "EXCEPTION COUNT:",
        result99.exception_count,
    )

    print(
        "N00:",
        result99.n00,
    )

    print(
        "N01:",
        result99.n01,
    )

    print(
        "N10:",
        result99.n10,
    )

    print(
        "N11:",
        result99.n11,
    )

    print(
        "TRANSITION PROBABILITY 0:",
        result99.transition_probability_0,
    )

    print(
        "TRANSITION PROBABILITY 1:",
        result99.transition_probability_1,
    )

    print(
        "LR INDEPENDENCE:",
        result99.lr_independence,
    )

    print(
        "INDEPENDENCE P-VALUE:",
        result99.independence_p_value,
    )

    print(
        "LR UNCONDITIONAL COVERAGE:",
        result99.lr_unconditional_coverage,
    )

    print(
        "UNCONDITIONAL COVERAGE P-VALUE:",
        result99.unconditional_coverage_p_value,
    )

    print(
        "LR CONDITIONAL COVERAGE:",
        result99.lr_conditional_coverage,
    )

    print(
        "CONDITIONAL COVERAGE P-VALUE:",
        result99.conditional_coverage_p_value,
    )

    print(
        "REJECT INDEPENDENCE NULL:",
        result99.reject_independence_null,
    )

    print(
        "REJECT CONDITIONAL COVERAGE NULL:",
        result99.reject_conditional_coverage_null,
    )

    print("\n--- Validation ---")

    valid95 = (
        result95.observation_count
        == observation_count
        and result95.exception_count >= 0
        and (
            result95.n00
            + result95.n01
            + result95.n10
            + result95.n11
        )
        == observation_count - 1
        and np.isfinite(
            result95.lr_independence
        )
        and np.isfinite(
            result95.independence_p_value
        )
        and np.isfinite(
            result95.lr_conditional_coverage
        )
        and np.isfinite(
            result95.conditional_coverage_p_value
        )
    )

    valid99 = (
        result99.observation_count
        == observation_count
        and result99.exception_count >= 0
        and (
            result99.n00
            + result99.n01
            + result99.n10
            + result99.n11
        )
        == observation_count - 1
        and np.isfinite(
            result99.lr_independence
        )
        and np.isfinite(
            result99.independence_p_value
        )
        and np.isfinite(
            result99.lr_conditional_coverage
        )
        and np.isfinite(
            result99.conditional_coverage_p_value
        )
    )

    print(
        "95% RESULT VALID:",
        valid95,
    )

    print(
        "99% RESULT VALID:",
        valid99,
    )

    if not valid95 or not valid99:
        raise AssertionError(
            "Christoffersen integration validation failed."
        )

    print(
        "\nCHRISTOFFERSEN INTEGRATION STACK OK"
    )


if __name__ == "__main__":
    main()