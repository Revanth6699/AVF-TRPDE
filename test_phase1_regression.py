from __future__ import annotations

import subprocess
import sys

from backend.app.research.regimes.hmm import HMMConfig


def run_command(
    command: list[str],
    label: str,
) -> None:
    print(
        "\n"
        + "=" * 72
    )

    print(
        f"PHASE 1 REGRESSION / {label}"
    )

    print(
        "=" * 72
    )

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} FAILED "
            f"with exit code "
            f"{result.returncode}."
        )

    print(
        f"\n{label}: PASS"
    )


def main() -> None:
    print(
        "\n"
        "========== "
        "AVF-TRPDE PHASE 1 FINAL REGRESSION "
        "=========="
    )

    # ------------------------------------------------------------------
    # 1. Verify the production HMM configuration.
    # ------------------------------------------------------------------

    hmm_config = HMMConfig(
        n_components=3,
    )

    print(
        "\nHMM DEFAULT CONFIG:"
    )

    print(
        "  n_components:",
        hmm_config.n_components,
    )

    print(
        "  covariance_type:",
        hmm_config.covariance_type,
    )

    print(
        "  n_iter:",
        hmm_config.n_iter,
    )

    print(
        "  tol:",
        hmm_config.tol,
    )

    print(
        "  random_state:",
        hmm_config.random_state,
    )

    if hmm_config.n_iter != 300:
        raise AssertionError(
            "Production HMM n_iter "
            "must be 300."
        )

    if hmm_config.covariance_type != "full":
        raise AssertionError(
            "Production HMM covariance_type "
            "must remain 'full'."
        )

    if hmm_config.tol != 1e-4:
        raise AssertionError(
            "Production HMM tolerance "
            "must remain 1e-4."
        )

    print(
        "HMM CONFIGURATION: PASS"
    )

    # ------------------------------------------------------------------
    # 2. Existing unit/regression test suite.
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "backend/tests",
            "-v",
        ],
        "BACKEND TEST SUITE",
    )

    # ------------------------------------------------------------------
    # 3. Controlled walk-forward A0-A3 regression.
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "test_walk_forward_controlled.py",
        ],
        "CONTROLLED WALK-FORWARD A0-A3",
    )

    # ------------------------------------------------------------------
    # 4. Portfolio end-to-end integration.
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "test_portfolio_integration.py",
        ],
        "PORTFOLIO INTEGRATION",
    )

    # ------------------------------------------------------------------
    # 5. Stress scenario integration.
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "test_stress_integration.py",
        ],
        "STRESS INTEGRATION",
    )

    # ------------------------------------------------------------------
    # 6. HMM fold convergence with production n_iter=300.
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "test_hmm_fold_convergence.py",
        ],
        "HMM FOLD CONVERGENCE",
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE 1 FINAL REGRESSION: PASS"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()