from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "scenario",
    "observation_count",
    "asset_count",
    "total_return",
    "mean_return",
    "volatility",
    "downside_deviation",
    "max_drawdown",
    "cumulative_return",
)


class StressAnalysisError(ValueError):
    """Raised when stress-test analysis cannot be performed."""


@dataclass(frozen=True)
class StressAnalysisResult:
    """Container for stress-test analytical results."""

    scenario_summary: pd.DataFrame
    comparison: pd.DataFrame
    scenario_count: int
    valid_scenario_count: int


def analyze_stress_results(
    results: pd.DataFrame,
) -> StressAnalysisResult:
    """
    Analyze scenario-level stress-test results.

    The analysis is descriptive and compares the metrics produced by
    the stress runner.

    Scenarios with zero observations are retained in the scenario
    summary but excluded from numerical comparisons.

    No new trading rules, models, or scenario definitions are created.
    """

    _validate_input(results)

    dataframe = results.loc[:, REQUIRED_COLUMNS].copy()

    dataframe["scenario"] = dataframe["scenario"].astype("string")

    numeric_columns = [
        "observation_count",
        "asset_count",
        "total_return",
        "mean_return",
        "volatility",
        "downside_deviation",
        "max_drawdown",
        "cumulative_return",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.sort_values(
        by="scenario",
        ascending=True,
    ).reset_index(drop=True)

    valid = dataframe.loc[
        dataframe["observation_count"] > 0
    ].copy()

    comparison = _build_comparison(valid)

    return StressAnalysisResult(
        scenario_summary=dataframe,
        comparison=comparison,
        scenario_count=len(dataframe),
        valid_scenario_count=len(valid),
    )


def summarize_stress_analysis(
    analysis: StressAnalysisResult,
) -> pd.DataFrame:
    """
    Return the scenario-level stress summary.
    """

    if not isinstance(analysis, StressAnalysisResult):
        raise TypeError(
            "analysis must be a StressAnalysisResult."
        )

    return analysis.scenario_summary.copy()


def compare_stress_scenarios(
    analysis: StressAnalysisResult,
) -> pd.DataFrame:
    """
    Return the numerical comparison across valid stress scenarios.
    """

    if not isinstance(analysis, StressAnalysisResult):
        raise TypeError(
            "analysis must be a StressAnalysisResult."
        )

    return analysis.comparison.copy()


def calculate_stress_changes(
    results: pd.DataFrame,
    *,
    reference_scenario: str,
) -> pd.DataFrame:
    """
    Calculate metric changes relative to a reference scenario.

    Changes are calculated as:

        scenario_metric - reference_metric

    Only scenarios containing observations are included.
    """

    _validate_input(results)

    reference_scenario = reference_scenario.strip()

    if not reference_scenario:
        raise StressAnalysisError(
            "reference_scenario must not be empty."
        )

    dataframe = results.loc[:, REQUIRED_COLUMNS].copy()

    dataframe["scenario"] = dataframe["scenario"].astype("string")

    numeric_columns = [
        "observation_count",
        "asset_count",
        "total_return",
        "mean_return",
        "volatility",
        "downside_deviation",
        "max_drawdown",
        "cumulative_return",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    valid = dataframe.loc[
        dataframe["observation_count"] > 0
    ].copy()

    reference = valid.loc[
        valid["scenario"] == reference_scenario
    ]

    if reference.empty:
        raise StressAnalysisError(
            f"Reference scenario '{reference_scenario}' "
            "was not found or contains no observations."
        )

    if len(reference) > 1:
        raise StressAnalysisError(
            f"Reference scenario '{reference_scenario}' "
            "must occur exactly once."
        )

    reference_row = reference.iloc[0]

    metric_columns = [
        "total_return",
        "mean_return",
        "volatility",
        "downside_deviation",
        "max_drawdown",
        "cumulative_return",
    ]

    comparison = valid.loc[
        :,
        ["scenario", "observation_count", "asset_count"],
    ].copy()

    for column in metric_columns:
        comparison[f"{column}_change"] = (
            valid[column].to_numpy()
            - float(reference_row[column])
        )

    comparison["reference_scenario"] = reference_scenario

    return comparison.reset_index(drop=True)


def build_stress_analysis_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact analytical table for reports and dashboards.
    """

    _validate_input(results)

    dataframe = results.loc[:, REQUIRED_COLUMNS].copy()

    dataframe["scenario"] = dataframe["scenario"].astype("string")

    for column in REQUIRED_COLUMNS[1:]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["has_observations"] = (
        dataframe["observation_count"] > 0
    )

    dataframe["return_per_observation"] = np.where(
        dataframe["observation_count"] > 0,
        dataframe["total_return"]
        / dataframe["observation_count"],
        np.nan,
    )

    dataframe["drawdown_magnitude"] = (
        dataframe["max_drawdown"].abs()
    )

    dataframe["volatility_to_return_ratio"] = np.where(
        dataframe["cumulative_return"].abs() > 0,
        dataframe["volatility"]
        / dataframe["cumulative_return"].abs(),
        np.nan,
    )

    columns = [
        "scenario",
        "observation_count",
        "asset_count",
        "total_return",
        "mean_return",
        "volatility",
        "downside_deviation",
        "max_drawdown",
        "cumulative_return",
        "has_observations",
        "return_per_observation",
        "drawdown_magnitude",
        "volatility_to_return_ratio",
    ]

    return dataframe.loc[:, columns].sort_values(
        by="scenario",
        ascending=True,
    ).reset_index(drop=True)


def _build_comparison(
    valid_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive cross-scenario comparison statistics.
    """

    if valid_results.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "observation_count",
                "asset_count",
                "total_return",
                "mean_return",
                "volatility",
                "downside_deviation",
                "max_drawdown",
                "cumulative_return",
                "return_change_vs_mean",
                "volatility_change_vs_mean",
                "drawdown_change_vs_mean",
            ]
        )

    comparison = valid_results.loc[
        :,
        [
            "scenario",
            "observation_count",
            "asset_count",
            "total_return",
            "mean_return",
            "volatility",
            "downside_deviation",
            "max_drawdown",
            "cumulative_return",
        ],
    ].copy()

    reference_metrics = {
        "total_return": float(
            valid_results["total_return"].mean()
        ),
        "volatility": float(
            valid_results["volatility"].mean()
        ),
        "max_drawdown": float(
            valid_results["max_drawdown"].mean()
        ),
    }

    comparison["return_change_vs_mean"] = (
        comparison["total_return"]
        - reference_metrics["total_return"]
    )

    comparison["volatility_change_vs_mean"] = (
        comparison["volatility"]
        - reference_metrics["volatility"]
    )

    comparison["drawdown_change_vs_mean"] = (
        comparison["max_drawdown"]
        - reference_metrics["max_drawdown"]
    )

    return comparison.reset_index(drop=True)


def _validate_input(
    results: pd.DataFrame,
) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "results must be a pandas.DataFrame."
        )

    if results.empty:
        raise StressAnalysisError(
            "Stress-test results must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in results.columns
    ]

    if missing_columns:
        raise StressAnalysisError(
            "Missing required stress-result columns: "
            + ", ".join(missing_columns)
        )

    if results["scenario"].isna().any():
        raise StressAnalysisError(
            "scenario contains missing values."
        )

    scenario_names = (
        results["scenario"]
        .astype("string")
        .str.strip()
    )

    if (scenario_names == "").any():
        raise StressAnalysisError(
            "scenario contains empty values."
        )

    for column in (
        "observation_count",
        "asset_count",
    ):
        values = pd.to_numeric(
            results[column],
            errors="coerce",
        )

        if values.isna().any():
            raise StressAnalysisError(
                f"{column} contains invalid or missing values."
            )

        if (values < 0).any():
            raise StressAnalysisError(
                f"{column} must not contain negative values."
            )

    for column in (
        "total_return",
        "mean_return",
        "volatility",
        "downside_deviation",
        "max_drawdown",
        "cumulative_return",
    ):
        values = pd.to_numeric(
            results[column],
            errors="coerce",
        )

        if values.isna().any():
            raise StressAnalysisError(
                f"{column} contains invalid or missing values."
            )

        if not np.isfinite(values.to_numpy()).all():
            raise StressAnalysisError(
                f"{column} contains non-finite values."
            )

    observation_counts = pd.to_numeric(
        results["observation_count"],
        errors="coerce",
    )

    asset_counts = pd.to_numeric(
        results["asset_count"],
        errors="coerce",
    )

    zero_observation_mask = observation_counts == 0

    if (
        asset_counts.loc[zero_observation_mask] != 0
    ).any():
        raise StressAnalysisError(
            "asset_count must be zero when observation_count is zero."
        )

    if (
        asset_counts.loc[observation_counts > 0] <= 0
    ).any():
        raise StressAnalysisError(
            "asset_count must be greater than zero when "
            "observation_count is greater than zero."
        )