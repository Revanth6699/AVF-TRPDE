from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "return",
)


class StressRunnerError(ValueError):
    """Raised when stress-test execution fails."""


@dataclass(frozen=True)
class StressResult:
    """Result of one stress scenario."""

    scenario: str
    observation_count: int
    asset_count: int
    total_return: float
    mean_return: float
    volatility: float
    downside_deviation: float
    max_drawdown: float
    cumulative_return: float


@dataclass(frozen=True)
class StressRunResult:
    """Complete stress-test execution result."""

    results: pd.DataFrame
    scenario_count: int
    total_observations: int


def run_stress_scenario(
    dataframe: pd.DataFrame,
    *,
    scenario_name: str,
) -> StressResult:
    """
    Evaluate portfolio/asset returns inside one stress scenario.

    The function does not modify the scenario definition and does not
    introduce any trading or decision rules.
    """

    _validate_input(dataframe)

    scenario = str(scenario_name).strip()

    if not scenario:
        raise StressRunnerError(
            "scenario_name must not be empty."
        )

    df = _prepare_dataframe(dataframe)

    returns = df["return"].to_numpy(dtype=float)

    total_return = float(
        returns.sum()
    )

    cumulative_return = float(
        np.prod(1.0 + returns) - 1.0
    )

    mean_return = float(
        np.mean(returns)
    )

    volatility = _annualized_volatility(
        returns
    )

    downside_deviation = _annualized_downside_deviation(
        returns
    )

    max_drawdown = _maximum_drawdown(
        returns
    )

    return StressResult(
        scenario=scenario,
        observation_count=len(df),
        asset_count=int(
            df["asset_id"].nunique()
        ),
        total_return=total_return,
        mean_return=mean_return,
        volatility=volatility,
        downside_deviation=downside_deviation,
        max_drawdown=max_drawdown,
        cumulative_return=cumulative_return,
    )


def run_stress_scenarios(
    scenarios: dict[str, pd.DataFrame],
) -> StressRunResult:
    """
    Execute all supplied stress scenarios.

    Parameters
    ----------
    scenarios:
        Mapping produced by ``stress.scenarios.build_stress_scenarios``.

    Returns
    -------
    StressRunResult
        Structured results for every scenario.

    Empty scenarios are retained in the output with zero observations.
    """

    if not isinstance(
        scenarios,
        dict,
    ):
        raise TypeError(
            "scenarios must be a dictionary."
        )

    result_records: list[dict[str, object]] = []

    for scenario_name, dataframe in scenarios.items():
        if not isinstance(
            scenario_name,
            str,
        ):
            raise StressRunnerError(
                "Scenario names must be strings."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise StressRunnerError(
                f"Scenario '{scenario_name}' "
                "must contain a pandas.DataFrame."
            )

        if dataframe.empty:
            result_records.append(
                _empty_scenario_record(
                    scenario_name
                )
            )
            continue

        result = run_stress_scenario(
            dataframe,
            scenario_name=scenario_name,
        )

        result_records.append(
            _result_to_record(result)
        )

    result_dataframe = pd.DataFrame(
        result_records,
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
        ],
    )

    return StressRunResult(
        results=result_dataframe,
        scenario_count=len(result_dataframe),
        total_observations=int(
            result_dataframe[
                "observation_count"
            ].sum()
        )
        if not result_dataframe.empty
        else 0,
    )


def compare_stress_scenarios(
    stress_result: StressRunResult,
) -> pd.DataFrame:
    """
    Return a comparison table across stress scenarios.

    No scenario is ranked or declared superior. The function only exposes
    the measured stress-test metrics.
    """

    if not isinstance(
        stress_result,
        StressRunResult,
    ):
        raise TypeError(
            "stress_result must be a StressRunResult."
        )

    return stress_result.results.copy()


def _annualized_volatility(
    returns: np.ndarray,
) -> float:
    if len(returns) < 2:
        return 0.0

    standard_deviation = float(
        np.std(
            returns,
            ddof=1,
        )
    )

    return standard_deviation * np.sqrt(
        252.0
    )


def _annualized_downside_deviation(
    returns: np.ndarray,
    *,
    target_return: float = 0.0,
) -> float:
    downside = np.minimum(
        returns - target_return,
        0.0,
    )

    if len(downside) == 0:
        return 0.0

    downside_squared_mean = float(
        np.mean(
            np.square(downside)
        )
    )

    return float(
        np.sqrt(
            downside_squared_mean
        )
        * np.sqrt(252.0)
    )


def _maximum_drawdown(
    returns: np.ndarray,
) -> float:
    if len(returns) == 0:
        return 0.0

    wealth = np.cumprod(
        1.0 + returns
    )

    running_peak = np.maximum.accumulate(
        wealth
    )

    drawdowns = (
        wealth / running_peak
    ) - 1.0

    return float(
        np.min(drawdowns)
    )


def _prepare_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["asset_id"] = (
        df["asset_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["return"] = pd.to_numeric(
        df["return"],
        errors="coerce",
    )

    return (
        df.sort_values(
            by=[
                "timestamp",
                "asset_id",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas.DataFrame."
        )

    if dataframe.empty:
        raise StressRunnerError(
            "Stress scenario dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise StressRunnerError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise StressRunnerError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise StressRunnerError(
            "asset_id contains missing values."
        )

    returns = pd.to_numeric(
        dataframe["return"],
        errors="coerce",
    )

    if returns.isna().any():
        raise StressRunnerError(
            "return contains invalid or missing values."
        )

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise StressRunnerError(
            "return contains non-finite values."
        )


def _result_to_record(
    result: StressResult,
) -> dict[str, object]:
    return {
        "scenario": result.scenario,
        "observation_count": result.observation_count,
        "asset_count": result.asset_count,
        "total_return": result.total_return,
        "mean_return": result.mean_return,
        "volatility": result.volatility,
        "downside_deviation": result.downside_deviation,
        "max_drawdown": result.max_drawdown,
        "cumulative_return": result.cumulative_return,
    }


def _empty_scenario_record(
    scenario_name: str,
) -> dict[str, object]:
    return {
        "scenario": scenario_name,
        "observation_count": 0,
        "asset_count": 0,
        "total_return": np.nan,
        "mean_return": np.nan,
        "volatility": np.nan,
        "downside_deviation": np.nan,
        "max_drawdown": np.nan,
        "cumulative_return": np.nan,
    }