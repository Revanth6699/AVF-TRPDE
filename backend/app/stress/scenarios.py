from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "return",
    "volatility",
)


class StressScenarioError(ValueError):
    """Raised when stress scenarios cannot be constructed."""


@dataclass(frozen=True)
class StressScenario:
    """Definition of one stress-testing scenario."""

    name: str
    description: str
    observation_count: int
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None
    asset_count: int


@dataclass(frozen=True)
class HistoricalWindow:
    """Explicit historical stress-testing window."""

    name: str
    start_date: date
    end_date: date


def identify_normal_volatility(
    dataframe: pd.DataFrame,
    *,
    lower_percentile: float = 0.00,
    upper_percentile: float = 50.00,
) -> pd.DataFrame:
    """
    Select historical observations in the normal-volatility range.

    The volatility thresholds are calculated from the supplied historical
    sample only.
    """

    _validate_input(dataframe)
    _validate_percentiles(
        lower_percentile,
        upper_percentile,
    )

    df = _prepare_dataframe(dataframe)

    lower = np.percentile(
        df["volatility"].to_numpy(dtype=float),
        lower_percentile,
    )
    upper = np.percentile(
        df["volatility"].to_numpy(dtype=float),
        upper_percentile,
    )

    result = df.loc[
        df["volatility"].between(
            lower,
            upper,
            inclusive="both",
        )
    ].copy()

    return _sort_result(result)


def identify_high_volatility(
    dataframe: pd.DataFrame,
    *,
    percentile: float = 90.00,
) -> pd.DataFrame:
    """
    Select historical observations at or above a high-volatility percentile.
    """

    _validate_input(dataframe)
    _validate_percentile(percentile)

    df = _prepare_dataframe(dataframe)

    threshold = np.percentile(
        df["volatility"].to_numpy(dtype=float),
        percentile,
    )

    result = df.loc[
        df["volatility"] >= threshold
    ].copy()

    return _sort_result(result)


def identify_volatility_spikes(
    dataframe: pd.DataFrame,
    *,
    percentile: float = 99.00,
) -> pd.DataFrame:
    """
    Select extreme historical volatility observations.

    The threshold is determined from the historical sample supplied to the
    function and is not fitted using future observations.
    """

    _validate_input(dataframe)
    _validate_percentile(percentile)

    df = _prepare_dataframe(dataframe)

    threshold = np.percentile(
        df["volatility"].to_numpy(dtype=float),
        percentile,
    )

    result = df.loc[
        df["volatility"] >= threshold
    ].copy()

    return _sort_result(result)


def identify_prolonged_drawdowns(
    dataframe: pd.DataFrame,
    *,
    minimum_drawdown: float = -0.10,
    minimum_duration: int = 5,
) -> pd.DataFrame:
    """
    Identify observations belonging to sustained portfolio/asset drawdowns.

    A drawdown is measured from the running cumulative-return peak.

    Parameters
    ----------
    minimum_drawdown:
        Negative drawdown threshold. For example, -0.10 means at least
        a 10% drawdown from the running peak.

    minimum_duration:
        Minimum consecutive number of observations below the threshold.
    """

    _validate_input(dataframe)

    if minimum_drawdown >= 0:
        raise StressScenarioError(
            "minimum_drawdown must be negative."
        )

    if (
        not isinstance(minimum_duration, int)
        or isinstance(minimum_duration, bool)
        or minimum_duration < 1
    ):
        raise StressScenarioError(
            "minimum_duration must be a positive integer."
        )

    df = _prepare_dataframe(dataframe)

    selected_frames: list[pd.DataFrame] = []

    for asset_id, group in df.groupby(
        "asset_id",
        sort=False,
    ):
        group = group.sort_values(
            "timestamp"
        ).copy()

        cumulative_return = (
            1.0 + group["return"]
        ).cumprod()

        running_peak = cumulative_return.cummax()

        drawdown = (
            cumulative_return / running_peak
        ) - 1.0

        group["drawdown"] = drawdown

        below_threshold = (
            group["drawdown"] <= minimum_drawdown
        )

        run_id = (
            below_threshold.ne(
                below_threshold.shift(fill_value=False)
            ).cumsum()
        )

        run_lengths = (
            below_threshold.groupby(run_id)
            .transform("sum")
        )

        sustained = (
            below_threshold
            & (run_lengths >= minimum_duration)
        )

        selected = group.loc[sustained].copy()

        if not selected.empty:
            selected_frames.append(selected)

    if not selected_frames:
        return _empty_result()

    result = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    return _sort_result(result)


def identify_regime_transitions(
    dataframe: pd.DataFrame,
    *,
    regime_column: str = "regime",
) -> pd.DataFrame:
    """
    Identify observations where the dominant HMM regime changes.

    The regime labels must already have been generated by the HMM
    walk-forward process. This function does not fit or infer a regime model.
    """

    _validate_input(dataframe)

    if regime_column not in dataframe.columns:
        raise StressScenarioError(
            f"Missing required regime column: {regime_column}"
        )

    df = _prepare_dataframe(dataframe)

    if df[regime_column].isna().any():
        raise StressScenarioError(
            "Regime column contains missing values."
        )

    selected_frames: list[pd.DataFrame] = []

    for asset_id, group in df.groupby(
        "asset_id",
        sort=False,
    ):
        group = group.sort_values(
            "timestamp"
        ).copy()

        previous_regime = (
            group[regime_column]
            .shift(1)
        )

        transition = (
            previous_regime.notna()
            & group[regime_column].ne(
                previous_regime
            )
        )

        selected = group.loc[
            transition
        ].copy()

        if not selected.empty:
            selected["previous_regime"] = (
                previous_regime.loc[
                    selected.index
                ].to_numpy()
            )

            selected_frames.append(selected)

    if not selected_frames:
        return _empty_result(
            include_regime_columns=True
        )

    result = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    return _sort_result(result)


def select_historical_window(
    dataframe: pd.DataFrame,
    *,
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """
    Select an explicitly defined historical stress window.

    Historical windows must be defined before evaluation rather than chosen
    retrospectively from observed strategy results.
    """

    _validate_input(dataframe)

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    if start > end:
        raise StressScenarioError(
            "start_date must be earlier than or equal to end_date."
        )

    df = _prepare_dataframe(dataframe)

    result = df.loc[
        df["timestamp"].dt.date.between(
            start,
            end,
            inclusive="both",
        )
    ].copy()

    return _sort_result(result)


def build_stress_scenarios(
    dataframe: pd.DataFrame,
    *,
    normal_lower_percentile: float = 0.00,
    normal_upper_percentile: float = 50.00,
    high_volatility_percentile: float = 90.00,
    volatility_spike_percentile: float = 99.00,
    minimum_drawdown: float = -0.10,
    minimum_drawdown_duration: int = 5,
    regime_column: str | None = None,
    historical_windows: tuple[
        HistoricalWindow, ...
    ] = (),
) -> dict[str, pd.DataFrame]:
    """
    Build the complete set of procedural stress scenarios.

    Returns
    -------
    dict[str, pd.DataFrame]
        Scenario name mapped to the historical observations belonging
        to that scenario.
    """

    _validate_input(dataframe)

    scenarios: dict[str, pd.DataFrame] = {}

    scenarios["normal_volatility"] = identify_normal_volatility(
        dataframe,
        lower_percentile=normal_lower_percentile,
        upper_percentile=normal_upper_percentile,
    )

    scenarios["high_volatility"] = identify_high_volatility(
        dataframe,
        percentile=high_volatility_percentile,
    )

    scenarios["volatility_spike"] = identify_volatility_spikes(
        dataframe,
        percentile=volatility_spike_percentile,
    )

    scenarios["prolonged_drawdown"] = identify_prolonged_drawdowns(
        dataframe,
        minimum_drawdown=minimum_drawdown,
        minimum_duration=minimum_drawdown_duration,
    )

    if regime_column is not None:
        scenarios["regime_transition"] = identify_regime_transitions(
            dataframe,
            regime_column=regime_column,
        )
    else:
        scenarios["regime_transition"] = _empty_result(
            include_regime_columns=True
        )

    for window in historical_windows:
        scenario_name = (
            f"historical_{window.name}"
        )

        scenarios[scenario_name] = select_historical_window(
            dataframe,
            start_date=window.start_date,
            end_date=window.end_date,
        )

    return scenarios


def summarize_scenarios(
    scenarios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Produce metadata describing each stress scenario.
    """

    records: list[dict[str, object]] = []

    for name, dataframe in scenarios.items():
        if dataframe.empty:
            records.append(
                {
                    "scenario": name,
                    "observation_count": 0,
                    "asset_count": 0,
                    "start_timestamp": None,
                    "end_timestamp": None,
                }
            )
            continue

        records.append(
            {
                "scenario": name,
                "observation_count": len(dataframe),
                "asset_count": int(
                    dataframe["asset_id"].nunique()
                ),
                "start_timestamp": dataframe[
                    "timestamp"
                ].min(),
                "end_timestamp": dataframe[
                    "timestamp"
                ].max(),
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "scenario",
            "observation_count",
            "asset_count",
            "start_timestamp",
            "end_timestamp",
        ],
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
        raise StressScenarioError(
            "Input dataframe must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise StressScenarioError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise StressScenarioError(
            "timestamp contains invalid or missing values."
        )

    if dataframe["asset_id"].isna().any():
        raise StressScenarioError(
            "asset_id contains missing values."
        )

    for column in (
        "return",
        "volatility",
    ):
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if values.isna().any():
            raise StressScenarioError(
                f"{column} contains invalid or missing values."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise StressScenarioError(
                f"{column} contains non-finite values."
            )

    if (
        pd.to_numeric(
            dataframe["volatility"],
            errors="coerce",
        ) < 0
    ).any():
        raise StressScenarioError(
            "volatility must not contain negative values."
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

    df["volatility"] = pd.to_numeric(
        df["volatility"],
        errors="coerce",
    )

    return (
        df.sort_values(
            by=["asset_id", "timestamp"],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def _sort_result(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.reset_index(
            drop=True
        )

    return (
        dataframe.sort_values(
            by=["timestamp", "asset_id"],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def _empty_result(
    *,
    include_regime_columns: bool = False,
) -> pd.DataFrame:
    columns = list(REQUIRED_COLUMNS)

    if include_regime_columns:
        columns.extend(
            [
                "regime",
                "previous_regime",
            ]
        )

    return pd.DataFrame(
        {
            column: pd.Series(dtype="object")
            for column in columns
        }
    )


def _validate_percentile(
    percentile: float,
) -> None:
    if not isinstance(
        percentile,
        (int, float),
    ) or isinstance(
        percentile,
        bool,
    ):
        raise StressScenarioError(
            "percentile must be numeric."
        )

    if not 0 <= float(percentile) <= 100:
        raise StressScenarioError(
            "percentile must be between 0 and 100."
        )


def _validate_percentiles(
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    _validate_percentile(
        lower_percentile
    )
    _validate_percentile(
        upper_percentile
    )

    if lower_percentile > upper_percentile:
        raise StressScenarioError(
            "lower_percentile must not exceed "
            "upper_percentile."
        )


def _parse_date(
    value: str | date,
) -> date:
    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(
                value.strip()
            )
        except ValueError as exc:
            raise StressScenarioError(
                f"Invalid date value: {value}"
            ) from exc

    raise TypeError(
        "Date values must be str or date."
    )