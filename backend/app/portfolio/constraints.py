from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_WEIGHT_COLUMNS = (
    "timestamp",
    "asset_id",
    "target_weight",
)


class ConstraintError(ValueError):
    """Raised when portfolio constraints cannot be applied."""


@dataclass(frozen=True)
class ConstraintResult:
    """Result of applying portfolio constraints."""

    weights: pd.DataFrame
    observation_count: int
    asset_count: int
    total_turnover: float
    maximum_turnover: float
    position_limit: float


@dataclass(frozen=True)
class TurnoverResult:
    """Portfolio turnover calculation result."""

    turnover: float
    asset_count: int


def apply_position_limits(
    weights: pd.DataFrame,
    *,
    max_position: float,
) -> pd.DataFrame:
    """
    Apply absolute position limits to portfolio weights.

    For every asset:

        -max_position <= weight <= max_position

    The operation is deterministic and does not introduce
    any trading rule or asset-selection logic.
    """

    _validate_weights_input(weights)
    _validate_max_position(max_position)

    result = weights.copy()

    result["target_weight"] = pd.to_numeric(
        result["target_weight"],
        errors="coerce",
    )

    if result["target_weight"].isna().any():
        raise ConstraintError(
            "target_weight contains invalid or missing values."
        )

    result["target_weight"] = result["target_weight"].clip(
        lower=-max_position,
        upper=max_position,
    )

    return result


def calculate_turnover(
    current_weights: pd.DataFrame,
    previous_weights: pd.DataFrame | None = None,
) -> TurnoverResult:
    """
    Calculate portfolio turnover.

    Turnover is:

        Turnover_t = sum_i |w_i,t - w_i,t-1|

    Missing previous weights are treated as zero weights.
    """

    _validate_weights_input(current_weights)

    if previous_weights is None:
        previous = pd.DataFrame(
            {
                "asset_id": current_weights["asset_id"].astype("string"),
                "target_weight": 0.0,
            }
        )
    else:
        _validate_weights_input(
            previous_weights,
            require_timestamp=False,
        )

        previous = previous_weights.loc[
            :,
            ["asset_id", "target_weight"],
        ].copy()

    current = current_weights.loc[
        :,
        ["asset_id", "target_weight"],
    ].copy()

    current["asset_id"] = current["asset_id"].astype("string")
    previous["asset_id"] = previous["asset_id"].astype("string")

    current["target_weight"] = pd.to_numeric(
        current["target_weight"],
        errors="coerce",
    )

    previous["target_weight"] = pd.to_numeric(
        previous["target_weight"],
        errors="coerce",
    )

    if current["target_weight"].isna().any():
        raise ConstraintError(
            "current_weights contains invalid target_weight values."
        )

    if previous["target_weight"].isna().any():
        raise ConstraintError(
            "previous_weights contains invalid target_weight values."
        )

    current_grouped = (
        current.groupby("asset_id", as_index=False)["target_weight"]
        .sum()
        .rename(columns={"target_weight": "current_weight"})
    )

    previous_grouped = (
        previous.groupby("asset_id", as_index=False)["target_weight"]
        .sum()
        .rename(columns={"target_weight": "previous_weight"})
    )

    merged = current_grouped.merge(
        previous_grouped,
        on="asset_id",
        how="outer",
    ).fillna(0.0)

    turnover = float(
        np.abs(
            merged["current_weight"]
            - merged["previous_weight"]
        ).sum()
    )

    return TurnoverResult(
        turnover=turnover,
        asset_count=int(len(merged)),
    )


def apply_turnover_limit(
    current_weights: pd.DataFrame,
    previous_weights: pd.DataFrame | None = None,
    *,
    max_turnover: float,
) -> pd.DataFrame:
    """
    Apply a portfolio turnover constraint.

    If the proposed turnover is within the limit, weights are
    returned unchanged.

    If turnover exceeds the limit, the change from previous
    weights is scaled proportionally so that total turnover
    equals max_turnover.

    This is a mathematical constraint projection, not a
    rule-based trading strategy.
    """

    _validate_weights_input(current_weights)
    _validate_max_turnover(max_turnover)

    if previous_weights is None:
        previous = pd.DataFrame(
            {
                "asset_id": current_weights["asset_id"].astype("string"),
                "target_weight": 0.0,
            }
        )
    else:
        _validate_weights_input(
            previous_weights,
            require_timestamp=False,
        )

        previous = previous_weights.loc[
            :,
            ["asset_id", "target_weight"],
        ].copy()

    result = current_weights.copy()

    result["asset_id"] = result["asset_id"].astype("string")
    previous["asset_id"] = previous["asset_id"].astype("string")

    result["target_weight"] = pd.to_numeric(
        result["target_weight"],
        errors="coerce",
    )

    previous["target_weight"] = pd.to_numeric(
        previous["target_weight"],
        errors="coerce",
    )

    if result["target_weight"].isna().any():
        raise ConstraintError(
            "current_weights contains invalid target_weight values."
        )

    if previous["target_weight"].isna().any():
        raise ConstraintError(
            "previous_weights contains invalid target_weight values."
        )

    previous_grouped = (
        previous.groupby("asset_id", as_index=False)["target_weight"]
        .sum()
        .rename(columns={"target_weight": "previous_weight"})
    )

    result = result.merge(
        previous_grouped,
        on="asset_id",
        how="left",
    )

    result["previous_weight"] = result[
        "previous_weight"
    ].fillna(0.0)

    proposed_change = (
        result["target_weight"]
        - result["previous_weight"]
    )

    proposed_turnover = float(
        np.abs(proposed_change).sum()
    )

    if proposed_turnover > max_turnover:
        if proposed_turnover <= 0.0:
            scale = 0.0
        else:
            scale = max_turnover / proposed_turnover

        result["target_weight"] = (
            result["previous_weight"]
            + proposed_change * scale
        )

    result = result.drop(
        columns=["previous_weight"],
    )

    return result


def apply_constraints(
    weights: pd.DataFrame,
    *,
    max_position: float,
    previous_weights: pd.DataFrame | None = None,
    max_turnover: float | None = None,
) -> ConstraintResult:
    """
    Apply the complete portfolio constraint layer.

    Constraint order:

    1. Position limits
    2. Turnover limit, when configured
    3. Final turnover calculation

    The function does not generate trading signals or
    make asset-selection decisions.
    """

    _validate_weights_input(weights)
    _validate_max_position(max_position)

    constrained = apply_position_limits(
        weights,
        max_position=max_position,
    )

    if max_turnover is not None:
        _validate_max_turnover(max_turnover)

        constrained = apply_turnover_limit(
            constrained,
            previous_weights=previous_weights,
            max_turnover=max_turnover,
        )

    turnover_result = calculate_turnover(
        constrained,
        previous_weights=previous_weights,
    )

    return ConstraintResult(
        weights=constrained,
        observation_count=len(constrained),
        asset_count=int(
            constrained["asset_id"].nunique()
        ),
        total_turnover=turnover_result.turnover,
        maximum_turnover=(
            max_turnover
            if max_turnover is not None
            else turnover_result.turnover
        ),
        position_limit=max_position,
    )


def validate_constraints(
    weights: pd.DataFrame,
    *,
    max_position: float,
    previous_weights: pd.DataFrame | None = None,
    max_turnover: float | None = None,
) -> bool:
    """
    Validate that portfolio weights satisfy the configured
    position and turnover constraints.
    """

    _validate_weights_input(weights)
    _validate_max_position(max_position)

    numeric_weights = pd.to_numeric(
        weights["target_weight"],
        errors="coerce",
    )

    if numeric_weights.isna().any():
        return False

    if (
        (numeric_weights.abs() > max_position + 1e-12)
        .any()
    ):
        return False

    if max_turnover is not None:
        _validate_max_turnover(max_turnover)

        turnover_result = calculate_turnover(
            weights,
            previous_weights=previous_weights,
        )

        if turnover_result.turnover > max_turnover + 1e-12:
            return False

    return True


def _validate_weights_input(
    weights: pd.DataFrame,
    *,
    require_timestamp: bool = True,
) -> None:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "weights must be a pandas.DataFrame."
        )

    if weights.empty:
        raise ConstraintError(
            "weights must not be empty."
        )

    required_columns = (
        REQUIRED_WEIGHT_COLUMNS
        if require_timestamp
        else ("asset_id", "target_weight")
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in weights.columns
    ]

    if missing_columns:
        raise ConstraintError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if weights["asset_id"].isna().any():
        raise ConstraintError(
            "asset_id contains missing values."
        )

    if (
        weights["asset_id"]
        .astype("string")
        .str.strip()
        .eq("")
        .any()
    ):
        raise ConstraintError(
            "asset_id contains empty values."
        )

    numeric_weights = pd.to_numeric(
        weights["target_weight"],
        errors="coerce",
    )

    if numeric_weights.isna().any():
        raise ConstraintError(
            "target_weight contains invalid or missing values."
        )

    if not np.isfinite(numeric_weights.to_numpy()).all():
        raise ConstraintError(
            "target_weight contains non-finite values."
        )

    if require_timestamp:
        timestamps = pd.to_datetime(
            weights["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ConstraintError(
                "timestamp contains invalid or missing values."
            )


def _validate_max_position(
    max_position: float,
) -> None:
    if not isinstance(
        max_position,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "max_position must be numeric."
        )

    if not np.isfinite(max_position):
        raise ConstraintError(
            "max_position must be finite."
        )

    if max_position <= 0:
        raise ConstraintError(
            "max_position must be greater than zero."
        )


def _validate_max_turnover(
    max_turnover: float,
) -> None:
    if not isinstance(
        max_turnover,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "max_turnover must be numeric."
        )

    if not np.isfinite(max_turnover):
        raise ConstraintError(
            "max_turnover must be finite."
        )

    if max_turnover < 0:
        raise ConstraintError(
            "max_turnover must not be negative."
        )