from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.risk.pipeline import (
    RiskPipelineError,
    calculate_fhs_risk_points,
)


def _predictions(rows: int = 106, assets: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    frames = []
    timestamps = pd.date_range("2024-01-02", periods=rows, freq="B")
    for asset_index, asset_id in enumerate(assets):
        rng = np.random.default_rng(100 + asset_index)
        frames.append(
            pd.DataFrame(
                {
                    "target_timestamp": timestamps,
                    "asset_id": asset_id,
                    "target_return": rng.normal(
                        loc=0.0002 * asset_index,
                        scale=0.012 + 0.004 * asset_index,
                        size=rows,
                    ),
                    "prediction": rng.uniform(
                        0.008 + 0.002 * asset_index,
                        0.016 + 0.002 * asset_index,
                        size=rows,
                    ),
                    "model": "A0",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_fhs_risk_forecasts_use_only_strictly_prior_observations() -> None:
    frame = _predictions(rows=106)
    kwargs = {
        "assets": ["AAA"],
        "start_timestamp": frame["target_timestamp"].min(),
        "end_timestamp": frame["target_timestamp"].max(),
        "min_observations": 100,
        "n_simulations": 1_000,
        "random_state": 17,
    }

    baseline = calculate_fhs_risk_points(frame, **kwargs)
    assert len(baseline) == 6
    assert baseline[0]["history_count"] == 100

    changed = frame.copy()
    changed.loc[100, "target_return"] = -0.50
    after_change = calculate_fhs_risk_points(changed, **kwargs)

    # The return realized on the first forecast date cannot change that
    # date's own VaR/ES. It can affect only later forecasts.
    assert baseline[0]["timestamp"] == after_change[0]["timestamp"]
    assert baseline[0]["var_95"] == pytest.approx(after_change[0]["var_95"])
    assert baseline[0]["var_99"] == pytest.approx(after_change[0]["var_99"])
    assert baseline[0]["es_95"] == pytest.approx(after_change[0]["es_95"])
    assert baseline[1]["var_95"] != pytest.approx(after_change[1]["var_95"])


def test_fhs_risk_histories_are_independent_by_asset() -> None:
    frame = _predictions(rows=104, assets=("AAA", "BBB"))
    kwargs = {
        "start_timestamp": frame["target_timestamp"].min(),
        "end_timestamp": frame["target_timestamp"].max(),
        "min_observations": 100,
        "n_simulations": 1_000,
        "random_state": 31,
    }

    separate = calculate_fhs_risk_points(
        frame.loc[frame["asset_id"] == "AAA"],
        assets=["AAA"],
        **kwargs,
    )
    combined = calculate_fhs_risk_points(
        frame,
        assets=["AAA", "BBB"],
        **kwargs,
    )
    combined_aaa = [point for point in combined if point["asset_id"] == "AAA"]

    assert len(separate) == 4
    assert len(combined_aaa) == 4
    for expected, actual in zip(separate, combined_aaa, strict=True):
        assert expected["timestamp"] == actual["timestamp"]
        assert expected["var_95"] == pytest.approx(actual["var_95"])
        assert expected["var_99"] == pytest.approx(actual["var_99"])
        assert expected["es_95"] == pytest.approx(actual["es_95"])


def test_fhs_date_filter_keeps_prior_history_for_initialization() -> None:
    frame = _predictions(rows=106)
    points = calculate_fhs_risk_points(
        frame,
        assets=["AAA"],
        start_timestamp=frame.iloc[102]["target_timestamp"],
        end_timestamp=frame.iloc[105]["target_timestamp"],
        min_observations=100,
        n_simulations=1_000,
    )

    assert len(points) == 4
    assert points[0]["history_count"] == 102
    assert points[0]["timestamp"] == frame.iloc[102]["target_timestamp"]


def test_fhs_requires_next_day_return_alignment_fields() -> None:
    frame = _predictions(rows=105).drop(columns=["target_return"])
    with pytest.raises(RiskPipelineError, match="target_return"):
        calculate_fhs_risk_points(
            frame,
            assets=["AAA"],
            start_timestamp="2024-01-01",
            end_timestamp="2025-01-01",
            min_observations=100,
            n_simulations=1_000,
        )
