from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from backend.app.research.evaluation.diebold_mariano import (
    diebold_mariano_test,
)
from backend.app.research.evaluation.forecast_metrics import (
    calculate_forecast_metrics,
)
from backend.app.research.evaluation.mcs import model_confidence_set


_REQUIRED_COLUMNS = {
    "timestamp",
    "asset_id",
    "model",
    "prediction",
    "actual_volatility",
}
_SUPPORTED_METRICS = {"MAE", "RMSE", "QLIKE"}


class WalkForwardEvaluationError(ValueError):
    """Raised when out-of-sample predictions cannot be evaluated."""


def evaluate_walk_forward_predictions(
    predictions: pd.DataFrame,
    *,
    metrics: tuple[str, ...] | list[str] = ("MAE", "RMSE", "QLIKE"),
    n_bootstrap: int = 1_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate aligned, strictly out-of-sample predictions by asset.

    Per-asset reporting avoids pooling volatility levels across assets with
    materially different scales. Pairwise DM tests and MCS use only the
    common timestamp intersection for the asset being evaluated.
    Confidence intervals use a moving-block bootstrap to retain short-range
    serial dependence in the out-of-sample loss sequence.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas.DataFrame.")
    if predictions.empty:
        raise WalkForwardEvaluationError("No out-of-sample predictions were supplied.")

    missing = sorted(_REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise WalkForwardEvaluationError(
            "Prediction data is missing required columns: " + ", ".join(missing)
        )

    requested = tuple(str(metric).upper() for metric in metrics)
    if not requested or len(set(requested)) != len(requested):
        raise WalkForwardEvaluationError("Metrics must be a non-empty unique list.")
    invalid = sorted(set(requested) - _SUPPORTED_METRICS)
    if invalid:
        raise WalkForwardEvaluationError(
            "Unsupported metrics: " + ", ".join(invalid)
        )
    if isinstance(n_bootstrap, bool) or n_bootstrap < 100:
        raise WalkForwardEvaluationError("n_bootstrap must be at least 100.")
    if not 0.0 < confidence_level < 1.0:
        raise WalkForwardEvaluationError("confidence_level must be between 0 and 1.")

    frame = predictions.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("prediction", "actual_volatility"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(_REQUIRED_COLUMNS)].isna().any().any():
        raise WalkForwardEvaluationError("Predictions contain missing or invalid values.")
    if not np.isfinite(frame[["prediction", "actual_volatility"]].to_numpy(dtype=float)).all():
        raise WalkForwardEvaluationError("Predictions contain non-finite numeric values.")
    if (frame["prediction"] <= 0).any() or (frame["actual_volatility"] <= 0).any():
        raise WalkForwardEvaluationError(
            "Volatility forecasts and realized-volatility targets must be strictly positive."
        )
    if frame.duplicated(["asset_id", "timestamp", "model"]).any():
        raise WalkForwardEvaluationError(
            "Duplicate asset/timestamp/model prediction rows were found."
        )

    output: dict[str, Any] = {
        "method": "per_asset_walk_forward_oos",
        "metrics": list(requested),
        "confidence_level": float(confidence_level),
        "bootstrap": {
            "method": "moving_block_percentile",
            "samples": int(n_bootstrap),
            "block_length_rule": "ceil(sqrt(n_observations))",
        },
        "assets": {},
    }

    for asset_index, (asset_id, asset_frame) in enumerate(
        frame.groupby("asset_id", sort=True)
    ):
        asset_frame = asset_frame.sort_values("timestamp")
        asset_result: dict[str, Any] = {
            "observation_counts": {},
            "model_metrics": {},
            "pairwise_diebold_mariano": {},
            "model_confidence_set": None,
        }

        model_frames: dict[str, pd.DataFrame] = {}
        for model, model_frame in asset_frame.groupby("model", sort=True):
            model_frame = model_frame.sort_values("timestamp").reset_index(drop=True)
            model_frames[str(model)] = model_frame
            actual = model_frame["actual_volatility"].to_numpy(dtype=float)
            forecast = model_frame["prediction"].to_numpy(dtype=float)
            calculated = calculate_forecast_metrics(actual, forecast)
            all_values = calculated.as_dict()
            model_metrics: dict[str, Any] = {}
            for metric in requested:
                metric_key = metric.lower()
                estimate = float(all_values[metric_key])
                ci = _moving_block_metric_interval(
                    actual,
                    forecast,
                    metric=metric_key,
                    samples=n_bootstrap,
                    confidence_level=confidence_level,
                    random_state=random_state + asset_index * 1009 + len(model_metrics),
                ) if len(actual) >= 2 else None
                model_metrics[metric] = {
                    "value": estimate,
                    "confidence_interval": ci,
                }
            asset_result["model_metrics"][str(model)] = model_metrics
            asset_result["observation_counts"][str(model)] = int(len(model_frame))

        model_names = sorted(model_frames)
        if len(model_names) >= 2:
            aligned = None
            for model in model_names:
                current = model_frames[model].loc[
                    :, ["timestamp", "actual_volatility", "prediction"]
                ].rename(columns={
                    "actual_volatility": f"actual__{model}",
                    "prediction": f"forecast__{model}",
                })
                aligned = current if aligned is None else aligned.merge(
                    current, on="timestamp", how="inner", validate="one_to_one"
                )

            if aligned is not None and len(aligned) >= 2:
                actual_columns = [f"actual__{model}" for model in model_names]
                reference_actual = aligned[actual_columns[0]].to_numpy(dtype=float)
                if not all(
                    np.allclose(reference_actual, aligned[column].to_numpy(dtype=float))
                    for column in actual_columns[1:]
                ):
                    raise WalkForwardEvaluationError(
                        f"Actual targets do not agree across models for asset {asset_id}."
                    )

                loss_matrix = []
                for model in model_names:
                    forecast = aligned[f"forecast__{model}"].to_numpy(dtype=float)
                    dm = {}
                    for other in model_names:
                        if model == other:
                            continue
                        other_forecast = aligned[f"forecast__{other}"].to_numpy(dtype=float)
                        result = diebold_mariano_test(
                            reference_actual,
                            forecast,
                            other_forecast,
                            loss="qlike",
                            alternative="two_sided",
                        )
                        dm[other] = {
                            "statistic": float(result.statistic),
                            "p_value": float(result.p_value),
                            "mean_loss_difference": float(result.mean_loss_difference),
                            "observations": int(result.observations),
                            "loss": "QLIKE",
                        }
                    asset_result["pairwise_diebold_mariano"][model] = dm
                    ratio = reference_actual / forecast
                    loss_matrix.append(ratio - np.log(ratio) - 1.0)

                if len(aligned) >= 10:
                    mcs_result = model_confidence_set(
                        np.column_stack(loss_matrix),
                        models=model_names,
                        alpha=0.10,
                        n_bootstrap=max(100, n_bootstrap),
                        random_state=random_state + asset_index,
                    )
                    asset_result["model_confidence_set"] = {
                        "included_models": list(mcs_result.included_models),
                        "eliminated_models": list(mcs_result.eliminated_models),
                        "confidence_level": float(mcs_result.confidence_level),
                        "p_value": float(mcs_result.p_value),
                        "test_statistic": float(mcs_result.test_statistic),
                        "observations": int(len(aligned)),
                        "bootstrap_samples": int(mcs_result.bootstrap_samples),
                        "block_length": int(mcs_result.block_length),
                    }
                else:
                    asset_result["model_confidence_set"] = {
                        "status": "not_run",
                        "reason": "At least 10 common out-of-sample observations are required.",
                        "common_observations": int(len(aligned)),
                    }
            else:
                asset_result["comparison_status"] = (
                    "Insufficient common timestamps for paired model comparisons."
                )

        output["assets"][str(asset_id)] = asset_result

    if not output["assets"]:
        raise WalkForwardEvaluationError("No asset groups were available for evaluation.")
    return output


def _moving_block_metric_interval(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    metric: str,
    samples: int,
    confidence_level: float,
    random_state: int,
) -> dict[str, float | int]:
    n = len(actual)
    block_length = max(1, int(np.ceil(np.sqrt(n))))
    rng = np.random.default_rng(random_state)
    starts_max = n - block_length
    estimates = np.empty(samples, dtype=float)

    def calculate(indices: np.ndarray) -> float:
        y = actual[indices]
        yhat = forecast[indices]
        if metric == "mae":
            return float(np.mean(np.abs(y - yhat)))
        if metric == "rmse":
            return float(np.sqrt(np.mean(np.square(y - yhat))))
        ratio = y / yhat
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    for index in range(samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, starts_max + 1))
            indices.extend(range(start, start + block_length))
        estimates[index] = calculate(np.asarray(indices[:n], dtype=int))

    alpha = 1.0 - confidence_level
    return {
        "lower": float(np.quantile(estimates, alpha / 2.0)),
        "upper": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "samples": int(samples),
        "block_length": int(block_length),
    }
