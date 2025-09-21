"""Evaluation helpers for regression-style ML pipelines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _to_python_scalar(value):
    """Convert numpy scalar types to native Python values."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    """Compute a collection of regression error metrics."""

    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)

    if actual_arr.size == 0:
        raise ValueError("Cannot compute metrics with no observations")

    residuals = predicted_arr - actual_arr
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    variance = np.var(actual_arr)
    r2 = float(1 - (np.mean(residuals ** 2) / variance)) if variance > 1e-12 else float("nan")
    bias = float(np.mean(residuals))
    median_ae = float(np.median(np.abs(residuals)))

    non_zero_actual = np.where(np.abs(actual_arr) > 1e-12)[0]
    if non_zero_actual.size:
        mape = float(np.mean(np.abs(residuals[non_zero_actual] / actual_arr[non_zero_actual])) * 100)
    else:
        mape = float("nan")

    if actual_arr.size > 1 and np.std(actual_arr) > 1e-12 and np.std(predicted_arr) > 1e-12:
        pearson_r = float(np.corrcoef(actual_arr, predicted_arr)[0, 1])
    else:
        pearson_r = float("nan")

    return {
        "count": int(actual_arr.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "median_ae": median_ae,
        "mape": mape,
        "pearson_r": pearson_r,
        "actual_mean": float(np.mean(actual_arr)),
        "predicted_mean": float(np.mean(predicted_arr)),
    }


def grouped_metrics(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    group_column: str,
) -> List[Dict[str, float]]:
    """Evaluate metrics for each group in ``group_column``."""

    if group_column not in frame.columns:
        return []

    required = [actual_column, predicted_column, group_column]
    valid = frame.dropna(subset=required)
    if valid.empty:
        return []

    results: List[Dict[str, float]] = []
    for group_value, group_df in valid.groupby(group_column):
        metrics = regression_metrics(group_df[actual_column].to_numpy(), group_df[predicted_column].to_numpy())
        metrics.update({
            "group": _to_python_scalar(group_value),
        })
        results.append(metrics)

    return results


def compute_baseline_metrics(
    dataset: pd.DataFrame,
    target: str,
    history: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate naive baseline metrics for comparison with the model."""

    if dataset.empty:
        return {}

    required = {"season", "player_id", "gameweek", target}
    if not required.issubset(dataset.columns):
        return {}

    history_frame = history if history is not None else dataset
    history_ordered = history_frame.sort_values(["season", "player_id", "gameweek"]).reset_index(drop=True)

    key_columns = ["season", "player_id", "gameweek"]
    baseline_frame = history_ordered[key_columns].copy()

    player_group = history_ordered.groupby(["season", "player_id"], sort=False)
    baseline_frame["previous_match_points"] = player_group[target].shift(1)

    cumsum = player_group[target].cumsum() - history_ordered[target]
    counts = player_group.cumcount()
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_frame["player_cumulative_average"] = np.where(counts > 0, cumsum / counts, np.nan)

    if {"team_id"}.issubset(history_ordered.columns):
        team_group = history_ordered.groupby(["season", "team_id"], sort=False)
        team_cumsum = team_group[target].cumsum() - history_ordered[target]
        team_counts = team_group.cumcount()
        with np.errstate(divide="ignore", invalid="ignore"):
            baseline_frame["team_history_average"] = np.where(
                team_counts > 0, team_cumsum / team_counts, np.nan
            )

    merged = dataset.merge(baseline_frame, on=key_columns, how="left")

    baselines: Dict[str, Dict[str, float]] = {}
    for column in ("previous_match_points", "player_cumulative_average", "team_history_average"):
        if column in merged.columns:
            baselines.update(
                _metric_from_series(
                    name=column,
                    actual=merged[target],
                    predicted=merged[column],
                    total=len(merged),
                )
            )

    return {key: value for key, value in baselines.items() if value}


def _metric_from_series(
    name: str,
    actual: pd.Series,
    predicted: pd.Series,
    total: int,
) -> Dict[str, Dict[str, float]]:
    """Helper to compute metrics for a specific baseline series."""

    if isinstance(predicted, pd.Series):
        predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy()
    else:
        predicted_values = np.asarray(predicted, dtype=float)

    mask = ~np.isnan(predicted_values)
    if not np.any(mask):
        return {}

    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy()
    metrics = regression_metrics(actual_values[mask], predicted_values[mask])
    metrics["coverage"] = float(mask.sum() / total)
    return {name: metrics}

