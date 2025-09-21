"""Evaluation helpers for regression-style ML pipelines without pandas or numpy."""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Optional

from .utils import Dataset, to_float


def _to_python_scalar(value):
    return value


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    actual_list = list(actual)
    predicted_list = list(predicted)

    if not actual_list:
        raise ValueError("Cannot compute metrics with no observations")

    residuals = [pred - act for pred, act in zip(predicted_list, actual_list)]
    abs_residuals = [abs(r) for r in residuals]
    mae = sum(abs_residuals) / len(abs_residuals)
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    if len(actual_list) > 1:
        variance = statistics.pvariance(actual_list)
    else:
        variance = 0.0
    r2 = 1 - (sum(r * r for r in residuals) / len(residuals) / variance) if variance > 1e-12 else float("nan")
    bias = sum(residuals) / len(residuals)
    median_ae = statistics.median(abs_residuals)

    non_zero_actual = [abs(value) > 1e-12 for value in actual_list]
    if any(non_zero_actual):
        mape_values = [abs(residuals[i] / actual_list[i]) for i, keep in enumerate(non_zero_actual) if keep]
        mape = sum(mape_values) / len(mape_values) * 100 if mape_values else float("nan")
    else:
        mape = float("nan")

    if len(actual_list) > 1:
        mean_actual = sum(actual_list) / len(actual_list)
        mean_pred = sum(predicted_list) / len(predicted_list)
        cov = sum((a - mean_actual) * (p - mean_pred) for a, p in zip(actual_list, predicted_list))
        std_actual = math.sqrt(sum((a - mean_actual) ** 2 for a in actual_list))
        std_pred = math.sqrt(sum((p - mean_pred) ** 2 for p in predicted_list))
        if std_actual > 1e-12 and std_pred > 1e-12:
            pearson_r = cov / (std_actual * std_pred)
        else:
            pearson_r = float("nan")
    else:
        pearson_r = float("nan")

    return {
        "count": len(actual_list),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "median_ae": median_ae,
        "mape": mape,
        "pearson_r": pearson_r,
        "actual_mean": sum(actual_list) / len(actual_list),
        "predicted_mean": sum(predicted_list) / len(predicted_list),
    }


def grouped_metrics(
    records: Dataset,
    actual_column: str,
    predicted_column: str,
    group_column: str,
) -> List[Dict[str, float]]:
    if not records:
        return []

    results: List[Dict[str, float]] = []
    grouped: Dict[object, Dict[str, List[float]]] = {}
    for row in records:
        if group_column not in row:
            continue
        actual = to_float(row.get(actual_column))
        predicted = to_float(row.get(predicted_column))
        if math.isnan(actual) or math.isnan(predicted):
            continue
        group_value = row[group_column]
        entry = grouped.setdefault(group_value, {"actual": [], "predicted": []})
        entry["actual"].append(actual)
        entry["predicted"].append(predicted)

    for group_value, values in grouped.items():
        if not values["actual"]:
            continue
        metrics = regression_metrics(values["actual"], values["predicted"])
        metrics.update({"group": _to_python_scalar(group_value)})
        results.append(metrics)

    return results


def compute_baseline_metrics(
    dataset: Dataset,
    target: str,
    history: Optional[Dataset] = None,
) -> Dict[str, Dict[str, float]]:
    if not dataset:
        return {}

    required = {"season", "player_id", "gameweek", target}
    if not all(required.issubset(row.keys()) for row in dataset):
        return {}

    history_dataset = history if history is not None else dataset
    if not history_dataset:
        return {}

    sorted_history = sorted(history_dataset, key=lambda row: (row["season"], row["player_id"], row["gameweek"]))

    baseline_predictions: Dict[tuple, Dict[str, float]] = {}
    player_state: Dict[tuple, Dict[str, float]] = {}
    team_state: Dict[tuple, Dict[str, float]] = {}

    for row in sorted_history:
        key = (row["season"], row["player_id"], row["gameweek"])
        player_key = (row["season"], row["player_id"])
        player_stats = player_state.setdefault(player_key, {"count": 0.0, "sum": 0.0, "previous": float("nan")})

        team_id = row.get("team_id")
        if team_id is not None:
            team_key = (row["season"], team_id)
            team_stats = team_state.setdefault(team_key, {"count": 0.0, "sum": 0.0})
        else:
            team_stats = None

        entry = baseline_predictions.setdefault(key, {})
        previous_value = player_stats.get("previous")
        if previous_value is not None and not math.isnan(previous_value):
            entry["previous_match_points"] = player_stats["previous"]
        if player_stats["count"] > 0:
            entry["player_cumulative_average"] = player_stats["sum"] / player_stats["count"]
        if team_stats and team_stats["count"] > 0:
            entry["team_history_average"] = team_stats["sum"] / team_stats["count"]

        target_value = to_float(row.get(target))
        if math.isnan(target_value):
            continue
        player_stats["previous"] = target_value
        player_stats["sum"] += target_value
        player_stats["count"] += 1
        if team_stats is not None:
            team_stats["sum"] += target_value
            team_stats["count"] += 1

    baselines: Dict[str, Dict[str, float]] = {}
    total = len(dataset)
    for column in ("previous_match_points", "player_cumulative_average", "team_history_average"):
        actual_values: List[float] = []
        predicted_values: List[float] = []
        for row in dataset:
            key = (row["season"], row["player_id"], row["gameweek"])
            entry = baseline_predictions.get(key)
            if not entry or column not in entry:
                continue
            actual = to_float(row.get(target))
            predicted = to_float(entry[column])
            if math.isnan(actual) or math.isnan(predicted):
                continue
            actual_values.append(actual)
            predicted_values.append(predicted)
        if not actual_values:
            continue
        metrics = regression_metrics(actual_values, predicted_values)
        metrics["coverage"] = float(len(actual_values) / total)
        baselines[column] = metrics

    return baselines
