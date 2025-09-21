from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .utils import Dataset, to_float


def _clean_residuals(values: Iterable[float]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        numeric = to_float(value)
        if not math.isnan(numeric):
            cleaned.append(numeric)
    return cleaned


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _median_absolute_deviation(values: List[float]) -> float:
    if not values:
        return 0.0
    median = _quantile(values, 0.5)
    deviations = [abs(value - median) for value in values]
    return _quantile(deviations, 0.5)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


@dataclass
class ResidualIntervalEstimator:
    """Estimate prediction intervals and risk bands from residual distributions."""

    confidence_levels: Sequence[float] = (0.8, 0.95)
    group_column: str = "position"

    _global_stats: Dict[str, object] = field(default_factory=dict, init=False)
    _group_stats: Dict[object, Dict[str, object]] = field(default_factory=dict, init=False)
    _error_thresholds: Dict[str, float] = field(default_factory=dict, init=False)

    def fit(
        self,
        records: Dataset,
        *,
        actual_column: str = "actual_points",
        predicted_column: str = "predicted_points",
    ) -> "ResidualIntervalEstimator":
        if not records:
            self._global_stats = {}
            self._group_stats = {}
            self._error_thresholds = {}
            return self

        residuals = _clean_residuals(
            to_float(row.get(actual_column)) - to_float(row.get(predicted_column))
            for row in records
        )
        if not residuals:
            self._global_stats = {}
            self._group_stats = {}
            self._error_thresholds = {}
            return self

        self._global_stats = self._build_stats(residuals)
        self._error_thresholds = self._compute_error_thresholds(residuals)

        self._group_stats = {}
        if self.group_column:
            grouped: Dict[object, List[float]] = {}
            for row in records:
                if self.group_column not in row:
                    continue
                actual = to_float(row.get(actual_column))
                predicted = to_float(row.get(predicted_column))
                if math.isnan(actual) or math.isnan(predicted):
                    continue
                group_value = row[self.group_column]
                grouped.setdefault(group_value, []).append(actual - predicted)
            for group_value, values in grouped.items():
                stats = self._build_stats(_clean_residuals(values))
                if stats:
                    self._group_stats[group_value] = stats

        return self

    def describe(self) -> Dict[str, object]:
        description: Dict[str, object] = {}
        if self._global_stats:
            description["confidence_levels"] = [float(level) for level in self.confidence_levels]
            description["global"] = self._global_stats
        if self._group_stats:
            description["groups"] = {
                self._normalise_key(key): value for key, value in self._group_stats.items()
            }
        if self._error_thresholds:
            description["error_thresholds"] = {
                name: float(value) for name, value in self._error_thresholds.items()
            }
        return description

    def apply(self, records: Dataset, *, predicted_column: str = "predicted_points") -> Dataset:
        if not records or not self._global_stats:
            return records

        level_keys = self._level_keys()
        if not self._global_stats.get("levels"):
            return records

        result: Dataset = [dict(row) for row in records]
        group_values = [row.get(self.group_column) for row in result]

        lower_bounds: Dict[int, List[float]] = {}
        upper_bounds: Dict[int, List[float]] = {}
        for level_key in level_keys:
            lower, upper = self._bounds_for_group(level_key, group_values)
            lower_bounds[level_key] = lower
            upper_bounds[level_key] = upper
            lower_col = f"{predicted_column}_lower_p{level_key}"
            upper_col = f"{predicted_column}_upper_p{level_key}"
            for row, lower_offset, upper_offset in zip(result, lower, upper):
                predicted = to_float(row.get(predicted_column))
                if math.isnan(predicted):
                    row[lower_col] = float("nan")
                    row[upper_col] = float("nan")
                else:
                    row[lower_col] = predicted + lower_offset
                    row[upper_col] = predicted + upper_offset

        highest_key = level_keys[-1]
        expected_errors: List[float] = []
        for lower, upper in zip(lower_bounds[highest_key], upper_bounds[highest_key]):
            if math.isnan(lower) or math.isnan(upper):
                expected_errors.append(float("nan"))
            else:
                expected_errors.append((upper - lower) / 2.0)
        error_col = f"{predicted_column}_expected_error_p{highest_key}"
        for row, value in zip(result, expected_errors):
            row[error_col] = value

        if self._error_thresholds:
            classifications = self._classify_risk(expected_errors)
            for row, risk in zip(result, classifications):
                row["risk_band"] = risk

        return result

    def _level_keys(self) -> List[int]:
        return sorted({int(round(level * 100)) for level in self.confidence_levels})

    def _bounds_for_group(
        self,
        level_key: int,
        group_values: List[object],
    ) -> tuple[List[float], List[float]]:
        levels: Mapping[int, Mapping[str, float]] = self._global_stats.get("levels", {})  # type: ignore[arg-type]
        global_bounds = levels.get(level_key)
        if global_bounds is None:
            size = len(group_values)
            return [float("nan")] * size, [float("nan")] * size

        lower_default = float(global_bounds["lower"])
        upper_default = float(global_bounds["upper"])

        lower: List[float] = []
        upper: List[float] = []
        for group in group_values:
            stats = self._group_stats.get(group)
            if stats:
                group_levels = stats.get("levels", {})
                if isinstance(group_levels, Mapping) and level_key in group_levels:
                    bounds = group_levels[level_key]
                    lower.append(float(bounds["lower"]))
                    upper.append(float(bounds["upper"]))
                    continue
            lower.append(lower_default)
            upper.append(upper_default)

        return lower, upper

    def _classify_risk(self, expected_errors: List[float]) -> List[str]:
        low = self._error_thresholds.get("low")
        medium = self._error_thresholds.get("medium")
        high = self._error_thresholds.get("high")
        classifications: List[str] = []
        for value in expected_errors:
            if math.isnan(value):
                classifications.append("unknown")
            elif low is not None and value <= low:
                classifications.append("low")
            elif medium is not None and value <= medium:
                classifications.append("medium")
            elif high is not None and value <= high:
                classifications.append("high")
            else:
                classifications.append("high")
        return classifications

    def _build_stats(self, residuals: List[float]) -> Dict[str, object]:
        if not residuals:
            return {}

        stats: Dict[str, object] = {
            "count": len(residuals),
            "mean": _mean(residuals),
            "std": _stdev(residuals),
            "mad": _median_absolute_deviation(residuals),
        }

        levels: Dict[int, Dict[str, float]] = {}
        for level_key in self._level_keys():
            lower_prob = max(min((1 - level_key / 100.0) / 2.0, 0.5), 0.0)
            upper_prob = 1 - lower_prob
            lower = _quantile(residuals, lower_prob)
            upper = _quantile(residuals, upper_prob)
            levels[level_key] = {
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
            }
        stats["levels"] = levels

        abs_residuals = [abs(value) for value in residuals]
        stats["absolute_error_quantiles"] = {
            "p50": _quantile(abs_residuals, 0.5),
            "p80": _quantile(abs_residuals, 0.8),
            "p95": _quantile(abs_residuals, 0.95),
        }

        return stats

    def _compute_error_thresholds(self, residuals: List[float]) -> Dict[str, float]:
        abs_residuals = [abs(value) for value in residuals]
        if not abs_residuals:
            return {}
        return {
            "low": _quantile(abs_residuals, 0.33),
            "medium": _quantile(abs_residuals, 0.66),
            "high": _quantile(abs_residuals, 0.9),
        }

    @staticmethod
    def _normalise_key(value: object) -> object:
        return value
