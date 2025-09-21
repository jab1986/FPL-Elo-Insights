"""Uncertainty estimation utilities for points prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd


def _to_float(value: float | np.floating | np.ndarray) -> float:
    """Cast numpy scalar types to builtin ``float`` values."""

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        value = value.item()
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _as_series(values: Iterable[float]) -> pd.Series:
    """Convert iterable of numbers into a numeric Series without mutating caller data."""

    series = pd.Series(list(values), dtype="float64")
    return series.dropna()


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
        frame: pd.DataFrame,
        *,
        actual_column: str = "actual_points",
        predicted_column: str = "predicted_points",
    ) -> "ResidualIntervalEstimator":
        """Learn residual-based calibration statistics from a labelled dataset."""

        if frame.empty or actual_column not in frame.columns or predicted_column not in frame.columns:
            self._global_stats = {}
            self._group_stats = {}
            self._error_thresholds = {}
            return self

        residuals = _as_series(frame[actual_column] - frame[predicted_column])
        self._global_stats = self._build_stats(residuals)
        self._error_thresholds = self._compute_error_thresholds(residuals)

        self._group_stats = {}
        if self.group_column and self.group_column in frame.columns:
            for group_value, group_df in frame.groupby(self.group_column):
                stats = self._build_stats(
                    _as_series(group_df[actual_column] - group_df[predicted_column])
                )
                if stats:
                    self._group_stats[group_value] = stats

        return self

    def describe(self) -> Dict[str, object]:
        """Return a serialisable summary of the learned calibration statistics."""

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
                name: _to_float(value) for name, value in self._error_thresholds.items()
            }
        return description

    def apply(self, frame: pd.DataFrame, *, predicted_column: str = "predicted_points") -> pd.DataFrame:
        """Augment a prediction frame with calibrated intervals and risk bands."""

        if frame.empty or not self._global_stats:
            return frame

        result = frame.copy()
        level_keys = self._level_keys()
        if not self._global_stats.get("levels"):
            return result

        size = len(result)
        group_series = result[self.group_column] if self.group_column in result.columns else None

        for level_key in level_keys:
            bounds = self._bounds_for_group(level_key, group_series, size)
            lower_col = f"{predicted_column}_lower_p{level_key}"
            upper_col = f"{predicted_column}_upper_p{level_key}"
            result[lower_col] = result[predicted_column] + bounds["lower"]
            result[upper_col] = result[predicted_column] + bounds["upper"]

        highest_key = level_keys[-1]
        highest_bounds = self._bounds_for_group(highest_key, group_series, size)
        error_col = f"{predicted_column}_expected_error_p{highest_key}"
        result[error_col] = (highest_bounds["upper"] - highest_bounds["lower"]) / 2.0

        if self._error_thresholds:
            result["risk_band"] = self._classify_risk(result[error_col])

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _level_keys(self) -> List[int]:
        return sorted({int(round(level * 100)) for level in self.confidence_levels})

    def _bounds_for_group(
        self,
        level_key: int,
        group_series: Optional[pd.Series],
        size: int,
    ) -> Mapping[str, np.ndarray]:
        global_levels: Mapping[int, Mapping[str, float]] = self._global_stats.get("levels", {})
        global_bounds = global_levels.get(level_key)
        if global_bounds is None:
            lower = np.full(size, np.nan)
            upper = np.full(size, np.nan)
            return {"lower": lower, "upper": upper}

        lower_default = _to_float(global_bounds["lower"])
        upper_default = _to_float(global_bounds["upper"])

        if group_series is None or group_series.empty or not self._group_stats:
            lower = np.full(size, lower_default)
            upper = np.full(size, upper_default)
            return {"lower": lower, "upper": upper}

        lower_map: MutableMapping[object, float] = {}
        upper_map: MutableMapping[object, float] = {}
        for group_value, stats in self._group_stats.items():
            levels = stats.get("levels", {})
            if level_key not in levels:
                continue
            bounds = levels[level_key]
            lower_map[group_value] = _to_float(bounds["lower"])
            upper_map[group_value] = _to_float(bounds["upper"])

        lower_series = group_series.map(lower_map).fillna(lower_default)
        upper_series = group_series.map(upper_map).fillna(upper_default)
        return {"lower": lower_series.to_numpy(), "upper": upper_series.to_numpy()}

    def _classify_risk(self, expected_error: pd.Series) -> pd.Series:
        values = expected_error.fillna(expected_error.median())
        low = self._error_thresholds.get("low")
        medium = self._error_thresholds.get("medium")
        classifications: List[str] = []
        for value in values:
            if np.isnan(value):
                classifications.append("unknown")
            elif low is not None and value <= low:
                classifications.append("low")
            elif medium is not None and value <= medium:
                classifications.append("medium")
            else:
                classifications.append("high")
        return pd.Series(classifications, index=expected_error.index)

    def _build_stats(self, residuals: pd.Series) -> Dict[str, object]:
        if residuals.empty:
            return {}

        stats: Dict[str, object] = {
            "count": int(residuals.size),
            "mean": _to_float(residuals.mean()),
            "std": _to_float(residuals.std(ddof=1)) if residuals.size > 1 else 0.0,
            "mad": _to_float(np.median(np.abs(residuals - np.median(residuals)))),
        }

        levels: Dict[int, Dict[str, float]] = {}
        for level_key in self._level_keys():
            lower_prob = max(min((1 - level_key / 100.0) / 2.0, 0.5), 0.0)
            upper_prob = 1 - lower_prob
            lower = _to_float(residuals.quantile(lower_prob))
            upper = _to_float(residuals.quantile(upper_prob))
            levels[level_key] = {
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
            }
        stats["levels"] = levels

        abs_residuals = residuals.abs()
        stats["absolute_error_quantiles"] = {
            "p50": _to_float(abs_residuals.quantile(0.5)),
            "p80": _to_float(abs_residuals.quantile(0.8)),
            "p95": _to_float(abs_residuals.quantile(0.95)),
        }

        return stats

    def _compute_error_thresholds(self, residuals: pd.Series) -> Dict[str, float]:
        abs_residuals = residuals.abs()
        if abs_residuals.empty:
            return {}
        return {
            "low": _to_float(abs_residuals.quantile(0.33)),
            "medium": _to_float(abs_residuals.quantile(0.66)),
            "high": _to_float(abs_residuals.quantile(0.9)),
        }

    @staticmethod
    def _normalise_key(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

