"""Performance monitoring system for ML models with statistical process control."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config.monitoring_config import (
    MonitoringConfig,
    PerformanceThresholds,
    AlertLevel,
)


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""

    timestamp: datetime
    """When the metrics were recorded."""

    mae: float
    """Mean Absolute Error."""

    rmse: float
    """Root Mean Square Error."""

    r2: float
    """R² coefficient."""

    mape: Optional[float] = None
    """Mean Absolute Percentage Error."""

    sample_size: int = 0
    """Number of samples used for calculation."""

    feature_count: int = 0
    """Number of features used."""

    model_version: Optional[str] = None
    """Model version identifier."""


@dataclass
class PerformanceAlert:
    """Container for performance alerts."""

    timestamp: datetime
    """When the alert was triggered."""

    alert_type: str
    """Type of alert (e.g., 'mae_threshold', 'performance_degradation')."""

    level: AlertLevel
    """Alert severity level."""

    current_value: float
    """Current metric value."""

    threshold_value: float
    """Threshold value that was breached."""

    message: str
    """Human-readable alert message."""

    metadata: Dict[str, Any]
    """Additional metadata for the alert."""


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""

    metric: str
    """Metric being analyzed."""

    slope: float
    """Trend slope (positive = improving, negative = degrading)."""

    r_squared: float
    """R² value for the trend line."""

    significance: float
    """Statistical significance of the trend (p-value)."""

    forecast: List[float]
    """Forecasted values for the next period."""

    confidence_interval: Tuple[float, float]
    """Confidence interval for the forecast."""


class PerformanceMonitor:
    """Monitors ML model performance with statistical process control."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the performance monitor.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize storage
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[PerformanceAlert] = []

        # Create directories
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing performance history
        self._load_performance_history()

    def record_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_version: Optional[str] = None,
        feature_count: Optional[int] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> PerformanceMetrics:
        """Record model performance metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            model_version: Model version identifier.
            feature_count: Number of features used.
            additional_metrics: Additional metrics to record.

        Returns:
            PerformanceMetrics object with calculated metrics.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            raise ValueError("Cannot calculate metrics for empty arrays")

        # Calculate standard metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE if possible
        mape = None
        try:
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        except (ZeroDivisionError, RuntimeWarning):
            pass

        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            sample_size=len(y_true),
            feature_count=feature_count or 0,
            model_version=model_version,
        )

        # Store metrics
        self.performance_history.append(metrics)

        # Keep only recent history based on data window
        cutoff_date = datetime.now() - timedelta(days=self.config.data_window_days)
        self.performance_history = [
            m for m in self.performance_history
            if m.timestamp >= cutoff_date
        ]

        # Check for alerts
        self._check_performance_alerts(metrics)

        # Save to disk
        self._save_performance_history()

        self.logger.info(
            f"Recorded performance metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
        )

        return metrics

    def _check_performance_alerts(self, current_metrics: PerformanceMetrics) -> None:
        """Check for performance threshold breaches and create alerts.

        Args:
            current_metrics: Current performance metrics to check.
        """
        thresholds = self.config.performance_thresholds

        # Get recent performance history for trend analysis
        recent_metrics = self._get_recent_metrics(hours=24)

        # Check MAE thresholds
        if current_metrics.mae > thresholds.mae_critical:
            self._create_alert(
                alert_type="mae_threshold",
                level=AlertLevel.CRITICAL,
                current_value=current_metrics.mae,
                threshold_value=thresholds.mae_critical,
                message=f"MAE ({current_metrics.mae:.4f}) exceeded critical threshold ({thresholds.mae_critical:.4f})",
                metadata={"metric": "mae", "model_version": current_metrics.model_version}
            )
        elif current_metrics.mae > thresholds.mae_warning:
            self._create_alert(
                alert_type="mae_threshold",
                level=AlertLevel.WARNING,
                current_value=current_metrics.mae,
                threshold_value=thresholds.mae_warning,
                message=f"MAE ({current_metrics.mae:.4f}) exceeded warning threshold ({thresholds.mae_warning:.4f})",
                metadata={"metric": "mae", "model_version": current_metrics.model_version}
            )

        # Check RMSE thresholds
        if current_metrics.rmse > thresholds.rmse_critical:
            self._create_alert(
                alert_type="rmse_threshold",
                level=AlertLevel.CRITICAL,
                current_value=current_metrics.rmse,
                threshold_value=thresholds.rmse_critical,
                message=f"RMSE ({current_metrics.rmse:.4f}) exceeded critical threshold ({thresholds.rmse_critical:.4f})",
                metadata={"metric": "rmse", "model_version": current_metrics.model_version}
            )
        elif current_metrics.rmse > thresholds.rmse_warning:
            self._create_alert(
                alert_type="rmse_threshold",
                level=AlertLevel.WARNING,
                current_value=current_metrics.rmse,
                threshold_value=thresholds.rmse_warning,
                message=f"RMSE ({current_metrics.rmse:.4f}) exceeded warning threshold ({thresholds.rmse_warning:.4f})",
                metadata={"metric": "rmse", "model_version": current_metrics.model_version}
            )

        # Check R² thresholds
        if current_metrics.r2 < thresholds.r2_critical:
            self._create_alert(
                alert_type="r2_threshold",
                level=AlertLevel.CRITICAL,
                current_value=current_metrics.r2,
                threshold_value=thresholds.r2_critical,
                message=f"R² ({current_metrics.r2:.4f}) below critical threshold ({thresholds.r2_critical:.4f})",
                metadata={"metric": "r2", "model_version": current_metrics.model_version}
            )
        elif current_metrics.r2 < thresholds.r2_warning:
            self._create_alert(
                alert_type="r2_threshold",
                level=AlertLevel.WARNING,
                current_value=current_metrics.r2,
                threshold_value=thresholds.r2_warning,
                message=f"R² ({current_metrics.r2:.4f}) below warning threshold ({thresholds.r2_warning:.4f})",
                metadata={"metric": "r2", "model_version": current_metrics.model_version}
            )

        # Check for performance degradation trends
        if self.config.enable_trend_analysis and len(recent_metrics) >= 5:
            for metric_name in ['mae', 'rmse']:
                trend = self._analyze_trend(metric_name, recent_metrics)
                if trend and trend.significance < 0.05:  # Statistically significant trend
                    degradation_rate = abs(trend.slope) / np.mean([getattr(m, metric_name) for m in recent_metrics])
                    if (degradation_rate > thresholds.performance_degradation_rate and
                        trend.slope > 0):  # Degrading trend
                        self._create_alert(
                            alert_type="performance_degradation",
                            level=AlertLevel.WARNING,
                            current_value=getattr(current_metrics, metric_name),
                            threshold_value=thresholds.performance_degradation_rate,
                            message=f"Performance degrading: {metric_name.upper()} increased by {degradation_rate:.1%} "
                                   f"over recent period (slope: {trend.slope:.6f})",
                            metadata={
                                "metric": metric_name,
                                "degradation_rate": degradation_rate,
                                "slope": trend.slope,
                                "significance": trend.significance,
                                "model_version": current_metrics.model_version
                            }
                        )

    def _create_alert(
        self,
        alert_type: str,
        level: AlertLevel,
        current_value: float,
        threshold_value: float,
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Create a performance alert.

        Args:
            alert_type: Type of alert.
            level: Alert severity level.
            current_value: Current metric value.
            threshold_value: Threshold value.
            message: Alert message.
            metadata: Additional metadata.
        """
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            level=level,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            metadata=metadata
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        cutoff_date = datetime.now() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]

        self.logger.warning(f"Performance alert: {message}")

    def _analyze_trend(
        self,
        metric_name: str,
        metrics: List[PerformanceMetrics]
    ) -> Optional[TrendAnalysis]:
        """Analyze trend in performance metrics.

        Args:
            metric_name: Name of metric to analyze.
            metrics: List of performance metrics.

        Returns:
            TrendAnalysis object or None if insufficient data.
        """
        if len(metrics) < 3:
            return None

        # Extract metric values and timestamps
        timestamps = [(m.timestamp - metrics[0].timestamp).total_seconds() / 3600 for m in metrics]
        values = [getattr(m, metric_name) for m in metrics]

        # Perform linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)

            # Generate forecast
            forecast_horizon = self.config.forecasting_horizon
            future_timestamps = [timestamps[-1] + i * 24 for i in range(1, forecast_horizon + 1)]
            forecast_values = [slope * ts + intercept for ts in future_timestamps]

            # Calculate confidence interval
            confidence_interval = (forecast_values[-1] - 1.96 * std_err, forecast_values[-1] + 1.96 * std_err)

            return TrendAnalysis(
                metric=metric_name,
                slope=slope,
                r_squared=r_value ** 2,
                significance=p_value,
                forecast=forecast_values,
                confidence_interval=confidence_interval
            )
        except Exception as e:
            self.logger.warning(f"Trend analysis failed for {metric_name}: {e}")
            return None

    def _get_recent_metrics(
        self,
        hours: int = 24,
        min_samples: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """Get recent performance metrics.

        Args:
            hours: Number of hours to look back.
            min_samples: Minimum number of samples required.

        Returns:
            List of recent performance metrics.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]

        min_samples = min_samples or self.config.min_samples_for_analysis
        if len(recent_metrics) < min_samples:
            return []

        return recent_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with performance summary statistics.
        """
        if not self.performance_history:
            return {"error": "No performance data available"}

        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
                "mape": m.mape,
                "sample_size": m.sample_size,
                "model_version": m.model_version
            }
            for m in self.performance_history
        ])

        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat()
            },
            "current_metrics": {
                "mae": df["mae"].iloc[-1],
                "rmse": df["rmse"].iloc[-1],
                "r2": df["r2"].iloc[-1],
                "mape": df["mape"].iloc[-1] if df["mape"].iloc[-1] is not None else None
            },
            "mean_metrics": {
                "mae": df["mae"].mean(),
                "rmse": df["rmse"].mean(),
                "r2": df["r2"].mean(),
                "mape": df["mape"].mean() if df["mape"].notna().any() else None
            },
            "std_metrics": {
                "mae": df["mae"].std(),
                "rmse": df["rmse"].std(),
                "r2": df["r2"].std(),
                "mape": df["mape"].std() if df["mape"].notna().any() else None
            },
            "trend_analysis": {},
            "alerts_summary": self._get_alerts_summary()
        }

        # Add trend analysis for each metric
        for metric in ['mae', 'rmse', 'r2']:
            trend = self._analyze_trend(metric, self.performance_history)
            if trend:
                summary["trend_analysis"][metric] = {
                    "slope": trend.slope,
                    "r_squared": trend.r_squared,
                    "significance": trend.significance,
                    "forecast": trend.forecast[-1] if trend.forecast else None
                }

        return summary

    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts.

        Returns:
            Dictionary with alerts summary.
        """
        if not self.alerts:
            return {"total_alerts": 0, "recent_alerts": []}

        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).days <= 1]

        return {
            "total_alerts": len(self.alerts),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            "warning_alerts": len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
            "info_alerts": len([a for a in self.alerts if a.level == AlertLevel.INFO]),
            "latest_alert": {
                "type": self.alerts[-1].alert_type,
                "level": self.alerts[-1].level.value,
                "message": self.alerts[-1].message,
                "timestamp": self.alerts[-1].timestamp.isoformat()
            } if self.alerts else None
        }

    def _load_performance_history(self) -> None:
        """Load performance history from disk."""
        history_file = self.config.artifacts_dir / "performance_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                for item in data:
                    metrics = PerformanceMetrics(
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        mae=item["mae"],
                        rmse=item["rmse"],
                        r2=item["r2"],
                        mape=item.get("mape"),
                        sample_size=item.get("sample_size", 0),
                        feature_count=item.get("feature_count", 0),
                        model_version=item.get("model_version")
                    )
                    self.performance_history.append(metrics)

                self.logger.info(f"Loaded {len(self.performance_history)} performance records from disk")

            except Exception as e:
                self.logger.error(f"Failed to load performance history: {e}")

    def _save_performance_history(self) -> None:
        """Save performance history to disk."""
        history_file = self.config.artifacts_dir / "performance_history.json"

        try:
            data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "r2": m.r2,
                    "mape": m.mape,
                    "sample_size": m.sample_size,
                    "feature_count": m.feature_count,
                    "model_version": m.model_version
                }
                for m in self.performance_history
            ]

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")

    def export_metrics(self, output_path: Optional[Path] = None) -> str:
        """Export performance metrics to CSV.

        Args:
            output_path: Path to save CSV file. If None, uses default location.

        Returns:
            Path to exported CSV file.
        """
        if not output_path:
            output_path = self.config.artifacts_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
                "mape": m.mape,
                "sample_size": m.sample_size,
                "feature_count": m.feature_count,
                "model_version": m.model_version
            }
            for m in self.performance_history
        ])

        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported performance metrics to {output_path}")
        return str(output_path)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the model.

        Returns:
            Dictionary with health status information.
        """
        if not self.performance_history:
            return {"status": "unknown", "message": "No performance data available"}

        recent_metrics = self._get_recent_metrics(hours=24)
        if not recent_metrics:
            return {"status": "unknown", "message": "Insufficient recent data"}

        latest = recent_metrics[-1]
        thresholds = self.config.performance_thresholds

        # Determine health status
        if (latest.mae > thresholds.mae_critical or
            latest.rmse > thresholds.rmse_critical or
            latest.r2 < thresholds.r2_critical):
            status = "critical"
        elif (latest.mae > thresholds.mae_warning or
              latest.rmse > thresholds.rmse_warning or
              latest.r2 < thresholds.r2_warning):
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "latest_metrics": {
                "mae": latest.mae,
                "rmse": latest.rmse,
                "r2": latest.r2,
                "timestamp": latest.timestamp.isoformat()
            },
            "alerts_count": len(self._get_alerts_summary()["recent_alerts"]),
            "trend_direction": "degrading" if status in ["warning", "critical"] else "stable"
        }
