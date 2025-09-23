"""Data drift detection system for monitoring feature distribution changes."""

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

from ..config.monitoring_config import (
    MonitoringConfig,
    DriftThresholds,
    AlertLevel,
)


@dataclass
class DriftMetrics:
    """Container for feature drift metrics."""

    feature_name: str
    """Name of the feature."""

    psi: float
    """Population Stability Index."""

    ks_statistic: float
    """Kolmogorov-Smirnov statistic."""

    ks_pvalue: float
    """Kolmogorov-Smirnov p-value."""

    jensen_shannon_divergence: float
    """Jensen-Shannon divergence."""

    wasserstein_distance: float
    """Wasserstein distance."""

    timestamp: datetime
    """When the drift was calculated."""

    reference_mean: float
    """Mean of reference distribution."""

    current_mean: float
    """Mean of current distribution."""

    reference_std: float
    """Standard deviation of reference distribution."""

    current_std: float
    """Standard deviation of current distribution."""


@dataclass
class DriftAlert:
    """Container for drift alerts."""

    timestamp: datetime
    """When the alert was triggered."""

    feature_name: str
    """Name of the feature that drifted."""

    alert_type: str
    """Type of alert (e.g., 'psi_threshold', 'ks_threshold')."""

    level: AlertLevel
    """Alert severity level."""

    current_value: float
    """Current drift metric value."""

    threshold_value: float
    """Threshold value that was breached."""

    message: str
    """Human-readable alert message."""

    metadata: Dict[str, Any]
    """Additional metadata for the alert."""


class DataDriftDetector:
    """Detects data drift using multiple statistical methods."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the data drift detector.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize storage
        self.drift_history: Dict[str, List[DriftMetrics]] = {}
        self.alerts: List[DriftAlert] = []

        # Create directories
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing drift history
        self._load_drift_history()

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_list: Optional[List[str]] = None
    ) -> Dict[str, DriftMetrics]:
        """Detect drift between reference and current datasets.

        Args:
            reference_data: Reference dataset (training data).
            current_data: Current dataset (production data).
            feature_list: List of features to analyze. If None, uses configured features.

        Returns:
            Dictionary mapping feature names to drift metrics.
        """
        if reference_data.empty or current_data.empty:
            raise ValueError("Reference and current data cannot be empty")

        # Use configured features or provided list
        features_to_check = feature_list or self.config.monitored_features

        # Filter out excluded features
        features_to_check = [
            f for f in features_to_check
            if f not in (self.config.exclude_features or [])
            and f in reference_data.columns
            and f in current_data.columns
        ]

        if not features_to_check:
            raise ValueError("No valid features to check for drift")

        drift_results = {}

        for feature in features_to_check:
            try:
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()

                if len(ref_values) < 10 or len(curr_values) < 10:
                    self.logger.warning(f"Insufficient data for feature {feature}")
                    continue

                drift_metrics = self._calculate_drift_metrics(
                    feature, ref_values, curr_values
                )

                drift_results[feature] = drift_metrics
                self._store_drift_metrics(drift_metrics)

                # Check for drift alerts
                self._check_drift_alerts(drift_metrics)

            except Exception as e:
                self.logger.error(f"Error calculating drift for feature {feature}: {e}")
                continue

        # Save drift history
        self._save_drift_history()

        self.logger.info(f"Calculated drift metrics for {len(drift_results)} features")
        return drift_results

    def _calculate_drift_metrics(
        self,
        feature_name: str,
        reference_values: pd.Series,
        current_values: pd.Series
    ) -> DriftMetrics:
        """Calculate comprehensive drift metrics for a feature.

        Args:
            feature_name: Name of the feature.
            reference_values: Reference feature values.
            current_values: Current feature values.

        Returns:
            DriftMetrics object with calculated metrics.
        """
        # Population Stability Index (PSI)
        psi = self._calculate_psi(reference_values, current_values)

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_values, current_values)

        # Jensen-Shannon divergence (for continuous variables)
        js_divergence = self._calculate_js_divergence(reference_values, current_values)

        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(reference_values, current_values)

        # Basic statistics
        ref_mean = reference_values.mean()
        curr_mean = current_values.mean()
        ref_std = reference_values.std()
        curr_std = current_values.std()

        return DriftMetrics(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            jensen_shannon_divergence=js_divergence,
            wasserstein_distance=wasserstein_dist,
            timestamp=datetime.now(),
            reference_mean=ref_mean,
            current_mean=curr_mean,
            reference_std=ref_std,
            current_std=curr_std
        )

    def _calculate_psi(
        self,
        reference_values: pd.Series,
        current_values: pd.Series,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index.

        Args:
            reference_values: Reference feature values.
            current_values: Current feature values.
            bins: Number of bins for histogram.

        Returns:
            PSI value.
        """
        # Create histograms
        ref_hist, _ = np.histogram(reference_values, bins=bins)
        curr_hist, _ = np.histogram(current_values, bins=bins)

        # Add small epsilon to avoid division by zero
        eps = 1e-10
        ref_hist = ref_hist + eps
        curr_hist = curr_hist + eps

        # Normalize histograms
        ref_dist = ref_hist / len(reference_values)
        curr_dist = curr_hist / len(current_values)

        # Calculate PSI
        psi = np.sum((ref_dist - curr_dist) * np.log(ref_dist / curr_dist))
        return psi

    def _calculate_js_divergence(
        self,
        reference_values: pd.Series,
        current_values: pd.Series,
        bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon divergence.

        Args:
            reference_values: Reference feature values.
            current_values: Current feature values.
            bins: Number of bins for histogram.

        Returns:
            Jensen-Shannon divergence value.
        """
        # Create histograms
        ref_hist, bin_edges = np.histogram(reference_values, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_values, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ref_hist = ref_hist + eps
        curr_hist = curr_hist + eps

        # Normalize
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)

        # Calculate Jensen-Shannon divergence
        m = 0.5 * (ref_hist + curr_hist)
        js_divergence = 0.5 * (
            np.sum(ref_hist * np.log(ref_hist / m)) +
            np.sum(curr_hist * np.log(curr_hist / m))
        )

        return js_divergence

    def _check_drift_alerts(self, drift_metrics: DriftMetrics) -> None:
        """Check for drift threshold breaches and create alerts.

        Args:
            drift_metrics: Drift metrics to check.
        """
        thresholds = self.config.drift_thresholds

        # Check PSI thresholds
        if drift_metrics.psi > thresholds.psi_critical:
            self._create_drift_alert(
                feature_name=drift_metrics.feature_name,
                alert_type="psi_threshold",
                level=AlertLevel.CRITICAL,
                current_value=drift_metrics.psi,
                threshold_value=thresholds.psi_critical,
                message=f"Feature '{drift_metrics.feature_name}' PSI ({drift_metrics.psi:.4f}) "
                       f"exceeded critical threshold ({thresholds.psi_critical:.4f})",
                metadata={"metric": "psi", "reference_mean": drift_metrics.reference_mean}
            )
        elif drift_metrics.psi > thresholds.psi_warning:
            self._create_drift_alert(
                feature_name=drift_metrics.feature_name,
                alert_type="psi_threshold",
                level=AlertLevel.WARNING,
                current_value=drift_metrics.psi,
                threshold_value=thresholds.psi_warning,
                message=f"Feature '{drift_metrics.feature_name}' PSI ({drift_metrics.psi:.4f}) "
                       f"exceeded warning threshold ({thresholds.psi_warning:.4f})",
                metadata={"metric": "psi", "reference_mean": drift_metrics.reference_mean}
            )

        # Check KS statistic thresholds
        if drift_metrics.ks_statistic > thresholds.ks_statistic_critical:
            self._create_drift_alert(
                feature_name=drift_metrics.feature_name,
                alert_type="ks_threshold",
                level=AlertLevel.CRITICAL,
                current_value=drift_metrics.ks_statistic,
                threshold_value=thresholds.ks_statistic_critical,
                message=f"Feature '{drift_metrics.feature_name}' KS statistic ({drift_metrics.ks_statistic:.4f}) "
                       f"exceeded critical threshold ({thresholds.ks_statistic_critical:.4f})",
                metadata={"metric": "ks_statistic", "ks_pvalue": drift_metrics.ks_pvalue}
            )
        elif drift_metrics.ks_statistic > thresholds.ks_statistic_warning:
            self._create_drift_alert(
                feature_name=drift_metrics.feature_name,
                alert_type="ks_threshold",
                level=AlertLevel.WARNING,
                current_value=drift_metrics.ks_statistic,
                threshold_value=thresholds.ks_statistic_warning,
                message=f"Feature '{drift_metrics.feature_name}' KS statistic ({drift_metrics.ks_statistic:.4f}) "
                       f"exceeded warning threshold ({thresholds.ks_statistic_warning:.4f})",
                metadata={"metric": "ks_statistic", "ks_pvalue": drift_metrics.ks_pvalue}
            )

    def _create_drift_alert(
        self,
        feature_name: str,
        alert_type: str,
        level: AlertLevel,
        current_value: float,
        threshold_value: float,
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Create a drift alert.

        Args:
            feature_name: Name of the drifted feature.
            alert_type: Type of alert.
            level: Alert severity level.
            current_value: Current drift metric value.
            threshold_value: Threshold value.
            message: Alert message.
            metadata: Additional metadata.
        """
        alert = DriftAlert(
            timestamp=datetime.now(),
            feature_name=feature_name,
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

        self.logger.warning(f"Drift alert: {message}")

    def _store_drift_metrics(self, drift_metrics: DriftMetrics) -> None:
        """Store drift metrics in history.

        Args:
            drift_metrics: Drift metrics to store.
        """
        if drift_metrics.feature_name not in self.drift_history:
            self.drift_history[drift_metrics.feature_name] = []

        self.drift_history[drift_metrics.feature_name].append(drift_metrics)

        # Keep only recent history based on data window
        cutoff_date = datetime.now() - timedelta(days=self.config.data_window_days)
        self.drift_history[drift_metrics.feature_name] = [
            m for m in self.drift_history[drift_metrics.feature_name]
            if m.timestamp >= cutoff_date
        ]

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get comprehensive drift summary.

        Returns:
            Dictionary with drift summary statistics.
        """
        if not self.drift_history:
            return {"error": "No drift data available"}

        summary = {
            "total_features_tracked": len(self.drift_history),
            "total_drift_calculations": sum(len(metrics) for metrics in self.drift_history.values()),
            "features_with_drift": {},
            "alerts_summary": self._get_alerts_summary(),
            "feature_summary": {}
        }

        # Analyze each feature
        for feature_name, metrics_list in self.drift_history.items():
            if not metrics_list:
                continue

            latest_metrics = metrics_list[-1]

            # Check if feature has significant drift
            has_drift = (
                latest_metrics.psi > self.config.drift_thresholds.psi_warning or
                latest_metrics.ks_statistic > self.config.drift_thresholds.ks_statistic_warning
            )

            if has_drift:
                summary["features_with_drift"][feature_name] = {
                    "psi": latest_metrics.psi,
                    "ks_statistic": latest_metrics.ks_statistic,
                    "jensen_shannon_divergence": latest_metrics.jensen_shannon_divergence,
                    "mean_change": latest_metrics.current_mean - latest_metrics.reference_mean,
                    "std_change": latest_metrics.current_std - latest_metrics.reference_std,
                    "last_updated": latest_metrics.timestamp.isoformat()
                }

            # Feature summary statistics
            summary["feature_summary"][feature_name] = {
                "total_measurements": len(metrics_list),
                "date_range": {
                    "start": min(m.timestamp for m in metrics_list).isoformat(),
                    "end": max(m.timestamp for m in metrics_list).isoformat()
                },
                "current_metrics": {
                    "psi": latest_metrics.psi,
                    "ks_statistic": latest_metrics.ks_statistic,
                    "jensen_shannon_divergence": latest_metrics.jensen_shannon_divergence
                },
                "mean_metrics": {
                    "psi": np.mean([m.psi for m in metrics_list]),
                    "ks_statistic": np.mean([m.ks_statistic for m in metrics_list]),
                    "jensen_shannon_divergence": np.mean([m.jensen_shannon_divergence for m in metrics_list])
                }
            }

        return summary

    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift alerts.

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
                "feature": self.alerts[-1].feature_name,
                "type": self.alerts[-1].alert_type,
                "level": self.alerts[-1].level.value,
                "message": self.alerts[-1].message,
                "timestamp": self.alerts[-1].timestamp.isoformat()
            } if self.alerts else None
        }

    def _load_drift_history(self) -> None:
        """Load drift history from disk."""
        history_file = self.config.artifacts_dir / "drift_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                for feature_name, feature_data in data.items():
                    metrics_list = []
                    for item in feature_data:
                        metrics = DriftMetrics(
                            feature_name=feature_name,
                            psi=item["psi"],
                            ks_statistic=item["ks_statistic"],
                            ks_pvalue=item["ks_pvalue"],
                            jensen_shannon_divergence=item["jensen_shannon_divergence"],
                            wasserstein_distance=item["wasserstein_distance"],
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            reference_mean=item["reference_mean"],
                            current_mean=item["current_mean"],
                            reference_std=item["reference_std"],
                            current_std=item["current_std"]
                        )
                        metrics_list.append(metrics)
                    self.drift_history[feature_name] = metrics_list

                self.logger.info(f"Loaded drift history for {len(self.drift_history)} features")

            except Exception as e:
                self.logger.error(f"Failed to load drift history: {e}")

    def _save_drift_history(self) -> None:
        """Save drift history to disk."""
        history_file = self.config.artifacts_dir / "drift_history.json"

        try:
            data = {}
            for feature_name, metrics_list in self.drift_history.items():
                data[feature_name] = [
                    {
                        "psi": m.psi,
                        "ks_statistic": m.ks_statistic,
                        "ks_pvalue": m.ks_pvalue,
                        "jensen_shannon_divergence": m.jensen_shannon_divergence,
                        "wasserstein_distance": m.wasserstein_distance,
                        "timestamp": m.timestamp.isoformat(),
                        "reference_mean": m.reference_mean,
                        "current_mean": m.current_mean,
                        "reference_std": m.reference_std,
                        "current_std": m.current_std
                    }
                    for m in metrics_list
                ]

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save drift history: {e}")

    def export_drift_report(self, output_path: Optional[Path] = None) -> str:
        """Export drift report to CSV.

        Args:
            output_path: Path to save CSV file. If None, uses default location.

        Returns:
            Path to exported CSV file.
        """
        if not output_path:
            output_path = self.config.artifacts_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        rows = []
        for feature_name, metrics_list in self.drift_history.items():
            for metrics in metrics_list:
                rows.append({
                    "feature_name": metrics.feature_name,
                    "timestamp": metrics.timestamp,
                    "psi": metrics.psi,
                    "ks_statistic": metrics.ks_statistic,
                    "ks_pvalue": metrics.ks_pvalue,
                    "jensen_shannon_divergence": metrics.jensen_shannon_divergence,
                    "wasserstein_distance": metrics.wasserstein_distance,
                    "reference_mean": metrics.reference_mean,
                    "current_mean": metrics.current_mean,
                    "reference_std": metrics.reference_std,
                    "current_std": metrics.current_std
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported drift report to {output_path}")
        else:
            # Create empty DataFrame with correct columns
            df = pd.DataFrame(columns=[
                "feature_name", "timestamp", "psi", "ks_statistic", "ks_pvalue",
                "jensen_shannon_divergence", "wasserstein_distance",
                "reference_mean", "current_mean", "reference_std", "current_std"
            ])
            df.to_csv(output_path, index=False)

        return str(output_path)

    def get_drift_health_status(self) -> Dict[str, Any]:
        """Get overall drift health status.

        Returns:
            Dictionary with drift health status information.
        """
        if not self.drift_history:
            return {"status": "unknown", "message": "No drift data available"}

        # Count features with significant drift
        drifted_features = 0
        total_features = len(self.drift_history)

        for feature_name, metrics_list in self.drift_history.items():
            if metrics_list:
                latest = metrics_list[-1]
                if (latest.psi > self.config.drift_thresholds.psi_warning or
                    latest.ks_statistic > self.config.drift_thresholds.ks_statistic_warning):
                    drifted_features += 1

        # Calculate drift percentage
        drift_percentage = drifted_features / total_features if total_features > 0 else 0

        # Determine health status
        if drift_percentage > 0.5:  # More than 50% of features have drift
            status = "critical"
        elif drift_percentage > 0.2:  # More than 20% of features have drift
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "drift_percentage": drift_percentage,
            "drifted_features": drifted_features,
            "total_features": total_features,
            "alerts_count": len([a for a in self.alerts if a.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]])
        }
