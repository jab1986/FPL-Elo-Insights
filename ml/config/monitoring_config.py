"""Configuration for ML monitoring and automated retraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class AlertLevel(str, Enum):
    """Enumeration for alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Enumeration for notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    DASHBOARD = "dashboard"


@dataclass(frozen=True)
class PerformanceThresholds:
    """Configuration for performance monitoring thresholds."""

    mae_warning: float = 1.5
    """Mean Absolute Error threshold for warning alerts."""

    mae_critical: float = 2.0
    """Mean Absolute Error threshold for critical alerts."""

    rmse_warning: float = 2.5
    """Root Mean Square Error threshold for warning alerts."""

    rmse_critical: float = 3.0
    """Root Mean Square Error threshold for critical alerts."""

    r2_warning: float = 0.3
    """R² threshold for warning alerts (below this value)."""

    r2_critical: float = 0.1
    """R² threshold for critical alerts (below this value)."""

    performance_degradation_rate: float = 0.1
    """Rate of performance degradation to trigger alerts."""


@dataclass(frozen=True)
class DriftThresholds:
    """Configuration for data drift detection thresholds."""

    psi_warning: float = 0.1
    """Population Stability Index threshold for warning alerts."""

    psi_critical: float = 0.25
    """Population Stability Index threshold for critical alerts."""

    ks_statistic_warning: float = 0.1
    """Kolmogorov-Smirnov statistic threshold for warning alerts."""

    ks_statistic_critical: float = 0.2
    """Kolmogorov-Smirnov statistic threshold for critical alerts."""

    feature_drift_rate: float = 0.05
    """Rate of feature drift to trigger alerts."""


@dataclass(frozen=True)
class RetrainingConfig:
    """Configuration for automated retraining."""

    min_retrain_interval_hours: int = 24
    """Minimum hours between retraining attempts."""

    max_retrain_attempts: int = 3
    """Maximum number of consecutive retraining attempts."""

    performance_improvement_threshold: float = 0.05
    """Minimum performance improvement required for successful retraining."""

    enable_emergency_retraining: bool = True
    """Whether to enable emergency retraining for critical performance drops."""

    emergency_retraining_threshold: float = 0.2
    """Performance degradation threshold for emergency retraining."""

    enable_gradual_rollout: bool = True
    """Whether to enable gradual model rollout."""

    gradual_rollout_percentage: float = 0.1
    """Percentage of traffic for gradual rollout (0.1 = 10%)."""

    rollback_enabled: bool = True
    """Whether to enable automatic rollback on failures."""

    quality_gate_threshold: float = 0.95
    """Quality gate threshold for model acceptance."""


@dataclass(frozen=True)
class NotificationConfig:
    """Configuration for notification system."""

    enabled_channels: List[NotificationChannel] = None
    """Enabled notification channels."""

    email_recipients: List[str] = None
    """Email addresses for notifications."""

    slack_webhook_url: Optional[str] = None
    """Slack webhook URL for notifications."""

    discord_webhook_url: Optional[str] = None
    """Discord webhook URL for notifications."""

    dashboard_url: Optional[str] = None
    """Dashboard URL for alerts."""

    alert_cooldown_minutes: int = 15
    """Cooldown period between similar alerts."""

    critical_alerts_only: bool = False
    """Whether to send only critical alerts."""

    include_model_metrics: bool = True
    """Whether to include detailed model metrics in notifications."""

    include_charts: bool = False
    """Whether to include performance charts in notifications."""


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for retraining scheduler."""

    enabled: bool = True
    """Whether the scheduler is enabled."""

    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    """Cron expression for scheduled retraining."""

    enable_performance_based: bool = True
    """Whether to enable performance-based scheduling."""

    performance_check_interval_minutes: int = 60
    """Interval for performance monitoring checks."""

    max_concurrent_jobs: int = 1
    """Maximum number of concurrent retraining jobs."""

    job_timeout_hours: int = 6
    """Timeout for retraining jobs."""

    retry_failed_jobs: bool = True
    """Whether to retry failed jobs."""

    max_job_retries: int = 2
    """Maximum number of job retries."""


@dataclass(frozen=True)
class DatabaseConfig:
    """Configuration for database integration."""

    enabled: bool = True
    """Whether database integration is enabled."""

    connection_string: Optional[str] = None
    """Database connection string."""

    performance_table: str = "ml_performance_history"
    """Table name for performance history."""

    drift_table: str = "ml_drift_history"
    """Table name for drift history."""

    model_versions_table: str = "ml_model_versions"
    """Table name for model versions."""

    batch_size: int = 1000
    """Batch size for database operations."""

    connection_pool_size: int = 5
    """Database connection pool size."""


@dataclass(frozen=True)
class MonitoringConfig:
    """Main configuration for ML monitoring and retraining system."""

    # Core settings
    model_name: str = "fpl_points_predictor"
    """Name of the model being monitored."""

    environment: str = "development"
    """Environment name (development, staging, production)."""

    # Performance monitoring
    performance_thresholds: PerformanceThresholds = None
    """Performance monitoring thresholds."""

    # Data drift detection
    drift_thresholds: DriftThresholds = None
    """Data drift detection thresholds."""

    # Retraining configuration
    retraining: RetrainingConfig = None
    """Automated retraining configuration."""

    # Notification settings
    notifications: NotificationConfig = None
    """Notification system configuration."""

    # Scheduler settings
    scheduler: SchedulerConfig = None
    """Scheduler configuration."""

    # Database settings
    database: DatabaseConfig = None
    """Database integration configuration."""

    # File paths
    artifacts_dir: Path = Path("ml/artifacts")
    """Directory for storing monitoring artifacts."""

    logs_dir: Path = Path("ml/logs")
    """Directory for monitoring logs."""

    # Monitoring intervals
    monitoring_interval_minutes: int = 60
    """Interval between monitoring checks."""

    data_window_days: int = 30
    """Number of days of historical data to analyze."""

    min_samples_for_analysis: int = 100
    """Minimum number of samples required for analysis."""

    # Feature monitoring
    monitored_features: List[str] = None
    """List of features to monitor for drift."""

    exclude_features: List[str] = None
    """List of features to exclude from monitoring."""

    # Model validation
    validation_enabled: bool = True
    """Whether to enable model validation."""

    validation_metrics: List[str] = None
    """List of metrics to use for validation."""

    # API settings
    api_enabled: bool = True
    """Whether to enable REST API."""

    api_host: str = "0.0.0.0"
    """API host address."""

    api_port: int = 8001
    """API port number."""

    # Advanced settings
    enable_statistical_process_control: bool = True
    """Whether to enable statistical process control."""

    spc_control_limits: Tuple[float, float] = (2.0, 3.0)
    """Control limits for statistical process control (warning, critical)."""

    enable_trend_analysis: bool = True
    """Whether to enable performance trend analysis."""

    trend_analysis_window: int = 7
    """Window size for trend analysis (days)."""

    enable_prediction_forecasting: bool = True
    """Whether to enable performance prediction forecasting."""

    forecasting_horizon: int = 14
    """Forecasting horizon in days."""

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.performance_thresholds is None:
            object.__setattr__(self, 'performance_thresholds', PerformanceThresholds())

        if self.drift_thresholds is None:
            object.__setattr__(self, 'drift_thresholds', DriftThresholds())

        if self.retraining is None:
            object.__setattr__(self, 'retraining', RetrainingConfig())

        if self.notifications is None:
            object.__setattr__(self, 'notifications', NotificationConfig())

        if self.scheduler is None:
            object.__setattr__(self, 'scheduler', SchedulerConfig())

        if self.database is None:
            object.__setattr__(self, 'database', DatabaseConfig())

        # Set default monitored features if not provided
        if self.monitored_features is None:
            object.__setattr__(self, 'monitored_features', [
                'total_points', 'goals_scored', 'assists', 'clean_sheets',
                'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
                'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps',
                'influence', 'creativity', 'threat', 'ict_index'
            ])

        # Set default excluded features if not provided
        if self.exclude_features is None:
            object.__setattr__(self, 'exclude_features', [
                'player_id', 'season', 'gameweek', 'fixture_id',
                'team_id', 'opponent_team_id', 'position_code'
            ])

        # Set default validation metrics if not provided
        if self.validation_metrics is None:
            object.__setattr__(self, 'validation_metrics', ['mae', 'rmse', 'r2'])

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            model_name=os.getenv("ML_MODEL_NAME", "fpl_points_predictor"),
            environment=os.getenv("ML_ENVIRONMENT", "development"),
            artifacts_dir=Path(os.getenv("ML_ARTIFACTS_DIR", "ml/artifacts")),
            logs_dir=Path(os.getenv("ML_LOGS_DIR", "ml/logs")),
            monitoring_interval_minutes=int(os.getenv("ML_MONITORING_INTERVAL_MINUTES", "60")),
            data_window_days=int(os.getenv("ML_DATA_WINDOW_DAYS", "30")),
            performance_thresholds=PerformanceThresholds(
                mae_warning=float(os.getenv("ML_MAE_WARNING", "1.5")),
                mae_critical=float(os.getenv("ML_MAE_CRITICAL", "2.0")),
                rmse_warning=float(os.getenv("ML_RMSE_WARNING", "2.5")),
                rmse_critical=float(os.getenv("ML_RMSE_CRITICAL", "3.0")),
                r2_warning=float(os.getenv("ML_R2_WARNING", "0.3")),
                r2_critical=float(os.getenv("ML_R2_CRITICAL", "0.1")),
            ),
            notifications=NotificationConfig(
                enabled_channels=[ch.strip() for ch in os.getenv("ML_NOTIFICATION_CHANNELS", "email").split(",")],
                email_recipients=[email.strip() for email in os.getenv("ML_EMAIL_RECIPIENTS", "").split(",") if email.strip()],
                slack_webhook_url=os.getenv("ML_SLACK_WEBHOOK_URL"),
                discord_webhook_url=os.getenv("ML_DISCORD_WEBHOOK_URL"),
                alert_cooldown_minutes=int(os.getenv("ML_ALERT_COOLDOWN_MINUTES", "15")),
            ),
            database=DatabaseConfig(
                enabled=os.getenv("ML_DATABASE_ENABLED", "true").lower() == "true",
                connection_string=os.getenv("ML_DATABASE_CONNECTION_STRING"),
            ),
        )

    def to_dict(self) -> Dict[str, Union[str, int, float, bool, List[str]]]:
        """Convert configuration to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "environment": self.environment,
            "artifacts_dir": str(self.artifacts_dir),
            "logs_dir": str(self.logs_dir),
            "monitoring_interval_minutes": self.monitoring_interval_minutes,
            "data_window_days": self.data_window_days,
            "min_samples_for_analysis": self.min_samples_for_analysis,
            "performance_thresholds": {
                "mae_warning": self.performance_thresholds.mae_warning,
                "mae_critical": self.performance_thresholds.mae_critical,
                "rmse_warning": self.performance_thresholds.rmse_warning,
                "rmse_critical": self.performance_thresholds.rmse_critical,
                "r2_warning": self.performance_thresholds.r2_warning,
                "r2_critical": self.performance_thresholds.r2_critical,
                "performance_degradation_rate": self.performance_thresholds.performance_degradation_rate,
            },
            "drift_thresholds": {
                "psi_warning": self.drift_thresholds.psi_warning,
                "psi_critical": self.drift_thresholds.psi_critical,
                "ks_statistic_warning": self.drift_thresholds.ks_statistic_warning,
                "ks_statistic_critical": self.drift_thresholds.ks_statistic_critical,
                "feature_drift_rate": self.drift_thresholds.feature_drift_rate,
            },
            "retraining": {
                "min_retrain_interval_hours": self.retraining.min_retrain_interval_hours,
                "max_retrain_attempts": self.retraining.max_retrain_attempts,
                "performance_improvement_threshold": self.retraining.performance_improvement_threshold,
                "enable_emergency_retraining": self.retraining.enable_emergency_retraining,
                "emergency_retraining_threshold": self.retraining.emergency_retraining_threshold,
                "enable_gradual_rollout": self.retraining.enable_gradual_rollout,
                "gradual_rollout_percentage": self.retraining.gradual_rollout_percentage,
                "rollback_enabled": self.retraining.rollback_enabled,
                "quality_gate_threshold": self.retraining.quality_gate_threshold,
            },
            "notifications": {
                "enabled_channels": [ch.value for ch in self.notifications.enabled_channels],
                "email_recipients": self.notifications.email_recipients,
                "alert_cooldown_minutes": self.notifications.alert_cooldown_minutes,
                "critical_alerts_only": self.notifications.critical_alerts_only,
                "include_model_metrics": self.notifications.include_model_metrics,
            },
            "scheduler": {
                "enabled": self.scheduler.enabled,
                "schedule_cron": self.scheduler.schedule_cron,
                "enable_performance_based": self.scheduler.enable_performance_based,
                "performance_check_interval_minutes": self.scheduler.performance_check_interval_minutes,
                "max_concurrent_jobs": self.scheduler.max_concurrent_jobs,
                "job_timeout_hours": self.scheduler.job_timeout_hours,
                "retry_failed_jobs": self.scheduler.retry_failed_jobs,
                "max_job_retries": self.scheduler.max_job_retries,
            },
            "database": {
                "enabled": self.database.enabled,
                "performance_table": self.database.performance_table,
                "drift_table": self.database.drift_table,
                "model_versions_table": self.database.model_versions_table,
                "batch_size": self.database.batch_size,
                "connection_pool_size": self.database.connection_pool_size,
            },
            "monitored_features": self.monitored_features,
            "exclude_features": self.exclude_features,
            "validation_enabled": self.validation_enabled,
            "validation_metrics": self.validation_metrics,
            "api_enabled": self.api_enabled,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "enable_statistical_process_control": self.enable_statistical_process_control,
            "spc_control_limits": self.spc_control_limits,
            "enable_trend_analysis": self.enable_trend_analysis,
            "trend_analysis_window": self.trend_analysis_window,
            "enable_prediction_forecasting": self.enable_prediction_forecasting,
            "forecasting_horizon": self.forecasting_horizon,
        }
