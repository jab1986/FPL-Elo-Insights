# FPL ML Automated Retraining Pipeline

A comprehensive automated retraining pipeline for Fantasy Premier League ML models with performance monitoring, data drift detection, intelligent scheduling, and multi-channel notifications.

## 🚀 Features

### Core Components

- **Performance Monitoring System**: Real-time tracking of model metrics with statistical process control
- **Data Drift Detection**: Multi-method drift detection using PSI, KS tests, and distribution comparisons
- **Automated Retraining Engine**: Intelligent retraining with quality gates and rollback capabilities
- **Model Version Management**: Semantic versioning with comprehensive model registry
- **Notification System**: Multi-channel alerts via email, Slack, Discord, and dashboards
- **Retraining Scheduler**: Cron-like scheduling with resource management and queue handling

### Key Capabilities

- **Continuous Monitoring**: Real-time performance tracking with configurable thresholds
- **Intelligent Triggers**: Multiple trigger mechanisms (scheduled, performance-based, emergency)
- **Quality Assurance**: Pre and post-retraining validation with configurable quality gates
- **Gradual Rollouts**: Safe model deployment with gradual traffic shifting
- **Automatic Rollbacks**: Failed deployment rollback with minimal downtime
- **Resource Management**: Intelligent resource allocation and monitoring
- **Audit Trails**: Comprehensive logging and execution history

## 📋 Requirements

### Python Dependencies

```bash
pip install numpy pandas scikit-learn scipy packaging requests psutil python-dotenv
```

### Optional Dependencies

```bash
pip install lightgbm xgboost tensorflow  # For advanced models
pip install sqlalchemy psycopg2  # For database integration
pip install matplotlib seaborn  # For visualization
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.monitoring.example` to `.env.monitoring` and configure:

```bash
# Core Configuration
ML_MODEL_NAME=fpl_points_predictor
ML_ENVIRONMENT=development

# Monitoring
ML_MONITORING_INTERVAL_MINUTES=60
ML_DATA_WINDOW_DAYS=30

# Thresholds
ML_MAE_WARNING=1.5
ML_MAE_CRITICAL=2.0
ML_PSI_WARNING=0.1
ML_PSI_CRITICAL=0.25

# Notifications
ML_NOTIFICATION_CHANNELS=email,slack
ML_EMAIL_RECIPIENTS=ml-team@company.com
ML_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
```

### Configuration File

Use `ml/config/monitoring_config.yaml` for detailed configuration:

```yaml
model:
  name: "fpl_points_predictor"
  environment: "development"

monitoring:
  interval_minutes: 60
  data_window_days: 30

performance_thresholds:
  mae_warning: 1.5
  mae_critical: 2.0
  rmse_warning: 2.5
  rmse_critical: 3.0
  r2_warning: 0.3
  r2_critical: 0.1

drift_thresholds:
  psi_warning: 0.1
  psi_critical: 0.25
  ks_statistic_warning: 0.1
  ks_statistic_critical: 0.2

retraining:
  min_retrain_interval_hours: 24
  max_retrain_attempts: 3
  performance_improvement_threshold: 0.05
  enable_emergency_retraining: true
  rollback_enabled: true
  quality_gate_threshold: 0.95
```

## 🏗️ Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│ Performance      │    │   Retraining    │
│                 │    │ Monitoring       │───▶│   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model Output  │───▶│ Data Drift       │    │   Version       │
│                 │    │ Detection        │───▶│   Manager       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   External APIs │───▶│ Notification     │    │   Scheduler     │
│                 │    │ System           │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Monitoring Phase**:
   - Performance Monitor tracks model metrics in real-time
   - Drift Detector analyzes feature distributions
   - Statistical Process Control identifies anomalies

2. **Trigger Phase**:
   - Scheduler checks for scheduled retraining jobs
   - Performance-based triggers activate on threshold breaches
   - Emergency triggers respond to critical performance drops

3. **Retraining Phase**:
   - Retraining Engine executes model training
   - Quality gates validate new model performance
   - Version Manager handles model registration

4. **Deployment Phase**:
   - Gradual rollout enables safe deployment
   - Notification System alerts stakeholders
   - Rollback capability ensures system stability

## 📖 Usage Examples

### Basic Setup

```python
from ml.config.monitoring_config import MonitoringConfig
from ml.retraining.retraining_engine import RetrainingEngine
from ml.retraining.scheduler import RetrainingScheduler

# Initialize configuration
config = MonitoringConfig.from_env()

# Create retraining engine
engine = RetrainingEngine(config)

# Manual retraining trigger
request_id = await engine.trigger_retraining(
    trigger_type="manual",
    priority=1,
    reason="Initial model training",
    requested_by="data-scientist@company.com"
)

print(f"Retraining request submitted: {request_id}")
```

### Performance Monitoring

```python
from ml.monitoring.performance_monitor import PerformanceMonitor
import numpy as np

# Initialize monitor
monitor = PerformanceMonitor(config)

# Record model performance
y_true = np.array([10, 8, 15, 12, 9])
y_pred = np.array([9.5, 8.2, 14.8, 11.9, 9.1])

metrics = monitor.record_performance(
    y_true=y_true,
    y_pred=y_pred,
    model_version="1.0.0",
    feature_count=25
)

print(f"Performance metrics: MAE={metrics.mae:.4f}, R²={metrics.r2:.4f}")
```

### Data Drift Detection

```python
from ml.monitoring.data_drift_detector import DataDriftDetector

# Initialize drift detector
drift_detector = DataDriftDetector(config)

# Detect drift between training and production data
reference_data = load_training_data()  # Historical training data
current_data = load_production_data()   # Recent production data

drift_results = drift_detector.detect_drift(
    reference_data=reference_data,
    current_data=current_data
)

# Check for significant drift
for feature, metrics in drift_results.items():
    if metrics.psi > config.drift_thresholds.psi_warning:
        print(f"Drift detected in {feature}: PSI={metrics.psi:.4f}")
```

### Model Version Management

```python
from ml.retraining.version_manager import ModelVersionManager

# Initialize version manager
version_manager = ModelVersionManager(config)

# Register a new model
model_version = version_manager.register_model(
    model=trained_model,
    pipeline_config=pipeline_config,
    performance_metrics=test_metrics,
    metadata={"training_date": "2024-01-15", "features": feature_names}
)

print(f"Registered model version: {model_version.version}")

# List all versions
versions = version_manager.list_model_versions()
print(f"Total versions: {len(versions)}")

# Activate a specific version
version_manager.activate_model_version("1.2.0")
```

### Scheduler Management

```python
from ml.retraining.scheduler import RetrainingScheduler

# Initialize scheduler
scheduler = RetrainingScheduler(config)

# Create scheduled job
job_id = scheduler.create_scheduled_job(
    name="Daily Model Retraining",
    schedule_type="cron",
    cron_expression="0 2 * * *",  # Daily at 2 AM
    priority=1,
    enabled=True
)

# Force immediate execution
scheduler.force_run_job(job_id)

# Get scheduler status
status = scheduler.get_scheduler_status()
print(f"Scheduler status: {status['is_running']}")
print(f"Active jobs: {status['active_jobs']}")
```

### Notification Testing

```python
from ml.retraining.notifications import NotificationManager

# Initialize notification manager
notifications = NotificationManager(config)

# Test all notification channels
test_results = notifications.test_notifications()
for channel, success in test_results.items():
    print(f"{channel}: {'✓' if success else '✗'}")

# Send custom alert
await notifications.send_alert(
    title="Model Performance Degradation",
    message="Model MAE has exceeded warning threshold",
    level=AlertLevel.WARNING,
    metadata={"current_mae": 1.8, "threshold": 1.5}
)
```

## 🔧 Advanced Configuration

### Custom Performance Thresholds

```python
from ml.config.monitoring_config import PerformanceThresholds

# Custom thresholds for different environments
production_thresholds = PerformanceThresholds(
    mae_warning=1.2,
    mae_critical=1.8,
    rmse_warning=2.0,
    rmse_critical=2.8,
    r2_warning=0.4,
    r2_critical=0.2
)

config = MonitoringConfig(
    environment="production",
    performance_thresholds=production_thresholds
)
```

### Database Integration

```python
from ml.config.monitoring_config import DatabaseConfig

# Configure database connection
db_config = DatabaseConfig(
    enabled=True,
    connection_string="postgresql://user:pass@host:5432/fpl_ml",
    performance_table="ml_performance_history",
    drift_table="ml_drift_history",
    model_versions_table="ml_model_versions"
)

config = MonitoringConfig(database=db_config)
```

### Custom Notification Channels

```python
from ml.config.monitoring_config import NotificationConfig, NotificationChannel

# Configure multiple notification channels
notification_config = NotificationConfig(
    enabled_channels=[
        NotificationChannel.EMAIL,
        NotificationChannel.SLACK,
        NotificationChannel.DISCORD
    ],
    email_recipients=["team@company.com", "manager@company.com"],
    slack_webhook_url="https://hooks.slack.com/...",
    discord_webhook_url="https://discord.com/api/webhooks/...",
    alert_cooldown_minutes=30
)

config = MonitoringConfig(notifications=notification_config)
```

## 📊 Monitoring and Observability

### Health Checks

```python
# Get system health status
health_status = {
    "performance": performance_monitor.get_health_status(),
    "drift": drift_detector.get_drift_health_status(),
    "retraining": retraining_engine.get_health_status(),
    "scheduler": scheduler.get_scheduler_status()
}

# Overall system health
overall_status = "healthy"
if any(status["status"] == "critical" for status in health_status.values()):
    overall_status = "critical"
elif any(status["status"] == "warning" for status in health_status.values()):
    overall_status = "warning"

print(f"Overall system status: {overall_status}")
```

### Metrics and Reporting

```python
# Get comprehensive performance summary
performance_summary = performance_monitor.get_performance_summary()
print(f"Performance summary: {json.dumps(performance_summary, indent=2)}")

# Get drift analysis report
drift_summary = drift_detector.get_drift_summary()
print(f"Drift summary: {json.dumps(drift_summary, indent=2)}")

# Export metrics to CSV
performance_monitor.export_metrics("performance_report.csv")
drift_detector.export_drift_report("drift_report.csv")
```

## 🚨 Alert Management

### Alert Types

- **Performance Alerts**: MAE, RMSE, R² threshold breaches
- **Drift Alerts**: PSI, KS statistic, distribution changes
- **Retraining Alerts**: Success/failure notifications
- **System Alerts**: Resource usage, errors, anomalies

### Alert Levels

- **INFO**: Informational messages
- **WARNING**: Performance degradation detected
- **CRITICAL**: Immediate attention required

### Alert Channels

- **Email**: Traditional email notifications
- **Slack**: Real-time team messaging
- **Discord**: Community and team communication
- **Dashboard**: Internal monitoring dashboards

## 🔄 Retraining Workflows

### Scheduled Retraining

```python
# Set up daily retraining at 2 AM
scheduler.create_scheduled_job(
    name="Daily Retraining",
    schedule_type="cron",
    cron_expression="0 2 * * *",
    priority=2,
    metadata={"type": "scheduled", "frequency": "daily"}
)
```

### Performance-Based Retraining

```python
# Automatic retraining on performance degradation
async def handle_performance_degradation():
    health_status = performance_monitor.get_health_status()

    if health_status["status"] == "critical":
        await retraining_engine.emergency_retraining(
            reason=f"Critical performance degradation: {health_status['latest_metrics']}"
        )
```

### Emergency Retraining

```python
# Immediate retraining for critical issues
emergency_request_id = await retraining_engine.emergency_retraining(
    reason="Critical model failure detected"
)

# Monitor emergency retraining
status = retraining_engine.get_retraining_status(emergency_request_id)
print(f"Emergency retraining status: {status['status']}")
```

## 📈 Best Practices

### Model Quality Gates

```python
# Define strict quality gates
quality_gates = {
    "min_r2": 0.5,
    "max_mae": 1.2,
    "performance_improvement": 0.03,
    "data_quality_threshold": 0.95
}

# Apply quality gates during retraining
if not retraining_engine._check_quality_gates(metrics, improvement):
    logger.warning("Quality gates failed - model rejected")
    # Trigger rollback or alternative model
```

### Gradual Rollouts

```python
# Enable gradual rollout for production
config.retraining.enable_gradual_rollout = True
config.retraining.gradual_rollout_percentage = 0.1  # 10% traffic

# Monitor rollout performance
rollout_metrics = monitor_gradual_rollout(model_version)
if rollout_metrics["error_rate"] > threshold:
    rollback_to_previous_version()
```

### Resource Management

```python
# Monitor system resources
resource_usage = scheduler._get_resource_usage()

if resource_usage["memory_percent"] > 80:
    # Scale down or pause non-critical jobs
    pause_non_critical_jobs()

if resource_usage["cpu_percent"] > 70:
    # Reduce concurrency
    scheduler.scheduler_config.max_concurrent_jobs = 1
```

## 🐛 Troubleshooting

### Common Issues

1. **High Memory Usage**:
   - Reduce `max_concurrent_jobs` in scheduler config
   - Increase monitoring intervals
   - Clean up old model artifacts

2. **False Positive Alerts**:
   - Adjust threshold values in configuration
   - Review alert cooldown settings
   - Analyze historical data for baseline establishment

3. **Retraining Failures**:
   - Check data quality and availability
   - Review model training logs
   - Verify resource constraints

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Detailed component logging
logger = logging.getLogger("ml.retraining")
logger.setLevel(logging.DEBUG)
```

### Health Check Commands

```bash
# Check system health via CLI
python -c "
from ml.config.monitoring_config import MonitoringConfig
from ml.monitoring.performance_monitor import PerformanceMonitor
config = MonitoringConfig.from_env()
monitor = PerformanceMonitor(config)
print('Health Status:', monitor.get_health_status())
"
```

## 📚 API Reference

### RetrainingEngine

- `trigger_retraining()`: Trigger model retraining
- `emergency_retraining()`: Emergency retraining
- `get_retraining_status()`: Check retraining status

### PerformanceMonitor

- `record_performance()`: Record model metrics
- `get_performance_summary()`: Get performance statistics
- `get_health_status()`: Get system health

### DataDriftDetector

- `detect_drift()`: Detect data drift
- `get_drift_summary()`: Get drift statistics
- `export_drift_report()`: Export drift analysis

### NotificationManager

- `send_alert()`: Send notification
- `test_notifications()`: Test notification channels
- `get_notification_summary()`: Get notification statistics

### RetrainingScheduler

- `create_scheduled_job()`: Create scheduled job
- `get_scheduler_status()`: Get scheduler status
- `force_run_job()`: Execute job immediately

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive type hints
3. Include unit tests for new components
4. Update documentation for API changes
5. Test with multiple environments

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Contact the ML engineering team
- Check the troubleshooting guide

---

**FPL ML Automated Retraining Pipeline** - Ensuring your models stay fresh and perform at their best!
