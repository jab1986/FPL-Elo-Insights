# Quick Start Guide - FPL ML Automated Retraining Pipeline

This guide will help you get the automated retraining pipeline up and running in minutes.

## 🚀 1-Minute Setup

### Step 1: Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy packaging requests psutil python-dotenv
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.monitoring.example .env.monitoring

# Edit the configuration file
nano .env.monitoring
```

### Step 3: Basic Usage

```python
from ml.config.monitoring_config import MonitoringConfig
from ml.retraining.retraining_engine import RetrainingEngine
from ml.retraining.scheduler import RetrainingScheduler

# Initialize the system
config = MonitoringConfig.from_env()
engine = RetrainingEngine(config)
scheduler = RetrainingScheduler(config)

# Trigger retraining
import asyncio
request_id = asyncio.run(engine.trigger_retraining(
    trigger_type="manual",
    reason="Initial setup"
))

print(f"Retraining started: {request_id}")
```

## ⚡ 5-Minute Advanced Setup

### 1. Performance Monitoring

```python
from ml.monitoring.performance_monitor import PerformanceMonitor
import numpy as np

# Initialize monitor
monitor = PerformanceMonitor(config)

# Monitor your model performance
y_true = np.array([10, 8, 15, 12, 9])
y_pred = np.array([9.5, 8.2, 14.8, 11.9, 9.1])

# Record metrics
metrics = monitor.record_performance(
    y_true=y_true,
    y_pred=y_pred,
    model_version="1.0.0"
)

print(f"Current MAE: {metrics.mae:.4f}")
```

### 2. Data Drift Detection

```python
from ml.monitoring.data_drift_detector import DataDriftDetector

# Initialize drift detector
drift_detector = DataDriftDetector(config)

# Check for drift
drift_results = drift_detector.detect_drift(
    reference_data=training_data,
    current_data=production_data
)

# Check for significant drift
for feature, metrics in drift_results.items():
    if metrics.psi > 0.1:  # 10% drift threshold
        print(f"⚠️  Drift detected in {feature}: PSI = {metrics.psi:.3f}")
```

### 3. Scheduled Retraining

```python
# Create a daily retraining schedule
job_id = scheduler.create_scheduled_job(
    name="Daily Model Update",
    schedule_type="cron",
    cron_expression="0 2 * * *",  # Every day at 2 AM
    priority=1
)

print(f"Scheduled job created: {job_id}")

# Or create an interval-based schedule
job_id = scheduler.create_scheduled_job(
    name="Hourly Performance Check",
    schedule_type="interval",
    interval_minutes=60,  # Every hour
    priority=2
)
```

### 4. Model Version Management

```python
from ml.retraining.version_manager import ModelVersionManager

# Initialize version manager
version_manager = ModelVersionManager(config)

# Register your trained model
model_version = version_manager.register_model(
    model=your_trained_model,
    pipeline_config=pipeline_config,
    performance_metrics=performance_metrics,
    metadata={"training_date": "2024-01-15"}
)

print(f"Model registered as version: {model_version.version}")

# List all versions
versions = version_manager.list_model_versions()
print(f"Total model versions: {len(versions)}")

# Activate a specific version
version_manager.activate_model_version("1.1.0")
```

### 5. Notification Setup

```python
from ml.retraining.notifications import NotificationManager

# Initialize notification manager
notifications = NotificationManager(config)

# Test notifications
test_results = notifications.test_notifications()
print("Notification test results:", test_results)

# Send a custom alert
import asyncio
asyncio.run(notifications.send_alert(
    title="Model Performance Alert",
    message="Model performance has degraded",
    level="warning"
))
```

## 📊 Monitoring Dashboard

### Quick Health Check

```python
# Get system health status
health_status = {
    "performance": monitor.get_health_status(),
    "drift": drift_detector.get_drift_health_status(),
    "retraining": engine.get_health_status(),
    "scheduler": scheduler.get_scheduler_status()
}

# Print summary
for component, status in health_status.items():
    print(f"{component.title()}: {status['status']}")
```

### Performance Summary

```python
# Get detailed performance metrics
summary = monitor.get_performance_summary()
print(f"Model Performance Summary:")
print(f"- Current MAE: {summary['current_metrics']['mae']:.4f}")
print(f"- R² Score: {summary['current_metrics']['r2']:.4f}")
print(f"- Total Records: {summary['total_records']}")
```

## 🔧 Common Configuration

### Basic Configuration

```python
# Minimal configuration for testing
config = MonitoringConfig(
    model_name="fpl_test_model",
    environment="development",
    monitoring_interval_minutes=60,
    performance_thresholds=PerformanceThresholds(
        mae_warning=2.0,
        mae_critical=3.0
    )
)
```

### Production Configuration

```python
# Production-ready configuration
config = MonitoringConfig(
    model_name="fpl_production_model",
    environment="production",
    monitoring_interval_minutes=30,
    performance_thresholds=PerformanceThresholds(
        mae_warning=1.2,
        mae_critical=1.8,
        r2_warning=0.4,
        r2_critical=0.2
    ),
    notifications=NotificationConfig(
        enabled_channels=["email", "slack"],
        email_recipients=["ml-team@company.com"],
        slack_webhook_url="https://hooks.slack.com/..."
    )
)
```

## 📈 Example Workflow

### Complete ML Pipeline with Monitoring

```python
import asyncio
from ml.pipeline import PointsPredictionPipeline
from ml.config import PipelineConfig

async def complete_ml_workflow():
    # 1. Train model
    pipeline_config = PipelineConfig()
    pipeline = PointsPredictionPipeline(pipeline_config)
    result = pipeline.run()

    # 2. Register model with monitoring
    model_version = version_manager.register_model(
        model=result.model,
        pipeline_config=pipeline_config,
        performance_metrics=result.metrics,
        metadata={"training_type": "automated", "dataset_size": len(result.dataset)}
    )

    # 3. Set up monitoring
    monitor.record_performance(
        y_true=result.test_data["event_points"],
        y_pred=result.model.predict(result.test_data[result.features]),
        model_version=model_version.version
    )

    # 4. Schedule regular retraining
    scheduler.create_scheduled_job(
        name="Weekly Model Update",
        schedule_type="cron",
        cron_expression="0 3 * * 1",  # Every Monday at 3 AM
        priority=2
    )

    print(f"Model {model_version.version} deployed and monitored!")

# Run the workflow
asyncio.run(complete_ml_workflow())
```

## 🎯 Next Steps

### 1. Set Up Notifications

Configure your preferred notification channels:

- **Email**: Set `ML_EMAIL_RECIPIENTS` in `.env.monitoring`
- **Slack**: Set `ML_SLACK_WEBHOOK_URL` in `.env.monitoring`
- **Discord**: Set `ML_DISCORD_WEBHOOK_URL` in `.env.monitoring`

### 2. Configure Thresholds

Adjust thresholds based on your model performance:

```python
# Tighter thresholds for production
config.performance_thresholds = PerformanceThresholds(
    mae_warning=1.0,
    mae_critical=1.5,
    r2_warning=0.5,
    r2_critical=0.3
)
```

### 3. Enable Database Integration

```python
# Enable persistent storage
config.database = DatabaseConfig(
    enabled=True,
    connection_string="postgresql://user:pass@host/db"
)
```

### 4. Set Up Scheduling

```python
# Create multiple scheduled jobs
scheduler.create_scheduled_job(
    name="Daily Retraining",
    schedule_type="cron",
    cron_expression="0 2 * * *"
)

scheduler.create_scheduled_job(
    name="Performance Check",
    schedule_type="interval",
    interval_minutes=60
)
```

### 5. Monitor and Maintain

```python
# Regular health checks
def daily_health_check():
    status = scheduler.get_scheduler_status()
    if status['active_jobs'] > 3:
        print("⚠️ High job queue, consider scaling")

    if status['failed_executions'] > 5:
        print("❌ Multiple failures, check system logs")

# Run health check
daily_health_check()
```

## 🆘 Troubleshooting

### Common Issues

**Issue**: No notifications received
```python
# Test notification system
test_results = notifications.test_notifications()
print(test_results)
```

**Issue**: High memory usage
```python
# Reduce concurrent jobs
config.scheduler.max_concurrent_jobs = 1
```

**Issue**: Retraining failures
```python
# Check system resources
resources = scheduler._get_resource_usage()
print(f"Memory: {resources['memory_percent']:.1f}%")
print(f"CPU: {resources['cpu_percent']:.1f}%")
```

### Getting Help

1. Check the logs: `ml/logs/`
2. Review configuration: `ml/config/monitoring_config.yaml`
3. Test individual components
4. Check the main documentation: `ml/retraining/README.md`

---

🎉 **Congratulations!** Your automated retraining pipeline is now ready to keep your FPL models performing at their best!
