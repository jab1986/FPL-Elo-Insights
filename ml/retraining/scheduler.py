"""Retraining scheduler with cron-like scheduling and intelligent resource management."""

from __future__ import annotations

import asyncio
import json
import logging
import psutil
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import pandas as pd

from ..config.monitoring_config import (
    MonitoringConfig,
    SchedulerConfig,
)
from ..data import load_merged_gameweek_data
from ..config import PipelineConfig
from .retraining_engine import RetrainingEngine, RetrainingRequest
from ..monitoring.performance_monitor import PerformanceMonitor
from ..monitoring.data_drift_detector import DataDriftDetector
from ..retraining.notifications import NotificationManager


@dataclass
class ScheduledJob:
    """Container for scheduled retraining jobs."""

    job_id: str
    """Unique job identifier."""

    name: str
    """Human-readable job name."""

    schedule_type: str
    """Type of schedule (cron, interval, performance_based)."""

    cron_expression: Optional[str] = None
    """Cron expression for scheduled jobs."""

    interval_minutes: Optional[int] = None
    """Interval in minutes for interval-based jobs."""

    enabled: bool = True
    """Whether the job is enabled."""

    last_run: Optional[datetime] = None
    """When the job last ran."""

    next_run: Optional[datetime] = None
    """When the job is next scheduled to run."""

    priority: int = 1
    """Job priority (1-5)."""

    max_retries: int = 3
    """Maximum number of retries."""

    retry_count: int = 0
    """Current retry count."""

    metadata: Dict[str, Any] = None
    """Additional job metadata."""


@dataclass
class JobExecution:
    """Container for job execution results."""

    job_id: str
    """Job identifier."""

    started_at: datetime
    """When execution started."""

    completed_at: Optional[datetime] = None
    """When execution completed."""

    success: bool = False
    """Whether execution was successful."""

    error_message: Optional[str] = None
    """Error message if execution failed."""

    retraining_request_id: Optional[str] = None
    """Associated retraining request ID."""

    metadata: Dict[str, Any] = None
    """Execution metadata."""


class RetrainingScheduler:
    """Intelligent scheduler for ML model retraining."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the retraining scheduler.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.scheduler_config = config.scheduler
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.retraining_engine = RetrainingEngine(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.drift_detector = DataDriftDetector(config)
        self.notification_manager = NotificationManager(config)

        # Scheduler state
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.execution_history: List[JobExecution] = []
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Resource monitoring
        self.resource_check_interval = 30  # seconds
        self.max_memory_usage = 0.8  # 80% of available memory
        self.max_cpu_usage = 0.7  # 70% of available CPU

        # Create directories
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state
        self._load_scheduler_state()

        # Start scheduler if enabled
        if self.scheduler_config.enabled:
            self.start()

    def start(self) -> None:
        """Start the scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return

        self.is_running = True
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("Retraining scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)

        self.logger.info("Retraining scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        self.logger.info("Scheduler loop started")

        while self.is_running and not self.stop_event.is_set():
            try:
                # Check for scheduled jobs
                self._check_scheduled_jobs()

                # Check performance-based triggers
                if self.scheduler_config.enable_performance_based:
                    self._check_performance_triggers()

                # Resource management
                self._manage_resources()

                # Process pending retraining requests
                self._process_pending_requests()

                # Sleep for scheduler interval
                self.stop_event.wait(self.resource_check_interval)

            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                self.stop_event.wait(60)  # Wait 1 minute on error

        self.logger.info("Scheduler loop stopped")

    def _check_scheduled_jobs(self) -> None:
        """Check and execute scheduled jobs."""
        current_time = datetime.now()

        for job in self.scheduled_jobs.values():
            if not job.enabled or not job.next_run:
                continue

            if current_time >= job.next_run:
                try:
                    self.logger.info(f"Executing scheduled job: {job.name}")
                    self._execute_job(job)

                except Exception as e:
                    self.logger.error(f"Failed to execute scheduled job {job.job_id}: {e}")

    def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job.

        Args:
            job: Job to execute.
        """
        # Check resource availability
        if not self._check_resources_available():
            self.logger.warning(f"Insufficient resources for job {job.job_id}, skipping")
            return

        # Create execution record
        execution = JobExecution(
            job_id=job.job_id,
            started_at=datetime.now(),
            metadata={"scheduled_job": True, "job_name": job.name}
        )

        try:
            # Trigger retraining
            retraining_request_id = asyncio.run(self.retraining_engine.trigger_retraining(
                trigger_type=job.schedule_type,
                priority=job.priority,
                reason=f"Scheduled job: {job.name}",
                metadata=job.metadata
            ))

            execution.retraining_request_id = retraining_request_id
            execution.success = True
            execution.completed_at = datetime.now()

            # Update job execution history
            job.last_run = datetime.now()
            job.retry_count = 0

            # Calculate next run time
            job.next_run = self._calculate_next_run(job)

            self.logger.info(f"Scheduled job {job.job_id} executed successfully")

        except Exception as e:
            execution.success = False
            execution.error_message = str(e)
            execution.completed_at = datetime.now()

            # Handle retries
            job.retry_count += 1
            if job.retry_count >= job.max_retries:
                self.logger.error(f"Job {job.job_id} failed after {job.max_retries} retries")
                self.notification_manager.send_retraining_alert(
                    "failed",
                    f"Scheduled job {job.name} failed after {job.max_retries} retries: {e}",
                    {"job_id": job.job_id, "error": str(e)}
                )
            else:
                # Retry with exponential backoff
                retry_delay = 2 ** job.retry_count * 60  # 2, 4, 8 minutes
                job.next_run = datetime.now() + timedelta(minutes=retry_delay)
                self.logger.warning(f"Job {job.job_id} failed, retrying in {retry_delay} minutes")

        finally:
            self.execution_history.append(execution)
            self._save_scheduler_state()

    def _calculate_next_run(self, job: ScheduledJob) -> datetime:
        """Calculate next run time for a job.

        Args:
            job: Job for which to calculate next run.

        Returns:
            Next run datetime.
        """
        current_time = datetime.now()

        if job.schedule_type == "interval" and job.interval_minutes:
            return current_time + timedelta(minutes=job.interval_minutes)

        elif job.schedule_type == "cron" and job.cron_expression:
            return self._parse_cron_expression(job.cron_expression, current_time)

        else:
            # Default to daily
            return current_time + timedelta(days=1)

    def _parse_cron_expression(self, cron_expr: str, base_time: datetime) -> datetime:
        """Parse cron expression and return next run time.

        Args:
            cron_expr: Cron expression (minute hour day month weekday).
            base_time: Base time for calculation.

        Returns:
            Next run datetime.
        """
        # Simplified cron parsing - would need full implementation for production
        parts = cron_expr.split()
        if len(parts) != 5:
            return base_time + timedelta(days=1)

        minute, hour, day, month, weekday = parts

        # Calculate next run (simplified logic)
        next_time = base_time + timedelta(hours=1)
        next_time = next_time.replace(minute=int(minute) if minute != "*" else 0)

        return next_time

    def _check_performance_triggers(self) -> None:
        """Check for performance-based retraining triggers."""
        try:
            # Get current model health status
            health_status = self.performance_monitor.get_health_status()

            if health_status["status"] in ["warning", "critical"]:
                self.logger.info(f"Performance degradation detected: {health_status['status']}")

                # Trigger performance-based retraining
                asyncio.run(self.retraining_engine.trigger_retraining(
                    trigger_type="performance",
                    priority=3,
                    reason=f"Performance degradation: {health_status['status']}",
                    metadata=health_status
                ))

                self.notification_manager.send_retraining_alert(
                    "triggered",
                    f"Performance-based retraining triggered due to {health_status['status']} status",
                    health_status
                )

        except Exception as e:
            self.logger.error(f"Error checking performance triggers: {e}")

    def _check_resources_available(self) -> bool:
        """Check if system resources are available for retraining.

        Returns:
            True if resources are available.
        """
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory_usage * 100:
                self.logger.warning(f"Memory usage too high: {memory.percent:.1f}%")
                return False

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.max_cpu_usage * 100:
                self.logger.warning(f"CPU usage too high: {cpu_percent:.1f}%")
                return False

            # Check active retraining jobs
            active_jobs = len([e for e in self.execution_history
                             if e.started_at > datetime.now() - timedelta(hours=1)
                             and not e.completed_at])

            if active_jobs >= self.scheduler_config.max_concurrent_jobs:
                self.logger.warning(f"Too many concurrent jobs: {active_jobs}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            return False

    def _manage_resources(self) -> None:
        """Manage system resources and cleanup."""
        try:
            # Clean up old execution history
            cutoff_time = datetime.now() - timedelta(days=30)
            self.execution_history = [
                e for e in self.execution_history
                if e.started_at >= cutoff_time
            ]

            # Clean up old scheduled jobs
            inactive_jobs = [
                job_id for job_id, job in self.scheduled_jobs.items()
                if not job.enabled and job.last_run and
                job.last_run < datetime.now() - timedelta(days=90)
            ]

            for job_id in inactive_jobs:
                del self.scheduled_jobs[job_id]

            self.logger.debug("Resource management completed")

        except Exception as e:
            self.logger.error(f"Error in resource management: {e}")

    def _process_pending_requests(self) -> None:
        """Process any pending retraining requests."""
        try:
            # This is handled by the retraining engine
            # We just need to check if there are any urgent requests
            status = self.retraining_engine.get_retraining_status()
            if status.get("pending_requests", 0) > 0:
                self.logger.info(f"Processing {status['pending_requests']} pending retraining requests")

        except Exception as e:
            self.logger.error(f"Error processing pending requests: {e}")

    def create_scheduled_job(
        self,
        name: str,
        schedule_type: str,
        cron_expression: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        priority: int = 1,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new scheduled job.

        Args:
            name: Job name.
            schedule_type: Type of schedule.
            cron_expression: Cron expression for cron jobs.
            interval_minutes: Interval for interval jobs.
            priority: Job priority.
            enabled: Whether job is enabled.
            metadata: Additional metadata.

        Returns:
            Job ID.
        """
        job_id = f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            schedule_type=schedule_type,
            cron_expression=cron_expression,
            interval_minutes=interval_minutes,
            enabled=enabled,
            priority=priority,
            max_retries=3,
            metadata=metadata or {}
        )

        # Calculate next run
        job.next_run = self._calculate_next_run(job)

        self.scheduled_jobs[job_id] = job
        self._save_scheduler_state()

        self.logger.info(f"Created scheduled job: {name} ({job_id})")
        return job_id

    def update_job_schedule(
        self,
        job_id: str,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        cron_expression: Optional[str] = None,
        interval_minutes: Optional[int] = None
    ) -> bool:
        """Update a scheduled job.

        Args:
            job_id: Job ID to update.
            enabled: New enabled status.
            priority: New priority.
            cron_expression: New cron expression.
            interval_minutes: New interval.

        Returns:
            True if update was successful.
        """
        if job_id not in self.scheduled_jobs:
            return False

        job = self.scheduled_jobs[job_id]

        if enabled is not None:
            job.enabled = enabled

        if priority is not None:
            job.priority = priority

        if cron_expression is not None:
            job.cron_expression = cron_expression
            job.interval_minutes = None

        if interval_minutes is not None:
            job.interval_minutes = interval_minutes
            job.cron_expression = None

        # Recalculate next run
        job.next_run = self._calculate_next_run(job)

        self._save_scheduler_state()
        self.logger.info(f"Updated scheduled job: {job_id}")
        return True

    def delete_scheduled_job(self, job_id: str) -> bool:
        """Delete a scheduled job.

        Args:
            job_id: Job ID to delete.

        Returns:
            True if deletion was successful.
        """
        if job_id not in self.scheduled_jobs:
            return False

        del self.scheduled_jobs[job_id]
        self._save_scheduler_state()
        self.logger.info(f"Deleted scheduled job: {job_id}")
        return True

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler status.
        """
        active_jobs = len([e for e in self.execution_history
                          if e.started_at > datetime.now() - timedelta(hours=1)
                          and not e.completed_at])

        return {
            "is_running": self.is_running,
            "scheduled_jobs": len(self.scheduled_jobs),
            "active_jobs": active_jobs,
            "total_executions": len(self.execution_history),
            "successful_executions": len([e for e in self.execution_history if e.success]),
            "failed_executions": len([e for e in self.execution_history if not e.success]),
            "pending_requests": self.retraining_engine.get_retraining_status().get("pending_requests", 0),
            "resource_usage": self._get_resource_usage()
        }

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage.

        Returns:
            Dictionary with resource usage statistics.
        """
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            return {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            return {}

    def _load_scheduler_state(self) -> None:
        """Load scheduler state from disk."""
        state_file = self.config.artifacts_dir / "scheduler_state.json"

        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)

                # Load scheduled jobs
                for job_id, job_data in data.get("scheduled_jobs", {}).items():
                    job = ScheduledJob(
                        job_id=job_id,
                        name=job_data["name"],
                        schedule_type=job_data["schedule_type"],
                        cron_expression=job_data.get("cron_expression"),
                        interval_minutes=job_data.get("interval_minutes"),
                        enabled=job_data.get("enabled", True),
                        last_run=datetime.fromisoformat(job_data["last_run"]) if job_data.get("last_run") else None,
                        next_run=datetime.fromisoformat(job_data["next_run"]) if job_data.get("next_run") else None,
                        priority=job_data.get("priority", 1),
                        max_retries=job_data.get("max_retries", 3),
                        retry_count=job_data.get("retry_count", 0),
                        metadata=job_data.get("metadata", {})
                    )
                    self.scheduled_jobs[job_id] = job

                # Load execution history
                for exec_data in data.get("execution_history", []):
                    execution = JobExecution(
                        job_id=exec_data["job_id"],
                        started_at=datetime.fromisoformat(exec_data["started_at"]),
                        completed_at=datetime.fromisoformat(exec_data["completed_at"]) if exec_data.get("completed_at") else None,
                        success=exec_data.get("success", False),
                        error_message=exec_data.get("error_message"),
                        retraining_request_id=exec_data.get("retraining_request_id"),
                        metadata=exec_data.get("metadata", {})
                    )
                    self.execution_history.append(execution)

                self.logger.info(f"Loaded scheduler state: {len(self.scheduled_jobs)} jobs, {len(self.execution_history)} executions")

            except Exception as e:
                self.logger.error(f"Failed to load scheduler state: {e}")

    def _save_scheduler_state(self) -> None:
        """Save scheduler state to disk."""
        try:
            data = {
                "scheduled_jobs": {},
                "execution_history": []
            }

            # Save scheduled jobs
            for job_id, job in self.scheduled_jobs.items():
                data["scheduled_jobs"][job_id] = {
                    "name": job.name,
                    "schedule_type": job.schedule_type,
                    "cron_expression": job.cron_expression,
                    "interval_minutes": job.interval_minutes,
                    "enabled": job.enabled,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "priority": job.priority,
                    "max_retries": job.max_retries,
                    "retry_count": job.retry_count,
                    "metadata": job.metadata
                }

            # Save recent execution history
            for execution in self.execution_history[-100:]:  # Keep last 100 executions
                data["execution_history"].append({
                    "job_id": execution.job_id,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "success": execution.success,
                    "error_message": execution.error_message,
                    "retraining_request_id": execution.retraining_request_id,
                    "metadata": execution.metadata
                })

            state_file = self.config.artifacts_dir / "scheduler_state.json"
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save scheduler state: {e}")

    def get_job_history(self, job_id: Optional[str] = None, limit: int = 50) -> List[JobExecution]:
        """Get job execution history.

        Args:
            job_id: Specific job ID to filter by. If None, returns all jobs.
            limit: Maximum number of executions to return.

        Returns:
            List of job executions.
        """
        executions = self.execution_history

        if job_id:
            executions = [e for e in executions if e.job_id == job_id]

        # Sort by start time (newest first)
        executions.sort(key=lambda e: e.started_at, reverse=True)

        return executions[:limit]

    def force_run_job(self, job_id: str) -> bool:
        """Force immediate execution of a scheduled job.

        Args:
            job_id: Job ID to execute.

        Returns:
            True if execution was started.
        """
        if job_id not in self.scheduled_jobs:
            return False

        job = self.scheduled_jobs[job_id]

        if not job.enabled:
            return False

        try:
            self._execute_job(job)
            return True
        except Exception as e:
            self.logger.error(f"Failed to force run job {job_id}: {e}")
            return False
