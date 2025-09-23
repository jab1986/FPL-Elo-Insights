"""Automated retraining engine with intelligent model selection and rollback capabilities."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config.monitoring_config import (
    MonitoringConfig,
    RetrainingConfig,
    AlertLevel,
)
from ..pipeline import PointsPredictionPipeline, PipelineResult
from ..config import PipelineConfig
from .version_manager import ModelVersionManager
from ..monitoring.performance_monitor import PerformanceMonitor
from ..monitoring.data_drift_detector import DataDriftDetector


@dataclass
class RetrainingRequest:
    """Container for retraining requests."""

    request_id: str
    """Unique identifier for the retraining request."""

    trigger_type: str
    """Type of trigger (e.g., 'scheduled', 'performance', 'drift', 'emergency')."""

    priority: int = 1
    """Priority level (1 = low, 5 = critical)."""

    requested_at: datetime
    """When the retraining was requested."""

    requested_by: Optional[str] = None
    """Who requested the retraining."""

    reason: Optional[str] = None
    """Reason for retraining."""

    metadata: Dict[str, Any] = None
    """Additional metadata for the request."""


@dataclass
class RetrainingResult:
    """Container for retraining results."""

    request_id: str
    """ID of the retraining request."""

    started_at: datetime
    """When retraining started."""

    completed_at: datetime
    """When retraining completed."""

    success: bool
    """Whether retraining was successful."""

    old_model_version: Optional[str] = None
    """Version of the previous model."""

    new_model_version: Optional[str] = None
    """Version of the new model."""

    performance_improvement: Optional[float] = None
    """Performance improvement percentage."""

    error_message: Optional[str] = None
    """Error message if retraining failed."""

    artifacts_path: Optional[Path] = None
    """Path to retraining artifacts."""

    metadata: Dict[str, Any] = None
    """Additional metadata."""


class RetrainingEngine:
    """Automated retraining engine with intelligent model selection."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the retraining engine.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.version_manager = ModelVersionManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.drift_detector = DataDriftDetector(config)

        # Initialize storage
        self.retraining_history: List[RetrainingResult] = []
        self.pending_requests: List[RetrainingRequest] = []
        self.active_jobs: Dict[str, asyncio.Task] = {}

        # Create directories
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing retraining history
        self._load_retraining_history()

    async def trigger_retraining(
        self,
        trigger_type: str = "manual",
        priority: int = 1,
        requested_by: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        emergency: bool = False
    ) -> str:
        """Trigger a retraining request.

        Args:
            trigger_type: Type of trigger.
            priority: Priority level (1-5).
            requested_by: Who requested the retraining.
            reason: Reason for retraining.
            metadata: Additional metadata.
            emergency: Whether this is an emergency retraining.

        Returns:
            Request ID for tracking.
        """
        request_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"

        request = RetrainingRequest(
            request_id=request_id,
            trigger_type=trigger_type,
            priority=priority,
            requested_at=datetime.now(),
            requested_by=requested_by,
            reason=reason,
            metadata=metadata or {}
        )

        self.pending_requests.append(request)

        # Sort by priority (higher priority first)
        self.pending_requests.sort(key=lambda x: (-x.priority, x.requested_at))

        self.logger.info(f"Retraining request {request_id} triggered by {trigger_type}")

        # Process immediately if emergency or high priority
        if emergency or priority >= 4:
            asyncio.create_task(self._process_retraining_request(request))
        else:
            # Schedule for background processing
            asyncio.create_task(self._process_pending_requests())

        return request_id

    async def _process_retraining_request(self, request: RetrainingRequest) -> RetrainingResult:
        """Process a single retraining request.

        Args:
            request: Retraining request to process.

        Returns:
            Retraining result.
        """
        self.logger.info(f"Processing retraining request {request.request_id}")

        # Check if retraining is allowed
        if not self._can_retrain(request):
            error_msg = "Retraining not allowed: minimum interval not met"
            result = RetrainingResult(
                request_id=request.request_id,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                success=False,
                error_message=error_msg,
                metadata={"reason": "interval_check_failed"}
            )
            self.retraining_history.append(result)
            self._save_retraining_history()
            return result

        try:
            # Get current model version
            current_version = self.version_manager.get_current_version()

            # Load and prepare data
            pipeline_config = PipelineConfig()
            pipeline = PointsPredictionPipeline(pipeline_config)

            # Run the pipeline
            pipeline_result = pipeline.run()

            # Evaluate the new model
            new_version = self.version_manager.register_model(
                model=pipeline_result.model,
                pipeline_config=pipeline_config,
                performance_metrics=pipeline_result.metrics,
                metadata={
                    "trigger_type": request.trigger_type,
                    "request_id": request.request_id,
                    "features": pipeline_result.features,
                    "training_samples": pipeline_result.train_rows
                }
            )

            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(
                current_version, new_version, pipeline_result.metrics
            )

            # Check quality gates
            quality_passed = self._check_quality_gates(
                pipeline_result.metrics, performance_improvement
            )

            if quality_passed:
                # Activate the new model
                self.version_manager.activate_model_version(new_version)

                result = RetrainingResult(
                    request_id=request.request_id,
                    started_at=request.requested_at,
                    completed_at=datetime.now(),
                    success=True,
                    old_model_version=current_version.version if current_version else None,
                    new_model_version=new_version.version,
                    performance_improvement=performance_improvement,
                    artifacts_path=self.config.artifacts_dir,
                    metadata={
                        "trigger_type": request.trigger_type,
                        "quality_gates_passed": True,
                        "training_metrics": pipeline_result.metrics,
                        "features_used": len(pipeline_result.features)
                    }
                )

                self.logger.info(f"Retraining successful: {request.request_id}")
            else:
                # Quality gates failed - rollback or reject
                if self.config.retraining.rollback_enabled:
                    result = RetrainingResult(
                        request_id=request.request_id,
                        started_at=request.requested_at,
                        completed_at=datetime.now(),
                        success=False,
                        old_model_version=current_version.version if current_version else None,
                        new_model_version=new_version.version,
                        performance_improvement=performance_improvement,
                        error_message="Quality gates failed",
                        metadata={
                            "trigger_type": request.trigger_type,
                            "quality_gates_passed": False,
                            "training_metrics": pipeline_result.metrics
                        }
                    )

                    # Clean up failed model
                    self.version_manager.delete_model_version(new_version.version)

                    self.logger.warning(f"Retraining failed quality gates: {request.request_id}")
                else:
                    # Keep the new model but don't activate
                    result = RetrainingResult(
                        request_id=request.request_id,
                        started_at=request.requested_at,
                        completed_at=datetime.now(),
                        success=False,
                        old_model_version=current_version.version if current_version else None,
                        new_model_version=new_version.version,
                        performance_improvement=performance_improvement,
                        error_message="Quality gates failed - model not activated",
                        metadata={
                            "trigger_type": request.trigger_type,
                            "quality_gates_passed": False,
                            "training_metrics": pipeline_result.metrics,
                            "model_kept": True
                        }
                    )

                    self.logger.warning(f"Retraining quality gates failed, model kept: {request.request_id}")

        except Exception as e:
            error_msg = f"Retraining failed: {str(e)}"

            result = RetrainingResult(
                request_id=request.request_id,
                started_at=request.requested_at,
                completed_at=datetime.now(),
                success=False,
                error_message=error_msg,
                metadata={
                    "trigger_type": request.trigger_type,
                    "exception": str(e)
                }
            )

            self.logger.error(f"Retraining error: {error_msg}")

        self.retraining_history.append(result)
        self._save_retraining_history()

        return result

    def _can_retrain(self, request: RetrainingRequest) -> bool:
        """Check if retraining is allowed based on configuration.

        Args:
            request: Retraining request to check.

        Returns:
            True if retraining is allowed.
        """
        retrain_config = self.config.retraining

        # Check minimum interval
        recent_retraining = [
            r for r in self.retraining_history
            if r.completed_at > datetime.now() - timedelta(hours=retrain_config.min_retrain_interval_hours)
            and r.success
        ]

        if len(recent_retraining) >= retrain_config.max_retrain_attempts:
            return False

        # Emergency retraining bypasses some checks
        if request.trigger_type == "emergency":
            return retrain_config.enable_emergency_retraining

        return True

    def _calculate_performance_improvement(
        self,
        old_version: Optional[Any],
        new_version: Any,
        new_metrics: Dict[str, Dict[str, float]]
    ) -> Optional[float]:
        """Calculate performance improvement compared to previous model.

        Args:
            old_version: Previous model version.
            new_version: New model version.
            new_metrics: Metrics of the new model.

        Returns:
            Performance improvement percentage (positive = improvement).
        """
        if not old_version or "test" not in new_metrics:
            return None

        # Use MAE for comparison (lower is better)
        new_mae = new_metrics["test"].get("mae", float("inf"))

        # Get old model performance
        old_metrics = self.version_manager.get_model_metrics(old_version.version)
        if not old_metrics or "test" not in old_metrics:
            return None

        old_mae = old_metrics["test"].get("mae", float("inf"))

        if old_mae == 0 or old_mae == float("inf"):
            return None

        # Calculate improvement (positive = improvement)
        improvement = (old_mae - new_mae) / old_mae * 100
        return improvement

    def _check_quality_gates(
        self,
        metrics: Dict[str, Dict[str, float]],
        performance_improvement: Optional[float]
    ) -> bool:
        """Check if model meets quality gate requirements.

        Args:
            metrics: Model performance metrics.
            performance_improvement: Performance improvement percentage.

        Returns:
            True if quality gates are passed.
        """
        retrain_config = self.config.retraining

        # Check minimum performance thresholds
        if "test" not in metrics:
            return False

        test_metrics = metrics["test"]

        # Basic quality checks
        required_metrics = ["mae", "rmse", "r2"]
        if not all(metric in test_metrics for metric in required_metrics):
            return False

        # Check if metrics are reasonable (not NaN, not infinite)
        for metric_name, value in test_metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                return False

        # Check performance improvement threshold
        if performance_improvement is not None:
            if performance_improvement < retrain_config.performance_improvement_threshold:
                return False

        # Check R² is positive (basic sanity check)
        if test_metrics.get("r2", -1) < 0:
            return False

        return True

    async def _process_pending_requests(self) -> None:
        """Process pending retraining requests in background."""
        while self.pending_requests:
            # Get highest priority request
            request = self.pending_requests.pop(0)

            # Skip if already being processed
            if request.request_id in self.active_jobs:
                continue

            # Start processing
            task = asyncio.create_task(self._process_retraining_request(request))
            self.active_jobs[request.request_id] = task

            # Wait for completion
            try:
                await task
            except Exception as e:
                self.logger.error(f"Error processing retraining request {request.request_id}: {e}")

            # Clean up
            if request.request_id in self.active_jobs:
                del self.active_jobs[request.request_id]

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(1)

    def get_retraining_status(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get retraining status.

        Args:
            request_id: Specific request ID to check. If None, returns general status.

        Returns:
            Dictionary with retraining status.
        """
        if request_id:
            # Check specific request
            pending = [r for r in self.pending_requests if r.request_id == request_id]
            active = request_id in self.active_jobs
            completed = [r for r in self.retraining_history if r.request_id == request_id]

            if pending:
                return {"status": "pending", "request": pending[0].__dict__}
            elif active:
                return {"status": "active", "request_id": request_id}
            elif completed:
                return {"status": "completed", "result": completed[0].__dict__}
            else:
                return {"status": "not_found", "request_id": request_id}
        else:
            # General status
            return {
                "pending_requests": len(self.pending_requests),
                "active_jobs": len(self.active_jobs),
                "total_completed": len(self.retraining_history),
                "successful_retrainings": len([r for r in self.retraining_history if r.success]),
                "failed_retrainings": len([r for r in self.retraining_history if not r.success]),
                "latest_result": self.retraining_history[-1].__dict__ if self.retraining_history else None
            }

    def _load_retraining_history(self) -> None:
        """Load retraining history from disk."""
        history_file = self.config.artifacts_dir / "retraining_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                for item in data:
                    result = RetrainingResult(
                        request_id=item["request_id"],
                        started_at=datetime.fromisoformat(item["started_at"]),
                        completed_at=datetime.fromisoformat(item["completed_at"]),
                        success=item["success"],
                        old_model_version=item.get("old_model_version"),
                        new_model_version=item.get("new_model_version"),
                        performance_improvement=item.get("performance_improvement"),
                        error_message=item.get("error_message"),
                        artifacts_path=Path(item["artifacts_path"]) if item.get("artifacts_path") else None,
                        metadata=item.get("metadata", {})
                    )
                    self.retraining_history.append(result)

                self.logger.info(f"Loaded {len(self.retraining_history)} retraining records")

            except Exception as e:
                self.logger.error(f"Failed to load retraining history: {e}")

    def _save_retraining_history(self) -> None:
        """Save retraining history to disk."""
        history_file = self.config.artifacts_dir / "retraining_history.json"

        try:
            data = []
            for result in self.retraining_history:
                data.append({
                    "request_id": result.request_id,
                    "started_at": result.started_at.isoformat(),
                    "completed_at": result.completed_at.isoformat(),
                    "success": result.success,
                    "old_model_version": result.old_model_version,
                    "new_model_version": result.new_model_version,
                    "performance_improvement": result.performance_improvement,
                    "error_message": result.error_message,
                    "artifacts_path": str(result.artifacts_path) if result.artifacts_path else None,
                    "metadata": result.metadata
                })

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save retraining history: {e}")

    def emergency_retraining(self, reason: str) -> str:
        """Trigger emergency retraining.

        Args:
            reason: Reason for emergency retraining.

        Returns:
            Request ID for tracking.
        """
        return asyncio.run(self.trigger_retraining(
            trigger_type="emergency",
            priority=5,
            reason=reason,
            emergency=True
        ))

    def get_health_status(self) -> Dict[str, Any]:
        """Get retraining engine health status.

        Returns:
            Dictionary with health status information.
        """
        recent_successful = [
            r for r in self.retraining_history
            if r.success and (datetime.now() - r.completed_at).days <= 7
        ]

        recent_failed = [
            r for r in self.retraining_history
            if not r.success and (datetime.now() - r.completed_at).days <= 7
        ]

        return {
            "status": "healthy" if len(recent_failed) == 0 else "warning" if len(recent_failed) < len(recent_successful) else "critical",
            "recent_successful": len(recent_successful),
            "recent_failed": len(recent_failed),
            "pending_requests": len(self.pending_requests),
            "active_jobs": len(self.active_jobs),
            "success_rate": len(recent_successful) / max(len(recent_successful) + len(recent_failed), 1)
        }
