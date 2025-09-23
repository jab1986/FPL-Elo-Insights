"""Pipeline integration module for automatic Tableau data exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..pipeline import PipelineResult
from ..config import PipelineConfig
from .export import TableauDataExporter, TableauExportConfig


@dataclass
class TableauIntegrationConfig:
    """Configuration for Tableau pipeline integration."""

    # Export settings
    auto_export_enabled: bool = True
    export_after_training: bool = True
    export_after_evaluation: bool = True

    # Output settings
    base_output_dir: Path = Path("ml/artifacts/tableau")
    create_versioned_dirs: bool = True
    keep_export_history: int = 10

    # Notification settings
    enable_notifications: bool = False
    notification_webhook: Optional[str] = None
    notification_email: Optional[str] = None

    # Quality gates
    min_model_performance: float = 0.5  # Minimum R² score
    max_error_threshold: float = 10.0   # Maximum MAE
    require_evaluation: bool = True

    def __post_init__(self):
        if isinstance(self.base_output_dir, str):
            self.base_output_dir = Path(self.base_output_dir)


class TableauPipelineIntegration:
    """Integration layer for automatic Tableau exports from ML pipeline."""

    def __init__(self, config: TableauIntegrationConfig = None):
        self.config = config or TableauIntegrationConfig()
        self.exporter = TableauDataExporter()
        self.export_history: list = []

    def integrate_with_pipeline(self, pipeline_config: PipelineConfig) -> None:
        """Integrate Tableau exports with the ML pipeline.

        Args:
            pipeline_config: ML pipeline configuration
        """
        # Update pipeline config with Tableau integration
        pipeline_config.output_dir = str(self.config.base_output_dir / "pipeline_output")

        # Add evaluation to pipeline if not already enabled
        if not hasattr(pipeline_config, 'run_evaluation'):
            pipeline_config.run_evaluation = self.config.require_evaluation

    def export_after_training(
        self,
        pipeline_result: PipelineResult,
        model_version: str = None
    ) -> Dict[str, Path]:
        """Export Tableau data after model training.

        Args:
            pipeline_result: Result from pipeline execution
            model_version: Version identifier for the model

        Returns:
            Dictionary of exported file paths
        """
        if not self.config.export_after_training:
            return {}

        # Check quality gates
        if not self._check_quality_gates(pipeline_result):
            print("Warning: Model does not meet quality thresholds. Exporting with warnings.")
            self._log_quality_issues(pipeline_result)

        # Generate model version
        if model_version is None:
            model_version = self._generate_model_version(pipeline_result)

        # Create versioned output directory
        output_dir = self._create_versioned_output_dir(model_version)

        # Export comprehensive dashboard data
        exported_files = self.exporter.export_comprehensive_dashboard_data(
            pipeline_result, output_dir, model_version
        )

        # Record export in history
        self._record_export(model_version, exported_files, "training")

        # Send notifications if enabled
        if self.config.enable_notifications:
            self._send_export_notification(model_version, exported_files, "success")

        return exported_files

    def export_after_evaluation(
        self,
        pipeline_result: PipelineResult,
        model_version: str = None
    ) -> Dict[str, Path]:
        """Export Tableau data after model evaluation.

        Args:
            pipeline_result: Result from pipeline execution with evaluation
            model_version: Version identifier for the model

        Returns:
            Dictionary of exported file paths
        """
        if not self.config.export_after_evaluation:
            return {}

        # Require evaluation results
        if not pipeline_result.evaluation:
            print("Warning: No evaluation results found. Skipping evaluation export.")
            return {}

        # Generate model version
        if model_version is None:
            model_version = self._generate_model_version(pipeline_result)

        # Create versioned output directory
        output_dir = self._create_versioned_output_dir(model_version)

        # Export evaluation-specific data
        exported_files = self._export_evaluation_data(
            pipeline_result, output_dir, model_version
        )

        # Record export in history
        self._record_export(model_version, exported_files, "evaluation")

        return exported_files

    def cleanup_old_exports(self) -> int:
        """Clean up old export directories keeping only recent history.

        Returns:
            Number of directories cleaned up
        """
        if not self.config.keep_export_history:
            return 0

        base_path = self.config.base_output_dir
        if not base_path.exists():
            return 0

        # Get all versioned directories
        version_dirs = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    # Extract timestamp from directory name
                    timestamp_str = item.name.split("_")[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    version_dirs.append((item, timestamp))
                except (ValueError, IndexError):
                    continue

        # Sort by timestamp
        version_dirs.sort(key=lambda x: x[1])

        # Remove old directories
        removed_count = 0
        while len(version_dirs) > self.config.keep_export_history:
            old_dir, _ = version_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(old_dir)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Failed to remove old export directory {old_dir}: {e}")

        return removed_count

    def get_export_history(self) -> list:
        """Get export history.

        Returns:
            List of export history entries
        """
        return self.export_history.copy()

    def _check_quality_gates(self, pipeline_result: PipelineResult) -> bool:
        """Check if model meets quality thresholds.

        Args:
            pipeline_result: Pipeline execution result

        Returns:
            True if quality gates are passed
        """
        if not pipeline_result.evaluation:
            return False

        # Check R² score
        r2_score = pipeline_result.evaluation.metrics.get('r2', 0)
        if r2_score < self.config.min_model_performance:
            return False

        # Check MAE
        mae = pipeline_result.evaluation.metrics.get('mae', float('inf'))
        if mae > self.config.max_error_threshold:
            return False

        return True

    def _log_quality_issues(self, pipeline_result: PipelineResult) -> None:
        """Log quality issues for monitoring.

        Args:
            pipeline_result: Pipeline execution result
        """
        issues = []

        if not pipeline_result.evaluation:
            issues.append("No evaluation results found")
        else:
            r2_score = pipeline_result.evaluation.metrics.get('r2', 0)
            mae = pipeline_result.evaluation.metrics.get('mae', float('inf'))

            if r2_score < self.config.min_model_performance:
                issues.append(f"R² score {r2_score:.3f} below threshold {self.config.min_model_performance}")

            if mae > self.config.max_error_threshold:
                issues.append(f"MAE {mae:.3f} above threshold {self.config.max_error_threshold}")

        # Log issues
        quality_log = {
            "timestamp": datetime.now().isoformat(),
            "model_type": pipeline_result.selected_model_type or "unknown",
            "issues": issues,
            "metrics": pipeline_result.evaluation.metrics if pipeline_result.evaluation else {}
        }

        log_path = self.config.base_output_dir / "quality_issues.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'a') as f:
            f.write(json.dumps(quality_log) + '\n')

    def _generate_model_version(self, pipeline_result: PipelineResult) -> str:
        """Generate model version identifier.

        Args:
            pipeline_result: Pipeline execution result

        Returns:
            Model version string
        """
        model_type = pipeline_result.selected_model_type or "model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{timestamp}"

    def _create_versioned_output_dir(self, model_version: str) -> Path:
        """Create versioned output directory.

        Args:
            model_version: Model version identifier

        Returns:
            Path to created output directory
        """
        if self.config.create_versioned_dirs:
            output_dir = self.config.base_output_dir / model_version
        else:
            output_dir = self.config.base_output_dir / "latest"

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _export_evaluation_data(
        self,
        pipeline_result: PipelineResult,
        output_dir: Path,
        model_version: str
    ) -> Dict[str, Path]:
        """Export evaluation-specific data for Tableau.

        Args:
            pipeline_result: Pipeline execution result
            output_dir: Output directory
            model_version: Model version identifier

        Returns:
            Dictionary of exported file paths
        """
        files = {}

        # Use existing evaluation export functionality
        from ..evaluation_export import save_evaluation_artifacts

        try:
            eval_files = save_evaluation_artifacts(pipeline_result, output_dir)

            # Add model version to file paths
            files = {f"evaluation_{k}": v for k, v in eval_files.items()}

            # Create evaluation summary
            summary = self._create_evaluation_summary(pipeline_result, model_version)
            summary_path = output_dir / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            files["evaluation_summary"] = summary_path

        except Exception as e:
            print(f"Warning: Failed to export evaluation data: {e}")
            files["error"] = "Evaluation export failed"

        return files

    def _create_evaluation_summary(
        self,
        pipeline_result: PipelineResult,
        model_version: str
    ) -> Dict[str, Any]:
        """Create evaluation summary for Tableau.

        Args:
            pipeline_result: Pipeline execution result
            model_version: Model version identifier

        Returns:
            Evaluation summary dictionary
        """
        summary = {
            "model_version": model_version,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": pipeline_result.selected_model_type or "unknown",
            "quality_gates_passed": self._check_quality_gates(pipeline_result),
            "key_metrics": {},
            "position_performance": {},
            "quality_flags": []
        }

        if pipeline_result.evaluation:
            summary["key_metrics"] = {
                "mae": pipeline_result.evaluation.metrics.get('mae'),
                "rmse": pipeline_result.evaluation.metrics.get('rmse'),
                "r2": pipeline_result.evaluation.metrics.get('r2'),
                "mape": pipeline_result.evaluation.metrics.get('mape')
            }

            summary["position_performance"] = {
                pos: {
                    "mae": metrics.get('mae'),
                    "count": metrics.get('count')
                }
                for pos, metrics in pipeline_result.evaluation.position_metrics.items()
            }

            # Add quality flags
            if not self._check_quality_gates(pipeline_result):
                summary["quality_flags"].append("Model performance below thresholds")

            if pipeline_result.evaluation.diagnostics:
                overfitting_score = pipeline_result.evaluation.diagnostics.get('overfitting', {}).get('overfitting_score', 0)
                if overfitting_score > 0.1:
                    summary["quality_flags"].append("Potential overfitting detected")

                data_leakage_score = pipeline_result.evaluation.diagnostics.get('data_leakage', {}).get('leakage_score', 0)
                if data_leakage_score > 0.5:
                    summary["quality_flags"].append("Potential data leakage detected")

        return summary

    def _record_export(
        self,
        model_version: str,
        exported_files: Dict[str, Path],
        export_type: str
    ) -> None:
        """Record export in history.

        Args:
            model_version: Model version identifier
            exported_files: Dictionary of exported files
            export_type: Type of export (training/evaluation)
        """
        export_record = {
            "model_version": model_version,
            "export_timestamp": datetime.now().isoformat(),
            "export_type": export_type,
            "file_count": len(exported_files),
            "files": list(exported_files.keys()),
            "output_directory": str(self.config.base_output_dir / model_version)
        }

        self.export_history.append(export_record)

        # Keep only recent history
        max_history = 100
        if len(self.export_history) > max_history:
            self.export_history = self.export_history[-max_history:]

    def _send_export_notification(
        self,
        model_version: str,
        exported_files: Dict[str, Path],
        status: str
    ) -> None:
        """Send export notification.

        Args:
            model_version: Model version identifier
            exported_files: Dictionary of exported files
            status: Export status (success/error)
        """
        if not self.config.enable_notifications:
            return

        notification = {
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "file_count": len(exported_files),
            "dashboard_ready": "model_performance_metrics.csv" in str(exported_files)
        }

        # Send webhook notification
        if self.config.notification_webhook:
            try:
                import requests
                requests.post(
                    self.config.notification_webhook,
                    json=notification,
                    headers={"Content-Type": "application/json"}
                )
            except Exception as e:
                print(f"Warning: Failed to send webhook notification: {e}")

        # Send email notification
        if self.config.notification_email:
            try:
                # Email notification would be implemented here
                # This is a placeholder for email functionality
                print(f"Email notification would be sent to: {self.config.notification_email}")
                print(f"Notification content: {json.dumps(notification, indent=2)}")
            except Exception as e:
                print(f"Warning: Failed to send email notification: {e}")


def setup_tableau_integration(pipeline_config: PipelineConfig) -> TableauPipelineIntegration:
    """Convenience function to set up Tableau integration.

    Args:
        pipeline_config: ML pipeline configuration

    Returns:
        Configured Tableau integration instance
    """
    integration_config = TableauIntegrationConfig()
    integration = TableauPipelineIntegration(integration_config)
    integration.integrate_with_pipeline(pipeline_config)
    return integration


def export_tableau_data_from_pipeline(
    pipeline_result: PipelineResult,
    output_dir: Path = None,
    model_version: str = None,
    export_config: TableauIntegrationConfig = None
) -> Dict[str, Path]:
    """Convenience function to export Tableau data from pipeline results.

    Args:
        pipeline_result: Result from pipeline execution
        output_dir: Custom output directory (optional)
        model_version: Model version identifier (optional)
        export_config: Export configuration (optional)

    Returns:
        Dictionary of exported file paths
    """
    if export_config is None:
        export_config = TableauIntegrationConfig()

    if output_dir is not None:
        export_config.base_output_dir = output_dir

    integration = TableauPipelineIntegration(export_config)
    return integration.export_after_training(pipeline_result, model_version)
