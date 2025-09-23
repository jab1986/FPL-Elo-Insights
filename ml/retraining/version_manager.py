"""Model version management with semantic versioning and registry."""

from __future__ import annotations

import json
import pickle
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
import numpy as np
import pandas as pd

from ..config.monitoring_config import MonitoringConfig
from ..config import PipelineConfig
from ..model import RidgeRegressionModel
from ..models import PositionSpecificModel, DeepEnsembleModel


@dataclass
class ModelVersion:
    """Container for model version information."""

    version: str
    """Semantic version string (e.g., '1.2.3')."""

    created_at: datetime
    """When the model was created."""

    model_type: str
    """Type of model (ridge, position_specific, deep_ensemble)."""

    file_path: Path
    """Path to the saved model file."""

    performance_metrics: Dict[str, Dict[str, float]]
    """Performance metrics for this version."""

    feature_names: List[str]
    """Names of features used by this model."""

    training_samples: int
    """Number of samples used for training."""

    model_size_bytes: int
    """Size of the model file in bytes."""

    metadata: Dict[str, Any]
    """Additional metadata."""

    is_active: bool = False
    """Whether this is the currently active version."""

    activated_at: Optional[datetime] = None
    """When this version was activated."""


@dataclass
class ModelRegistry:
    """Container for model registry information."""

    current_version: Optional[str] = None
    """Currently active model version."""

    versions: Dict[str, ModelVersion] = None
    """All registered model versions."""

    def __post_init__(self):
        if self.versions is None:
            self.versions = {}


class ModelVersionManager:
    """Manages model versions with semantic versioning and registry."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the model version manager.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize registry
        self.registry = ModelRegistry()

        # Create directories
        self.models_dir = self.config.artifacts_dir / "models"
        self.registry_file = self.config.artifacts_dir / "model_registry.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def register_model(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
        pipeline_config: PipelineConfig,
        performance_metrics: Dict[str, Dict[str, float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model: Trained model to register.
            pipeline_config: Pipeline configuration used for training.
            performance_metrics: Performance metrics for the model.
            metadata: Additional metadata.

        Returns:
            ModelVersion object for the registered model.
        """
        # Generate new version number
        new_version_str = self._generate_version_number()

        # Save model to disk
        model_path = self._save_model(model, new_version_str)
        model_size = model_path.stat().st_size

        # Extract feature names
        feature_names = self._extract_feature_names(model)

        # Get model type
        model_type = getattr(model, 'model_type', 'ridge')

        # Create version object
        model_version = ModelVersion(
            version=new_version_str,
            created_at=datetime.now(),
            model_type=model_type,
            file_path=model_path,
            performance_metrics=performance_metrics,
            feature_names=feature_names,
            training_samples=sum(performance_metrics.get("train", {}).values()),
            model_size_bytes=model_size,
            metadata=metadata or {},
            is_active=False
        )

        # Add to registry
        self.registry.versions[new_version_str] = model_version

        # Save registry
        self._save_registry()

        self.logger.info(f"Registered model version {new_version_str} ({model_type})")

        return model_version

    def _generate_version_number(self) -> str:
        """Generate a new semantic version number.

        Returns:
            New version string.
        """
        if not self.registry.versions:
            return "1.0.0"

        # Get latest version
        latest_version = max(
            self.registry.versions.keys(),
            key=lambda v: version.parse(v)
        )

        # Increment patch version
        current_ver = version.parse(latest_version)
        new_ver = version.Version(f"{current_ver.major}.{current_ver.minor}.{current_ver.micro + 1}")

        return str(new_ver)

    def _save_model(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
        version_str: str
    ) -> Path:
        """Save model to disk.

        Args:
            model: Model to save.
            version_str: Version string for filename.

        Returns:
            Path to saved model file.
        """
        model_path = self.models_dir / f"model_v{version_str}.pkl"

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            # Fallback to joblib for complex models
            model_path = self.models_dir / f"model_v{version_str}.joblib"
            import joblib
            joblib.dump(model, model_path)

        return model_path

    def _extract_feature_names(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel]
    ) -> List[str]:
        """Extract feature names from model.

        Args:
            model: Model to extract features from.

        Returns:
            List of feature names.
        """
        # For RidgeRegressionModel, try to get from metadata
        if hasattr(model, 'feature_names'):
            return model.feature_names

        # For other models, return generic names
        return [f"feature_{i}" for i in range(50)]  # Reasonable default

    def activate_model_version(self, version_str: str) -> bool:
        """Activate a specific model version.

        Args:
            version_str: Version string to activate.

        Returns:
            True if activation was successful.
        """
        if version_str not in self.registry.versions:
            self.logger.error(f"Version {version_str} not found in registry")
            return False

        model_version = self.registry.versions[version_str]

        # Deactivate current version
        if self.registry.current_version and self.registry.current_version in self.registry.versions:
            self.registry.versions[self.registry.current_version].is_active = False

        # Activate new version
        model_version.is_active = True
        model_version.activated_at = datetime.now()
        self.registry.current_version = version_str

        # Save registry
        self._save_registry()

        self.logger.info(f"Activated model version {version_str}")
        return True

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version.

        Returns:
            Current ModelVersion or None if no version is active.
        """
        if not self.registry.current_version:
            return None

        return self.registry.versions.get(self.registry.current_version)

    def get_model_version(self, version_str: str) -> Optional[ModelVersion]:
        """Get a specific model version.

        Args:
            version_str: Version string to retrieve.

        Returns:
            ModelVersion object or None if not found.
        """
        return self.registry.versions.get(version_str)

    def get_model_metrics(self, version_str: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get performance metrics for a specific model version.

        Args:
            version_str: Version string.

        Returns:
            Performance metrics or None if not found.
        """
        model_version = self.get_model_version(version_str)
        if not model_version:
            return None

        return model_version.performance_metrics

    def load_model(self, version_str: str) -> Optional[Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel]]:
        """Load a model from disk.

        Args:
            version_str: Version string to load.

        Returns:
            Loaded model or None if not found.
        """
        model_version = self.get_model_version(version_str)
        if not model_version:
            return None

        try:
            with open(model_version.file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            try:
                # Fallback to joblib
                import joblib
                return joblib.load(model_version.file_path)
            except Exception as e2:
                self.logger.error(f"Failed to load model {version_str}: {e2}")
                return None

    def list_model_versions(
        self,
        active_only: bool = False,
        limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """List model versions.

        Args:
            active_only: Whether to return only active versions.
            limit: Maximum number of versions to return.

        Returns:
            List of ModelVersion objects.
        """
        versions = list(self.registry.versions.values())

        if active_only:
            versions = [v for v in versions if v.is_active]

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        if limit:
            versions = versions[:limit]

        return versions

    def delete_model_version(self, version_str: str, force: bool = False) -> bool:
        """Delete a model version.

        Args:
            version_str: Version string to delete.
            force: Whether to force deletion of active version.

        Returns:
            True if deletion was successful.
        """
        if version_str not in self.registry.versions:
            return False

        model_version = self.registry.versions[version_str]

        # Prevent deletion of active version unless forced
        if model_version.is_active and not force:
            self.logger.error(f"Cannot delete active model version {version_str}")
            return False

        try:
            # Delete model file
            if model_version.file_path.exists():
                model_version.file_path.unlink()

            # Remove from registry
            del self.registry.versions[version_str]

            # Update current version if necessary
            if self.registry.current_version == version_str:
                self.registry.current_version = None
                # Activate latest available version
                available_versions = list(self.registry.versions.keys())
                if available_versions:
                    latest = max(available_versions, key=lambda v: version.parse(v))
                    self.activate_model_version(latest)

            self._save_registry()
            self.logger.info(f"Deleted model version {version_str}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model version {version_str}: {e}")
            return False

    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping the most recent ones.

        Args:
            keep_versions: Number of recent versions to keep.

        Returns:
            Number of versions deleted.
        """
        if len(self.registry.versions) <= keep_versions:
            return 0

        # Get all versions sorted by creation date
        versions = list(self.registry.versions.values())
        versions.sort(key=lambda v: v.created_at, reverse=True)

        # Keep the most recent versions and current version
        versions_to_keep = set()
        versions_to_keep.update(v.version for v in versions[:keep_versions])
        if self.registry.current_version:
            versions_to_keep.add(self.registry.current_version)

        # Delete old versions
        deleted_count = 0
        for model_version in versions[keep_versions:]:
            if model_version.version not in versions_to_keep:
                if self.delete_model_version(model_version.version, force=True):
                    deleted_count += 1

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old model versions")
        else:
            self.logger.info("No old model versions to clean up")

        return deleted_count

    def compare_versions(
        self,
        version1_str: str,
        version2_str: str,
        metric: str = "mae"
    ) -> Optional[Dict[str, Any]]:
        """Compare two model versions.

        Args:
            version1_str: First version to compare.
            version2_str: Second version to compare.
            metric: Metric to use for comparison.

        Returns:
            Dictionary with comparison results or None if comparison failed.
        """
        v1 = self.get_model_version(version1_str)
        v2 = self.get_model_version(version2_str)

        if not v1 or not v2:
            return None

        # Get test metrics for comparison
        v1_metrics = v1.performance_metrics.get("test", {})
        v2_metrics = v2.performance_metrics.get("test", {})

        comparison_metric = v1_metrics.get(metric)
        if comparison_metric is None or v2_metrics.get(metric) is None:
            return None

        improvement = (comparison_metric - v2_metrics[metric]) / comparison_metric * 100

        return {
            "version1": version1_str,
            "version2": version2_str,
            "metric": metric,
            "version1_value": comparison_metric,
            "version2_value": v2_metrics[metric],
            "improvement_percent": improvement,
            "version1_better": improvement > 0,
            "version2_better": improvement < 0,
            "metadata": {
                "version1_created": v1.created_at.isoformat(),
                "version2_created": v2.created_at.isoformat(),
                "version1_type": v1.model_type,
                "version2_type": v2.model_type
            }
        }

    def get_version_summary(self) -> Dict[str, Any]:
        """Get summary of all model versions.

        Returns:
            Dictionary with version summary statistics.
        """
        if not self.registry.versions:
            return {"total_versions": 0, "active_version": None}

        versions = list(self.registry.versions.values())
        active_version = self.get_current_version()

        # Count by model type
        model_types = {}
        for v in versions:
            model_types[v.model_type] = model_types.get(v.model_type, 0) + 1

        # Calculate size statistics
        sizes = [v.model_size_bytes for v in versions]
        total_size = sum(sizes)
        avg_size = np.mean(sizes)

        # Get performance ranges
        test_metrics = []
        for v in versions:
            if "test" in v.performance_metrics:
                test_metrics.append(v.performance_metrics["test"])

        summary = {
            "total_versions": len(versions),
            "active_version": active_version.version if active_version else None,
            "model_types": model_types,
            "size_statistics": {
                "total_size_mb": total_size / (1024 * 1024),
                "average_size_mb": avg_size / (1024 * 1024),
                "min_size_mb": min(sizes) / (1024 * 1024),
                "max_size_mb": max(sizes) / (1024 * 1024)
            },
            "date_range": {
                "oldest": min(v.created_at for v in versions).isoformat(),
                "newest": max(v.created_at for v in versions).isoformat()
            }
        }

        # Add performance statistics if available
        if test_metrics:
            mae_values = [m.get("mae", 0) for m in test_metrics]
            r2_values = [m.get("r2", 0) for m in test_metrics]

            summary["performance_statistics"] = {
                "mae_range": {"min": min(mae_values), "max": max(mae_values)},
                "r2_range": {"min": min(r2_values), "max": max(r2_values)},
                "best_mae": min(mae_values),
                "best_r2": max(r2_values)
            }

        return summary

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)

                self.registry.current_version = data.get("current_version")

                versions = {}
                for version_str, version_data in data.get("versions", {}).items():
                    model_version = ModelVersion(
                        version=version_str,
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        model_type=version_data["model_type"],
                        file_path=Path(version_data["file_path"]),
                        performance_metrics=version_data["performance_metrics"],
                        feature_names=version_data["feature_names"],
                        training_samples=version_data["training_samples"],
                        model_size_bytes=version_data["model_size_bytes"],
                        metadata=version_data.get("metadata", {}),
                        is_active=version_data.get("is_active", False),
                        activated_at=datetime.fromisoformat(version_data["activated_at"]) if version_data.get("activated_at") else None
                    )
                    versions[version_str] = model_version

                self.registry.versions = versions

                self.logger.info(f"Loaded model registry with {len(versions)} versions")

            except Exception as e:
                self.logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            data = {
                "current_version": self.registry.current_version,
                "versions": {}
            }

            for version_str, model_version in self.registry.versions.items():
                data["versions"][version_str] = {
                    "created_at": model_version.created_at.isoformat(),
                    "model_type": model_version.model_type,
                    "file_path": str(model_version.file_path),
                    "performance_metrics": model_version.performance_metrics,
                    "feature_names": model_version.feature_names,
                    "training_samples": model_version.training_samples,
                    "model_size_bytes": model_version.model_size_bytes,
                    "metadata": model_version.metadata,
                    "is_active": model_version.is_active,
                    "activated_at": model_version.activated_at.isoformat() if model_version.activated_at else None
                }

            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def rollback_to_version(self, version_str: str) -> bool:
        """Rollback to a specific model version.

        Args:
            version_str: Version to rollback to.

        Returns:
            True if rollback was successful.
        """
        if version_str not in self.registry.versions:
            self.logger.error(f"Cannot rollback: version {version_str} not found")
            return False

        success = self.activate_model_version(version_str)
        if success:
            self.logger.info(f"Successfully rolled back to model version {version_str}")
        else:
            self.logger.error(f"Failed to rollback to model version {version_str}")

        return success
