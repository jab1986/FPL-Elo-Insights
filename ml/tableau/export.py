"""Enhanced data export module for Tableau dashboard integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

from ..evaluation import EvaluationResult, AdvancedEvaluator
from ..pipeline import PipelineResult
from ..config import PipelineConfig


@dataclass
class TableauExportConfig:
    """Configuration for Tableau data exports."""

    # Export settings
    include_predictions: bool = True
    include_feature_importance: bool = True
    include_model_metadata: bool = True
    include_drift_analysis: bool = True

    # Data quality settings
    max_missing_threshold: float = 0.1
    outlier_std_threshold: float = 3.0

    # Business metrics
    price_points: List[float] = None
    value_thresholds: List[float] = None

    def __post_init__(self):
        if self.price_points is None:
            self.price_points = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
        if self.value_thresholds is None:
            self.value_thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


class TableauDataExporter:
    """Enhanced exporter for Tableau dashboard data integration."""

    def __init__(self, config: TableauExportConfig = None):
        self.config = config or TableauExportConfig()

    def export_comprehensive_dashboard_data(
        self,
        pipeline_result: PipelineResult,
        output_dir: Path,
        model_version: str = None,
        include_historical: bool = False
    ) -> Dict[str, Path]:
        """Export comprehensive data for all Tableau dashboards.

        Args:
            pipeline_result: Pipeline result with evaluation data
            output_dir: Directory to save exported files
            model_version: Version identifier for the model
            include_historical: Whether to include historical comparisons

        Returns:
            Dictionary mapping export types to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}

        if not pipeline_result.evaluation:
            raise ValueError("Pipeline result does not contain evaluation data")

        # Generate model version
        if model_version is None:
            model_version = f"{pipeline_result.selected_model_type or 'model'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Export model performance data
        performance_files = self._export_model_performance_data(
            pipeline_result, output_dir, model_version
        )
        exported_files.update(performance_files)

        # Export prediction analysis data
        if self.config.include_predictions and pipeline_result.predictions:
            prediction_files = self._export_prediction_analysis_data(
                pipeline_result, output_dir, model_version
            )
            exported_files.update(prediction_files)

        # Export data quality metrics
        if self.config.include_drift_analysis:
            quality_files = self._export_data_quality_data(
                pipeline_result, output_dir, model_version
            )
            exported_files.update(quality_files)

        # Export business impact data
        business_files = self._export_business_impact_data(
            pipeline_result, output_dir, model_version
        )
        exported_files.update(business_files)

        # Export feature importance and model metadata
        if self.config.include_feature_importance and pipeline_result.evaluation.feature_importance:
            metadata_files = self._export_model_metadata(
                pipeline_result, output_dir, model_version
            )
            exported_files.update(metadata_files)

        # Create dashboard configuration files
        config_files = self._create_dashboard_configurations(
            pipeline_result, output_dir, model_version
        )
        exported_files.update(config_files)

        # Export summary report
        summary_file = self._create_export_summary(
            exported_files, output_dir, model_version
        )
        exported_files["summary"] = summary_file

        return exported_files

    def _export_model_performance_data(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Export data for Model Performance Dashboard."""
        files = {}

        # Core metrics with confidence intervals
        metrics_df = self._create_metrics_dataframe(pipeline_result.evaluation)
        metrics_path = output_dir / "model_performance_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        files["performance_metrics"] = metrics_path

        # Position-specific performance
        if pipeline_result.evaluation.position_metrics:
            position_df = self._create_position_performance_dataframe(pipeline_result.evaluation)
            position_path = output_dir / "position_performance.csv"
            position_df.to_csv(position_path, index=False)
            files["position_metrics"] = position_path

        # Model comparison data (if available)
        if pipeline_result.model_comparison:
            comparison_df = self._create_model_comparison_dataframe(
                pipeline_result, model_version
            )
            comparison_path = output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            files["model_comparison"] = comparison_path

        # Cross-validation results
        if pipeline_result.cross_validation:
            cv_df = pd.DataFrame(pipeline_result.cross_validation)
            cv_path = output_dir / "cross_validation_results.csv"
            cv_df.to_csv(cv_path, index=False)
            files["cross_validation"] = cv_path

        return files

    def _export_prediction_analysis_data(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Export data for Prediction Analysis Dashboard."""
        files = {}

        # Combine all predictions
        all_predictions = []
        for split_name, pred_df in pipeline_result.predictions.items():
            if not pred_df.empty:
                enriched = pred_df.copy()
                enriched["split"] = split_name
                enriched["model_version"] = model_version
                all_predictions.append(enriched)

        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)

            # Add prediction analysis fields
            combined_predictions = self._enrich_prediction_data(
                combined_predictions, pipeline_result.evaluation
            )

            # Export predictions
            predictions_path = output_dir / "prediction_analysis.csv"
            combined_predictions.to_csv(predictions_path, index=False)
            files["predictions"] = predictions_path

            # Create calibration data
            calibration_df = self._create_calibration_dataframe(combined_predictions)
            if not calibration_df.empty:
                calibration_path = output_dir / "calibration_data.csv"
                calibration_df.to_csv(calibration_path, index=False)
                files["calibration"] = calibration_path

            # Create error distribution data
            error_df = self._create_error_distribution_dataframe(combined_predictions)
            error_path = output_dir / "error_distribution.csv"
            error_df.to_csv(error_path, index=False)
            files["error_distribution"] = error_path

        return files

    def _export_data_quality_data(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Export data for Data Quality Dashboard."""
        files = {}

        # Data coverage and completeness
        coverage_df = self._create_data_coverage_dataframe(pipeline_result)
        coverage_path = output_dir / "data_coverage.csv"
        coverage_df.to_csv(coverage_path, index=False)
        files["data_coverage"] = coverage_path

        # Feature statistics
        if pipeline_result.predictions:
            feature_stats_df = self._create_feature_statistics_dataframe(pipeline_result)
            feature_stats_path = output_dir / "feature_statistics.csv"
            feature_stats_df.to_csv(feature_stats_path, index=False)
            files["feature_statistics"] = feature_stats_path

        # Outlier analysis
        outlier_df = self._create_outlier_analysis_dataframe(pipeline_result)
        outlier_path = output_dir / "outlier_analysis.csv"
        outlier_df.to_csv(outlier_path, index=False)
        files["outlier_analysis"] = outlier_path

        # Data drift indicators
        drift_df = self._create_drift_analysis_dataframe(pipeline_result)
        drift_path = output_dir / "data_drift.csv"
        drift_df.to_csv(drift_path, index=False)
        files["data_drift"] = drift_path

        return files

    def _export_business_impact_data(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Export data for Business Impact Dashboard."""
        files = {}

        if pipeline_result.predictions:
            # Combine predictions for business analysis
            all_predictions = []
            for split_name, pred_df in pipeline_result.predictions.items():
                if not pred_df.empty:
                    enriched = pred_df.copy()
                    enriched["split"] = split_name
                    all_predictions.append(enriched)

            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)

                # Player recommendations with value metrics
                recommendations_df = self._create_player_recommendations_dataframe(combined_predictions)
                recommendations_path = output_dir / "player_recommendations.csv"
                recommendations_df.to_csv(recommendations_path, index=False)
                files["recommendations"] = recommendations_path

                # Value analysis by price points
                value_df = self._create_value_analysis_dataframe(combined_predictions)
                value_path = output_dir / "value_analysis.csv"
                value_df.to_csv(value_path, index=False)
                files["value_analysis"] = value_path

                # Risk assessment
                risk_df = self._create_risk_assessment_dataframe(
                    combined_predictions, pipeline_result.evaluation
                )
                risk_path = output_dir / "risk_assessment.csv"
                risk_df.to_csv(risk_path, index=False)
                files["risk_assessment"] = risk_path

                # Seasonal trends
                trends_df = self._create_seasonal_trends_dataframe(combined_predictions)
                trends_path = output_dir / "seasonal_trends.csv"
                trends_df.to_csv(trends_path, index=False)
                files["seasonal_trends"] = trends_path

        return files

    def _export_model_metadata(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Export model metadata and feature importance."""
        files = {}

        # Feature importance
        if pipeline_result.evaluation.feature_importance:
            importance_df = pd.DataFrame([
                {"feature": feature, "importance": importance, "model_version": model_version}
                for feature, importance in pipeline_result.evaluation.feature_importance.items()
            ])
            importance_df = importance_df.sort_values("importance", ascending=False)
            importance_path = output_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            files["feature_importance"] = importance_path

        # Model diagnostics
        if pipeline_result.evaluation.diagnostics:
            diagnostics_df = self._create_diagnostics_dataframe(
                pipeline_result.evaluation, model_version
            )
            diagnostics_path = output_dir / "model_diagnostics.csv"
            diagnostics_df.to_csv(diagnostics_path, index=False)
            files["diagnostics"] = diagnostics_path

        # Statistical tests results
        if pipeline_result.evaluation.statistical_tests:
            tests_df = self._create_statistical_tests_dataframe(
                pipeline_result.evaluation, model_version
            )
            tests_path = output_dir / "statistical_tests.csv"
            tests_df.to_csv(tests_path, index=False)
            files["statistical_tests"] = tests_path

        return files

    def _create_dashboard_configurations(
        self, pipeline_result: PipelineResult, output_dir: Path, model_version: str
    ) -> Dict[str, Path]:
        """Create configuration files for Tableau dashboards."""
        files = {}

        # Dashboard metadata
        metadata = {
            "model_version": model_version,
            "export_timestamp": datetime.now().isoformat(),
            "model_type": pipeline_result.selected_model_type or "ridge",
            "feature_count": len(pipeline_result.features),
            "train_rows": pipeline_result.train_rows,
            "test_rows": pipeline_result.test_rows,
            "has_evaluation": pipeline_result.evaluation is not None,
            "evaluation_metrics": list(pipeline_result.evaluation.metrics.keys()) if pipeline_result.evaluation else [],
            "available_dashboards": [
                "model_performance",
                "prediction_analysis",
                "data_quality",
                "business_impact"
            ],
            "tableau_version": "2023.1+",
            "refresh_required": True
        }

        metadata_path = output_dir / "dashboard_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        files["metadata"] = metadata_path

        # Calculated fields configuration for Tableau
        calc_fields_config = self._create_calculated_fields_config(pipeline_result)
        calc_fields_path = output_dir / "calculated_fields.json"
        with open(calc_fields_path, 'w') as f:
            json.dump(calc_fields_config, f, indent=2)
        files["calculated_fields"] = calc_fields_path

        # Dashboard actions and filters
        actions_config = self._create_dashboard_actions_config()
        actions_path = output_dir / "dashboard_actions.json"
        with open(actions_path, 'w') as f:
            json.dump(actions_config, f, indent=2)
        files["dashboard_actions"] = actions_path

        return files

    def _create_export_summary(
        self, exported_files: Dict[str, Path], output_dir: Path, model_version: str
    ) -> Path:
        """Create a summary of all exported files."""
        summary = {
            "model_version": model_version,
            "export_timestamp": datetime.now().isoformat(),
            "total_files": len(exported_files),
            "files_by_category": {},
            "dashboard_readiness": {
                "model_performance": "performance_metrics" in exported_files,
                "prediction_analysis": "predictions" in exported_files,
                "data_quality": "data_coverage" in exported_files,
                "business_impact": "recommendations" in exported_files
            }
        }

        # Group files by category
        categories = {
            "performance": ["performance_metrics", "position_metrics", "model_comparison", "cross_validation"],
            "predictions": ["predictions", "calibration", "error_distribution"],
            "quality": ["data_coverage", "feature_statistics", "outlier_analysis", "data_drift"],
            "business": ["recommendations", "value_analysis", "risk_assessment", "seasonal_trends"],
            "metadata": ["feature_importance", "diagnostics", "statistical_tests", "metadata", "calculated_fields", "dashboard_actions"]
        }

        for category, file_patterns in categories.items():
            category_files = [name for name in exported_files.keys() if any(pattern in name for pattern in file_patterns)]
            if category_files:
                summary["files_by_category"][category] = category_files

        summary_path = output_dir / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary_path

    # Helper methods for data enrichment
    def _create_metrics_dataframe(self, evaluation: EvaluationResult) -> pd.DataFrame:
        """Create metrics dataframe with confidence intervals."""
        metrics_data = []

        for metric_name, value in evaluation.metrics.items():
            row = {
                "metric": metric_name,
                "value": float(value),
                "model_version": "current"
            }

            # Add confidence intervals if available
            if metric_name in evaluation.confidence_intervals:
                ci = evaluation.confidence_intervals[metric_name]
                row["ci_lower"] = float(ci[0])
                row["ci_upper"] = float(ci[1])
                row["ci_width"] = float(ci[1] - ci[0])
            else:
                row["ci_lower"] = None
                row["ci_upper"] = None
                row["ci_width"] = None

            metrics_data.append(row)

        return pd.DataFrame(metrics_data)

    def _create_position_performance_dataframe(self, evaluation: EvaluationResult) -> pd.DataFrame:
        """Create position-specific performance dataframe."""
        position_data = []

        for position, metrics in evaluation.position_metrics.items():
            row = {"position": position}
            row.update({k: float(v) for k, v in metrics.items()})
            position_data.append(row)

        return pd.DataFrame(position_data)

    def _create_model_comparison_dataframe(
        self, pipeline_result: PipelineResult, model_version: str
    ) -> pd.DataFrame:
        """Create model comparison dataframe."""
        comparison_data = []

        if pipeline_result.model_comparison:
            for model_type, metrics in pipeline_result.model_comparison.items():
                row = {
                    "model_type": model_type,
                    "selected": model_type == (pipeline_result.selected_model_type or "ridge")
                }
                row.update({k: float(v) for k, v in metrics.items()})
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def _enrich_prediction_data(
        self, predictions: pd.DataFrame, evaluation: EvaluationResult
    ) -> pd.DataFrame:
        """Enrich prediction data with additional metrics."""
        enriched = predictions.copy()

        # Add error metrics
        enriched["absolute_error"] = abs(enriched["actual_points"] - enriched["predicted_points"])
        enriched["squared_error"] = (enriched["actual_points"] - enriched["predicted_points"]) ** 2
        enriched["error_percentage"] = enriched["absolute_error"] / (enriched["actual_points"] + 1e-8) * 100

        # Add prediction confidence indicators
        if evaluation.reliability_scores:
            enriched["error_consistency"] = evaluation.reliability_scores.get("error_consistency", 0.5)
            enriched["prediction_stability"] = evaluation.reliability_scores.get("prediction_stability", 0.5)

        # Add overfitting risk
        if evaluation.diagnostics and "overfitting" in evaluation.diagnostics:
            enriched["overfitting_risk"] = evaluation.diagnostics["overfitting"].get("overfitting_score", 0.0)

        return enriched

    def _create_calibration_dataframe(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create calibration analysis dataframe."""
        # Bin predictions for calibration analysis
        predictions = predictions.dropna(subset=["actual_points", "predicted_points"])

        if predictions.empty:
            return pd.DataFrame()

        # Create calibration bins
        n_bins = 10
        predictions = predictions.sort_values("predicted_points")
        predictions["calibration_bin"] = pd.qcut(predictions["predicted_points"], n_bins, duplicates="drop")

        calibration_data = []
        for bin_name, bin_group in predictions.groupby("calibration_bin", observed=False):
            calibration_data.append({
                "bin_min": float(bin_group["predicted_points"].min()),
                "bin_max": float(bin_group["predicted_points"].max()),
                "bin_mean_predicted": float(bin_group["predicted_points"].mean()),
                "bin_mean_actual": float(bin_group["actual_points"].mean()),
                "bin_count": int(len(bin_group)),
                "bin_calibration_error": float(abs(bin_group["predicted_points"].mean() - bin_group["actual_points"].mean()))
            })

        return pd.DataFrame(calibration_data)

    def _create_error_distribution_dataframe(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create error distribution analysis dataframe."""
        error_data = []

        # Error by position
        if "position" in predictions.columns:
            for position in predictions["position"].unique():
                pos_data = predictions[predictions["position"] == position]
                if not pos_data.empty:
                    error_data.append({
                        "category": "position",
                        "category_value": position,
                        "mean_error": float(pos_data["absolute_error"].mean()),
                        "median_error": float(pos_data["absolute_error"].median()),
                        "error_std": float(pos_data["absolute_error"].std()),
                        "sample_count": int(len(pos_data))
                    })

        # Error by gameweek
        if "gameweek" in predictions.columns:
            for gameweek in sorted(predictions["gameweek"].unique())[:20]:  # Last 20 gameweeks
                gw_data = predictions[predictions["gameweek"] == gameweek]
                if not gw_data.empty:
                    error_data.append({
                        "category": "gameweek",
                        "category_value": int(gameweek),
                        "mean_error": float(gw_data["absolute_error"].mean()),
                        "median_error": float(gw_data["absolute_error"].median()),
                        "error_std": float(gw_data["absolute_error"].std()),
                        "sample_count": int(len(gw_data))
                    })

        return pd.DataFrame(error_data)

    def _create_data_coverage_dataframe(self, pipeline_result: PipelineResult) -> pd.DataFrame:
        """Create data coverage analysis dataframe."""
        coverage_data = []

        # Analyze predictions for missing data patterns
        if pipeline_result.predictions:
            for split_name, pred_df in pipeline_result.predictions.items():
                if not pred_df.empty:
                    total_rows = len(pred_df)

                    # Check for missing values in key columns
                    key_columns = ["actual_points", "predicted_points"]
                    for col in key_columns:
                        if col in pred_df.columns:
                            missing_count = pred_df[col].isna().sum()
                            coverage_data.append({
                                "split": split_name,
                                "metric": f"{col}_missing",
                                "value": float(missing_count / total_rows * 100),
                                "count": int(missing_count),
                                "total": int(total_rows)
                            })

        return pd.DataFrame(coverage_data)

    def _create_feature_statistics_dataframe(self, pipeline_result: PipelineResult) -> pd.DataFrame:
        """Create feature statistics dataframe."""
        feature_stats = []

        # Analyze feature distributions from predictions
        if pipeline_result.predictions:
            # Use test set for feature analysis if available
            test_df = None
            for split_name, pred_df in pipeline_result.predictions.items():
                if split_name == "test" and not pred_df.empty:
                    test_df = pred_df
                    break

            if test_df is None:
                test_df = next(iter(pipeline_result.predictions.values()))

            # Analyze numeric columns
            numeric_columns = test_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ["actual_points", "predicted_points", "absolute_error", "squared_error", "error_percentage"]:
                    stats = {
                        "feature": col,
                        "mean": float(test_df[col].mean()),
                        "median": float(test_df[col].median()),
                        "std": float(test_df[col].std()),
                        "min": float(test_df[col].min()),
                        "max": float(test_df[col].max()),
                        "missing_count": int(test_df[col].isna().sum()),
                        "missing_percentage": float(test_df[col].isna().sum() / len(test_df) * 100)
                    }
                    feature_stats.append(stats)

        return pd.DataFrame(feature_stats)

    def _create_outlier_analysis_dataframe(self, pipeline_result: PipelineResult) -> pd.DataFrame:
        """Create outlier analysis dataframe."""
        outlier_data = []

        if pipeline_result.predictions:
            test_df = None
            for split_name, pred_df in pipeline_result.predictions.items():
                if split_name == "test" and not pred_df.empty:
                    test_df = pred_df
                    break

            if test_df is None:
                test_df = next(iter(pipeline_result.predictions.values()))

            # Analyze outliers in key metrics
            numeric_columns = ["actual_points", "predicted_points", "absolute_error"]
            for col in numeric_columns:
                if col in test_df.columns:
                    values = test_df[col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()

                        # Count outliers using 3-sigma rule
                        outlier_count = ((values - mean_val).abs() > self.config.outlier_std_threshold * std_val).sum()

                        outlier_data.append({
                            "metric": col,
                            "mean": float(mean_val),
                            "std": float(std_val),
                            "outlier_threshold": float(self.config.outlier_std_threshold * std_val),
                            "outlier_count": int(outlier_count),
                            "outlier_percentage": float(outlier_count / len(values) * 100),
                            "total_count": int(len(values))
                        })

        return pd.DataFrame(outlier_data)

    def _create_drift_analysis_dataframe(self, pipeline_result: PipelineResult) -> pd.DataFrame:
        """Create data drift analysis dataframe."""
        drift_data = []

        # Simple drift indicators based on available data
        if pipeline_result.predictions:
            # Compare train vs test distributions
            train_df = None
            test_df = None

            for split_name, pred_df in pipeline_result.predictions.items():
                if split_name == "train" and not pred_df.empty:
                    train_df = pred_df
                elif split_name == "test" and not pred_df.empty:
                    test_df = pred_df

            if train_df is not None and test_df is not None:
                numeric_columns = ["actual_points", "predicted_points"]

                for col in numeric_columns:
                    if col in train_df.columns and col in test_df.columns:
                        train_mean = train_df[col].mean()
                        test_mean = test_df[col].mean()
                        train_std = train_df[col].std()
                        test_std = test_df[col].std()

                        # Calculate drift metrics
                        mean_drift = abs(test_mean - train_mean) / (train_std + 1e-8)
                        std_drift = abs(test_std - train_std) / (train_std + 1e-8)

                        drift_data.append({
                            "feature": col,
                            "train_mean": float(train_mean),
                            "test_mean": float(test_mean),
                            "train_std": float(train_std),
                            "test_std": float(test_std),
                            "mean_drift": float(mean_drift),
                            "std_drift": float(std_drift),
                            "significant_drift": bool(mean_drift > 0.5 or std_drift > 0.5)
                        })

        return pd.DataFrame(drift_data)

    def _create_player_recommendations_dataframe(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create player recommendations dataframe."""
        recommendations = []

        # Group by player and calculate value metrics
        if "player_id" in predictions.columns:
            player_groups = predictions.groupby("player_id")

            for player_id, group in player_groups:
                if len(group) >= 3:  # Minimum games for recommendation
                    avg_actual = group["actual_points"].mean()
                    avg_predicted = group["predicted_points"].mean()
                    consistency = 1.0 / (1.0 + group["actual_points"].std())  # Higher is more consistent

                    # Calculate value score (simplified)
                    # In real implementation, this would use player price data
                    value_score = avg_actual * consistency

                    recommendations.append({
                        "player_id": player_id,
                        "avg_actual_points": float(avg_actual),
                        "avg_predicted_points": float(avg_predicted),
                        "consistency_score": float(consistency),
                        "value_score": float(value_score),
                        "games_analyzed": int(len(group)),
                        "recommendation_rank": 0  # To be calculated later
                    })

        # Sort by value score and assign ranks
        rec_df = pd.DataFrame(recommendations)
        if not rec_df.empty:
            rec_df = rec_df.sort_values("value_score", ascending=False)
            rec_df["recommendation_rank"] = range(1, len(rec_df) + 1)

        return rec_df

    def _create_value_analysis_dataframe(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create value analysis dataframe by price points."""
        value_data = []

        for price_point in self.config.price_points:
            # In real implementation, this would use actual player price data
            # For now, create synthetic analysis based on points
            price_predictions = predictions.copy()
            price_predictions["price_category"] = f"£{price_point}M"

            if not price_predictions.empty:
                value_data.append({
                    "price_point": float(price_point),
                    "price_category": f"£{price_point}M",
                    "avg_points": float(price_predictions["actual_points"].mean()),
                    "avg_predicted": float(price_predictions["predicted_points"].mean()),
                    "points_per_million": float(price_predictions["actual_points"].mean() / price_point),
                    "sample_count": int(len(price_predictions))
                })

        return pd.DataFrame(value_data)

    def _create_risk_assessment_dataframe(
        self, predictions: pd.DataFrame, evaluation: EvaluationResult
    ) -> pd.DataFrame:
        """Create risk assessment dataframe."""
        risk_data = []

        # Risk by position
        if "position" in predictions.columns:
            for position in predictions["position"].unique():
                pos_data = predictions[predictions["position"] == position]
                if not pos_data.empty:
                    error_std = pos_data["absolute_error"].std()
                    error_mean = pos_data["absolute_error"].mean()

                    risk_data.append({
                        "risk_category": "position",
                        "risk_factor": position,
                        "error_mean": float(error_mean),
                        "error_std": float(error_std),
                        "risk_score": float(error_mean + error_std),  # Combined risk metric
                        "sample_count": int(len(pos_data))
                    })

        # Risk by prediction confidence
        if "absolute_error" in predictions.columns:
            high_error = predictions[predictions["absolute_error"] > predictions["absolute_error"].quantile(0.75)]
            low_error = predictions[predictions["absolute_error"] <= predictions["absolute_error"].quantile(0.25)]

            risk_data.append({
                "risk_category": "prediction_confidence",
                "risk_factor": "high_risk_predictions",
                "error_mean": float(high_error["absolute_error"].mean()) if not high_error.empty else 0.0,
                "error_std": float(high_error["absolute_error"].std()) if not high_error.empty else 0.0,
                "risk_score": 1.0,  # High risk by definition
                "sample_count": int(len(high_error))
            })

            risk_data.append({
                "risk_category": "prediction_confidence",
                "risk_factor": "low_risk_predictions",
                "error_mean": float(low_error["absolute_error"].mean()) if not low_error.empty else 0.0,
                "error_std": float(low_error["absolute_error"].std()) if not low_error.empty else 0.0,
                "risk_score": 0.0,  # Low risk by definition
                "sample_count": int(len(low_error))
            })

        return pd.DataFrame(risk_data)

    def _create_seasonal_trends_dataframe(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal trends dataframe."""
        trends_data = []

        # Trends by gameweek
        if "gameweek" in predictions.columns:
            for gameweek in sorted(predictions["gameweek"].unique()):
                gw_data = predictions[predictions["gameweek"] == gameweek]
                if not gw_data.empty:
                    trends_data.append({
                        "period_type": "gameweek",
                        "period": int(gameweek),
                        "avg_actual_points": float(gw_data["actual_points"].mean()),
                        "avg_predicted_points": float(gw_data["predicted_points"].mean()),
                        "avg_absolute_error": float(gw_data["absolute_error"].mean()),
                        "sample_count": int(len(gw_data))
                    })

        # Trends by season
        if "season" in predictions.columns:
            for season in predictions["season"].unique():
                season_data = predictions[predictions["season"] == season]
                if not season_data.empty:
                    trends_data.append({
                        "period_type": "season",
                        "period": season,
                        "avg_actual_points": float(season_data["actual_points"].mean()),
                        "avg_predicted_points": float(season_data["predicted_points"].mean()),
                        "avg_absolute_error": float(season_data["absolute_error"].mean()),
                        "sample_count": int(len(season_data))
                    })

        return pd.DataFrame(trends_data)

    def _create_diagnostics_dataframe(
        self, evaluation: EvaluationResult, model_version: str
    ) -> pd.DataFrame:
        """Create model diagnostics dataframe."""
        diagnostics_data = []

        if evaluation.diagnostics:
            for diagnostic_type, diagnostic_info in evaluation.diagnostics.items():
                if isinstance(diagnostic_info, dict):
                    for metric, value in diagnostic_info.items():
                        diagnostics_data.append({
                            "diagnostic_type": diagnostic_type,
                            "metric": metric,
                            "value": float(value) if isinstance(value, (int, float)) else str(value),
                            "model_version": model_version
                        })

        return pd.DataFrame(diagnostics_data)

    def _create_statistical_tests_dataframe(
        self, evaluation: EvaluationResult, model_version: str
    ) -> pd.DataFrame:
        """Create statistical tests dataframe."""
        tests_data = []

        if evaluation.statistical_tests:
            for test_name, test_result in evaluation.statistical_tests.items():
                if isinstance(test_result, dict):
                    for metric, value in test_result.items():
                        tests_data.append({
                            "test_name": test_name,
                            "metric": metric,
                            "value": float(value) if isinstance(value, (int, float)) else str(value),
                            "model_version": model_version
                        })

        return pd.DataFrame(tests_data)

    def _create_calculated_fields_config(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Create calculated fields configuration for Tableau."""
        return {
            "calculated_fields": [
                {
                    "name": "Points per Million",
                    "formula": "SUM([actual_points]) / AVG([price])",
                    "description": "Player points normalized by cost"
                },
                {
                    "name": "Prediction Accuracy",
                    "formula": "1 - ABS([actual_points] - [predicted_points]) / ([actual_points] + 1)",
                    "description": "Accuracy measure for individual predictions"
                },
                {
                    "name": "Error Category",
                    "formula": "IF ABS([actual_points] - [predicted_points]) <= 2 THEN 'Low Error' ELSEIF ABS([actual_points] - [predicted_points]) <= 5 THEN 'Medium Error' ELSE 'High Error' END",
                    "description": "Categorize prediction errors"
                },
                {
                    "name": "Value Score",
                    "formula": "SUM([actual_points]) * (1 / (1 + STDEV([actual_points])))",
                    "description": "Combined points and consistency score"
                },
                {
                    "name": "Model Confidence",
                    "formula": "1 - [overfitting_risk] * [error_consistency]",
                    "description": "Overall model confidence indicator"
                }
            ]
        }

    def _create_dashboard_actions_config(self) -> Dict[str, Any]:
        """Create dashboard actions and filters configuration."""
        return {
            "dashboard_actions": [
                {
                    "name": "Position Filter",
                    "type": "filter",
                    "source_field": "position",
                    "description": "Filter by player position"
                },
                {
                    "name": "Gameweek Range",
                    "type": "filter",
                    "source_field": "gameweek",
                    "description": "Filter by gameweek range"
                },
                {
                    "name": "Model Comparison",
                    "type": "parameter",
                    "values": ["ridge", "position_specific", "deep_ensemble"],
                    "description": "Compare different model types"
                },
                {
                    "name": "Error Threshold",
                    "type": "parameter",
                    "default_value": 3,
                    "description": "Filter predictions by error threshold"
                }
            ],
            "dashboard_layout": {
                "model_performance": {
                    "tabs": ["Overview", "Position Comparison", "Model Comparison", "Feature Importance"]
                },
                "prediction_analysis": {
                    "tabs": ["Prediction Accuracy", "Calibration Curves", "Confidence Intervals", "Error Distribution"]
                },
                "data_quality": {
                    "tabs": ["Data Coverage", "Feature Statistics", "Outlier Detection", "Data Drift"]
                },
                "business_impact": {
                    "tabs": ["Player Recommendations", "Value Analysis", "Risk Assessment", "Seasonal Trends"]
                }
            }
        }


def export_tableau_dashboard_data(
    pipeline_result: PipelineResult,
    output_dir: Path,
    model_version: str = None,
    config: TableauExportConfig = None
) -> Dict[str, Path]:
    """Convenience function to export comprehensive Tableau dashboard data.

    Args:
        pipeline_result: Pipeline result with evaluation data
        output_dir: Directory to save exported files
        model_version: Version identifier for the model
        config: Export configuration options

    Returns:
        Dictionary mapping export types to file paths
    """
    exporter = TableauDataExporter(config)
    return exporter.export_comprehensive_dashboard_data(
        pipeline_result, output_dir, model_version
    )
