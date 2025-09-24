"""Evaluation export utilities for Tableau integration and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
from numbers import Real
import json

import pandas as pd

from .evaluation import EvaluationResult, AdvancedEvaluator
from .pipeline import PipelineResult


def export_evaluation_for_tableau(
    pipeline_result: PipelineResult,
    output_dir: Path,
    include_predictions: bool = True
) -> Dict[str, Path]:
    """Export evaluation results in Tableau-friendly format.

    Args:
        pipeline_result: Pipeline result containing evaluation data
        output_dir: Directory to save exported files
        include_predictions: Whether to include prediction data

    Returns:
        Dictionary mapping export types to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    if not pipeline_result.evaluation:
        raise ValueError("Pipeline result does not contain evaluation data")

    # Export evaluation metrics
    metrics_df = pd.DataFrame([pipeline_result.evaluation.metrics])
    metrics_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    exported_files["metrics"] = metrics_path

    # Export position-specific metrics
    if pipeline_result.evaluation.position_metrics:
        position_metrics_list = []
        for position, metrics in pipeline_result.evaluation.position_metrics.items():
            row = {"position": position}
            row.update(metrics)
            position_metrics_list.append(row)

        position_df = pd.DataFrame(position_metrics_list)
        position_path = output_dir / "position_metrics.csv"
        position_df.to_csv(position_path, index=False)
        exported_files["position_metrics"] = position_path

    # Export feature importance
    if pipeline_result.evaluation.feature_importance:
        importance_df = pd.DataFrame([
            {"feature": feature, "importance": importance}
            for feature, importance in pipeline_result.evaluation.feature_importance.items()
        ])
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_path = output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        exported_files["feature_importance"] = importance_path

    # Export predictions with evaluation metadata
    if include_predictions and pipeline_result.predictions:
        enriched_predictions = []
        for split_name, pred_df in pipeline_result.predictions.items():
            if not pred_df.empty:
                enriched = pred_df.copy()
                enriched["split"] = split_name

                # Add reliability indicators
                enriched["reliability_score"] = pipeline_result.evaluation.reliability_scores.get("error_consistency", 0.5)
                enriched["overfitting_risk"] = pipeline_result.evaluation.diagnostics.get("overfitting", {}).get("overfitting_score", 0.0)

                enriched_predictions.append(enriched)

        if enriched_predictions:
            combined_predictions = pd.concat(enriched_predictions, ignore_index=True)
            predictions_path = output_dir / "predictions_with_evaluation.csv"
            combined_predictions.to_csv(predictions_path, index=False)
            exported_files["predictions"] = predictions_path

    # Export confidence intervals
    if pipeline_result.evaluation.confidence_intervals:
        ci_df = pd.DataFrame([
            {
                "metric": metric,
                "lower_bound": bounds[0],
                "upper_bound": bounds[1],
                "width": bounds[1] - bounds[0]
            }
            for metric, bounds in pipeline_result.evaluation.confidence_intervals.items()
        ])
        ci_path = output_dir / "confidence_intervals.csv"
        ci_df.to_csv(ci_path, index=False)
        exported_files["confidence_intervals"] = ci_path

    return exported_files


def generate_evaluation_report(
    pipeline_result: PipelineResult,
    output_dir: Optional[Path] = None,
    include_plots: bool = True
) -> str:
    """Generate a comprehensive evaluation report.

    Args:
        pipeline_result: Pipeline result containing evaluation data
        output_dir: Directory to save report files (optional)
        include_plots: Whether to generate calibration plots

    Returns:
        Complete evaluation report as markdown string
    """
    if not pipeline_result.evaluation:
        raise ValueError("Pipeline result does not contain evaluation data")

    # Generate the main report
    evaluator = AdvancedEvaluator(pipeline_result.config)
    report = evaluator.generate_evaluation_report(
        evaluation=pipeline_result.evaluation,
        model_name=pipeline_result.selected_model_type or "Model"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main report
        report_path = output_dir / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        # Generate calibration plots if requested
        if include_plots and pipeline_result.predictions:
            # Use test predictions for calibration analysis
            test_predictions = None
            for split_name, pred_df in pipeline_result.predictions.items():
                if split_name == "test" and not pred_df.empty:
                    test_predictions = pred_df
                    break

            if test_predictions is None:
                # Use first available predictions if no test split
                test_predictions = next(iter(pipeline_result.predictions.values()))

            if test_predictions is not None:
                try:
                    y_true = test_predictions["actual_points"].to_numpy()
                    y_pred = test_predictions["predicted_points"].to_numpy()

                    plot_data = evaluator.generate_calibration_plots(
                        y_true, y_pred, output_dir
                    )

                    # Add plot information to report
                    report += f"\n\n## Calibration Analysis\n\n"
                    scatter_path = plot_data.get('prediction_vs_actual', {}).get('plot_path', 'N/A')
                    report += f"Prediction vs actual chart: {scatter_path}\n\n"
                    bias_path = plot_data.get('bin_bias', {}).get('plot_path')
                    if bias_path:
                        report += f"Bias by bin chart: {bias_path}\n\n"

                    # Add calibration metrics summary
                    if pipeline_result.evaluation.calibration_metrics:
                        report += "### Calibration Metrics Summary\n\n"
                        report += "| Metric | Value |\n"
                        report += "|--------|-------|\n"
                        for metric, value in pipeline_result.evaluation.calibration_metrics.items():
                            formatted = f"{float(value):.4f}" if isinstance(value, Real) else str(value)
                            report += f"| {metric.upper()} | {formatted} |\n"

                except Exception as e:
                    report += f"\n\n## Calibration Analysis\n\n"
                    report += f"Warning: Could not generate calibration plots: {e}\n\n"

        # Export Tableau data
        try:
            tableau_files = export_evaluation_for_tableau(pipeline_result, output_dir)
            report += "\n\n## Tableau Export\n\n"
            report += "The following files have been exported for Tableau integration:\n\n"
            for export_type, file_path in tableau_files.items():
                report += f"- **{export_type}**: `{file_path.name}`\n"
        except Exception as e:
            report += f"\n\n## Tableau Export\n\n"
            report += f"Warning: Could not export Tableau data: {e}\n"

    return report


def create_model_comparison_dashboard(
    model_results: Dict[str, PipelineResult],
    output_dir: Path
) -> str:
    """Create a model comparison dashboard.

    Args:
        model_results: Dictionary of model names to pipeline results
        output_dir: Directory to save dashboard files

    Returns:
        Dashboard report as markdown string
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard = "# Model Comparison Dashboard\n\n"

    if not model_results:
        dashboard += "No model results provided for comparison.\n"
        return dashboard

    dashboard += f"Comparing {len(model_results)} models:\n\n"
    for model_name in model_results.keys():
        dashboard += f"- {model_name}\n"

    dashboard += "\n\n## Performance Summary\n\n"

    # Create comparison table
    dashboard += "| Model | MAE | RMSE | R² | Position Count |\n"
    dashboard += "|-------|-----|------|----|---------------|\n"

    model_summaries = []
    for model_name, result in model_results.items():
        if result.evaluation:
            metrics = result.evaluation.metrics
            mae = metrics.get('mae', 'N/A')
            rmse = metrics.get('rmse', 'N/A')
            r2 = metrics.get('r2', 'N/A')

            # Count total positions evaluated
            position_count = len(result.evaluation.position_metrics) if result.evaluation.position_metrics else 0

            dashboard += f"| {model_name} | {mae} | {rmse} | {r2} | {position_count} |\n"
            model_summaries.append({
                'name': model_name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })

    # Find best model by MAE
    if model_summaries:
        best_model = min(model_summaries, key=lambda x: x['mae'] if isinstance(x['mae'], (int, float)) else float('inf'))
        dashboard += f"\n\n**Best Model (by MAE):** {best_model['name']} (MAE: {best_model['mae']})\n\n"

    # Export comparison data for Tableau
    comparison_data = []
    for model_name, result in model_results.items():
        if result.evaluation:
            row = {
                'model': model_name,
                'selected_model': model_name == best_model['name'] if model_summaries else False
            }

            # Add metrics
            row.update(result.evaluation.metrics)

            # Add position metrics
            for position, pos_metrics in result.evaluation.position_metrics.items():
                pos_row = row.copy()
                pos_row['position'] = position
                pos_row.update(pos_metrics)
                comparison_data.append(pos_row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        dashboard += f"\n\n## Tableau Integration\n\n"
        dashboard += f"Model comparison data exported to: `{comparison_path.name}`\n\n"

        # Add key insights
        dashboard += "### Key Insights\n\n"

        # Compare model performance
        if len(model_summaries) >= 2:
            mae_improvement = None
            for summary in model_summaries:
                if summary['name'] != best_model['name']:
                    if isinstance(summary['mae'], (int, float)) and isinstance(best_model['mae'], (int, float)):
                        improvement = (summary['mae'] - best_model['mae']) / summary['mae'] * 100
                        dashboard += f"- {best_model['name']} shows {improvement".1f"}% improvement in MAE over {summary['name']}\n"

        # Position-specific analysis
        dashboard += "\n#### Position-Specific Performance\n\n"
        for model_name, result in model_results.items():
            if result.evaluation and result.evaluation.position_metrics:
                dashboard += f"**{model_name}:**\n"
                for position, metrics in result.evaluation.position_metrics.items():
                    mae = metrics.get('mae', 'N/A')
                    count = metrics.get('count', 0)
                    dashboard += f"- {position}: MAE = {mae}, Samples = {count}\n"

    dashboard += "\n\n## Recommendations\n\n"

    if model_summaries:
        dashboard += f"- **Recommended Model:** {best_model['name']}\n"
        dashboard += "- Consider position-specific performance when selecting models\n"
        dashboard += "- Review feature importance to understand model decisions\n"
        dashboard += "- Monitor calibration metrics for production deployment\n"

    return dashboard


def save_evaluation_artifacts(
    pipeline_result: PipelineResult,
    base_output_dir: Path
) -> Dict[str, Path]:
    """Save all evaluation artifacts to organized directory structure.

    Args:
        pipeline_result: Pipeline result with evaluation data
        base_output_dir: Base directory for saving artifacts

    Returns:
        Dictionary of saved file paths
    """
    if not pipeline_result.evaluation:
        raise ValueError("Pipeline result does not contain evaluation data")

    # Create organized directory structure
    artifacts_dir = base_output_dir / "evaluation_artifacts"
    tableau_dir = artifacts_dir / "tableau"
    reports_dir = artifacts_dir / "reports"

    # Export Tableau data
    tableau_files = export_evaluation_for_tableau(pipeline_result, tableau_dir)

    # Generate comprehensive report
    report = generate_evaluation_report(pipeline_result, reports_dir)

    # Save report
    report_path = reports_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    # Save evaluation metadata
    metadata = {
        "model_type": pipeline_result.selected_model_type,
        "feature_count": len(pipeline_result.features),
        "train_rows": pipeline_result.train_rows,
        "test_rows": pipeline_result.test_rows,
        "has_evaluation": True,
        "evaluation_components": [
            "core_metrics",
            "position_metrics",
            "confidence_intervals",
            "calibration_metrics",
            "feature_importance",
            "reliability_scores",
            "statistical_tests",
            "diagnostics",
            "validation_results"
        ]
    }

    metadata_path = artifacts_dir / "evaluation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    all_files = {
        **{f"tableau_{k}": v for k, v in tableau_files.items()},
        "report": report_path,
        "metadata": metadata_path
    }

    return all_files
