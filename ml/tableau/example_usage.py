#!/usr/bin/env python3
"""
Example usage of Tableau dashboard integration for FPL ML models.

This script demonstrates how to:
1. Run the ML pipeline with evaluation
2. Export comprehensive Tableau dashboard data
3. Configure automatic integration
4. Set up scheduled exports

Run with: python -m ml.tableau.example_usage
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.config import PipelineConfig
from ml.pipeline import PointsPredictionPipeline
from ml.tableau.export import TableauDataExporter, TableauExportConfig
from ml.tableau.pipeline_integration import (
    TableauPipelineIntegration,
    TableauIntegrationConfig,
    export_tableau_data_from_pipeline
)


def example_basic_export():
    """Example 1: Basic export after pipeline run."""
    print("🚀 Example 1: Basic Tableau Export")
    print("-" * 50)

    # Configure pipeline for evaluation
    config = PipelineConfig()
    config.cross_validation_folds = 3  # Enable cross-validation
    config.holdout_gameweeks = 3        # Reserve gameweeks for testing

    # Run pipeline
    print("Training model and generating evaluation data...")
    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    # Export Tableau data
    print("Exporting data for Tableau dashboards...")
    exported_files = export_tableau_data_from_pipeline(result)

    print(f"✅ Successfully exported {len(exported_files)} files:")
    for file_type, file_path in exported_files.items():
        print(f"   • {file_type}: {file_path.name}")

    return exported_files


def example_advanced_export():
    """Example 2: Advanced export with custom configuration."""
    print("\n🔧 Example 2: Advanced Export Configuration")
    print("-" * 50)

    # Custom export configuration
    export_config = TableauExportConfig(
        include_predictions=True,
        include_feature_importance=True,
        include_model_metadata=True,
        include_drift_analysis=True,
        max_missing_threshold=0.05,  # Stricter quality control
        outlier_std_threshold=2.5,   # More sensitive outlier detection
        price_points=[4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]
    )

    # Run pipeline
    config = PipelineConfig()
    config.model_comparison = True  # Compare multiple model types
    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    # Custom export
    exporter = TableauDataExporter(export_config)
    output_dir = Path("ml/artifacts/tableau/custom_export")
    exported_files = exporter.export_comprehensive_dashboard_data(
        pipeline_result=result,
        output_dir=output_dir,
        model_version="advanced_example_v1"
    )

    print(f"✅ Advanced export completed with {len(exported_files)} files:")
    for file_type, file_path in exported_files.items():
        print(f"   • {file_type}: {file_path.name}")

    return exported_files


def example_pipeline_integration():
    """Example 3: Automatic pipeline integration."""
    print("\n⚡ Example 3: Automatic Pipeline Integration")
    print("-" * 50)

    # Configure integration
    integration_config = TableauIntegrationConfig(
        auto_export_enabled=True,
        export_after_training=True,
        export_after_evaluation=True,
        base_output_dir=Path("ml/artifacts/tableau/integration_test"),
        keep_export_history=5,
        min_model_performance=0.6,      # Require good model performance
        max_error_threshold=8.0,        # Max acceptable MAE
        enable_notifications=False       # Disable for demo
    )

    # Set up integration
    integration = TableauPipelineIntegration(integration_config)

    # Integrate with pipeline
    config = PipelineConfig()
    integration.integrate_with_pipeline(config)

    # Run pipeline with automatic exports
    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    # Automatic export will happen
    exported_files = integration.export_after_training(result)

    print(f"✅ Integration export completed with {len(exported_files)} files:")
    for file_type, file_path in exported_files.items():
        print(f"   • {file_type}: {file_path.name}")

    return exported_files


def example_quality_gates():
    """Example 4: Quality gate enforcement."""
    print("\n🛡️ Example 4: Quality Gate Enforcement")
    print("-" * 50)

    # Strict quality configuration
    integration_config = TableauIntegrationConfig(
        min_model_performance=0.7,      # High R² requirement
        max_error_threshold=6.0,        # Low MAE requirement
        require_evaluation=True
    )

    integration = TableauPipelineIntegration(integration_config)

    # Run pipeline
    config = PipelineConfig()
    config.cross_validation_folds = 5
    pipeline = PointsPredictionPipeline(config)
    result = pipeline.run()

    # Check quality gates
    quality_passed = integration._check_quality_gates(result)

    if quality_passed:
        print("✅ Quality gates passed - proceeding with export")
        exported_files = integration.export_after_training(result)
        print(f"   Exported {len(exported_files)} files")
    else:
        print("❌ Quality gates failed - export blocked")
        integration._log_quality_issues(result)
        print("   See quality_issues.jsonl for details")

    return quality_passed


def main():
    """Run all examples."""
    print("🎯 FPL Tableau Dashboard Integration Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the Tableau integration.")
    print("Each example builds on the previous one with increasing complexity.\n")

    try:
        # Example 1: Basic export
        basic_files = example_basic_export()

        # Example 2: Advanced export
        advanced_files = example_advanced_export()

        # Example 3: Pipeline integration
        integration_files = example_pipeline_integration()

        # Example 4: Quality gates
        quality_ok = example_quality_gates()

        print("\n🎉 All examples completed successfully!")
        print("\n📁 Files created:")
        all_files = set()
        for file_dict in [basic_files, advanced_files, integration_files]:
            all_files.update(file_dict.keys())

        for file_type in sorted(all_files):
            print(f"   • {file_type}")

        print("\n📊 Next steps:")
        print("   1. Open Tableau Desktop")
        print("   2. Connect to the exported CSV files")
        print("   3. Import dashboard configurations from the JSON files")
        print("   4. Explore your ML model performance!")

        print("\n📖 For detailed documentation, see:")
        print("   • ml/tableau/README.md")
        print("   • ml/tableau/setup_guide.md")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nThis might be due to:")
        print("   • Missing data files in data/ directory")
        print("   • Insufficient training data")
        print("   • Model configuration issues")
        print("\nTry running the basic pipeline first:")
        print("   python -m ml.pipeline")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
