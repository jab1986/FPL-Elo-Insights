# Tableau Dashboard Integration for FPL ML Models

This directory contains comprehensive Tableau dashboard configurations and data export utilities for monitoring and analyzing Fantasy Premier League (FPL) machine learning models.

## 📊 Available Dashboards

### 1. Model Performance Dashboard
**File**: `model_performance_dashboard.json`

Comprehensive model performance monitoring including:
- **Overview Tab**: High-level metrics (MAE, RMSE, R²) across all models
- **Position Comparison Tab**: Position-specific performance breakdowns
- **Model Comparison Tab**: Side-by-side comparison of different architectures
- **Feature Importance Tab**: Visual ranking of feature contributions

### 2. Prediction Analysis Dashboard
**File**: `prediction_analysis_dashboard.json`

Detailed prediction quality analysis featuring:
- **Prediction Accuracy Tab**: Actual vs predicted points with error analysis
- **Calibration Curves Tab**: Model calibration assessment
- **Confidence Intervals Tab**: Prediction uncertainty visualization
- **Error Distribution Tab**: Error patterns by position and gameweek

### 3. Data Quality Dashboard
**File**: `data_quality_dashboard.json`

Data quality monitoring and drift detection including:
- **Data Coverage Tab**: Missing data patterns and completeness analysis
- **Feature Statistics Tab**: Feature distributions and correlations
- **Outlier Detection Tab**: Identification of unusual patterns
- **Data Drift Tab**: Monitoring for changes in data distributions

### 4. Business Impact Dashboard
**File**: `business_impact_dashboard.json`

Business value analysis and player recommendations featuring:
- **Player Recommendations Tab**: Top performers by position and value
- **Value Analysis Tab**: Points per million analysis
- **Risk Assessment Tab**: Prediction confidence and reliability
- **Seasonal Trends Tab**: Performance patterns over time

## 🚀 Quick Start

### Prerequisites

1. **Tableau Desktop** 2023.1 or later
2. **Python** 3.8+ with required packages
3. **FPL ML Pipeline** configured and trained

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Tableau Integration**
   ```python
   from ml.tableau.pipeline_integration import setup_tableau_integration
   from ml.config import PipelineConfig

   # Configure pipeline with Tableau integration
   config = PipelineConfig()
   tableau_integration = setup_tableau_integration(config)
   ```

### Basic Usage

#### Automatic Export After Training

```python
from ml.pipeline import PointsPredictionPipeline
from ml.tableau.pipeline_integration import export_tableau_data_from_pipeline

# Run ML pipeline
pipeline = PointsPredictionPipeline(config)
result = pipeline.run()

# Export Tableau data automatically
exported_files = export_tableau_data_from_pipeline(result)

print(f"Exported {len(exported_files)} files for Tableau integration")
```

#### Custom Export Configuration

```python
from ml.tableau.export import TableauDataExporter, TableauExportConfig
from pathlib import Path

# Configure export settings
export_config = TableauExportConfig(
    include_predictions=True,
    include_feature_importance=True,
    include_model_metadata=True,
    include_drift_analysis=True
)

# Create exporter
exporter = TableauDataExporter(export_config)

# Export comprehensive dashboard data
output_dir = Path("ml/artifacts/tableau/v1_20241201_120000")
exported_files = exporter.export_comprehensive_dashboard_data(
    pipeline_result=result,
    output_dir=output_dir,
    model_version="v1_20241201_120000"
)
```

## 📁 Directory Structure

```
ml/tableau/
├── export.py                           # Core data export functionality
├── pipeline_integration.py             # ML pipeline integration
├── README.md                          # This documentation
├── model_performance_dashboard.json    # Model performance dashboard config
├── prediction_analysis_dashboard.json  # Prediction analysis dashboard config
├── data_quality_dashboard.json         # Data quality dashboard config
└── business_impact_dashboard.json      # Business impact dashboard config
```

## 🔧 Configuration

### Export Configuration

The `TableauExportConfig` class controls export behavior:

```python
from ml.tableau.export import TableauExportConfig

config = TableauExportConfig(
    # Data inclusion settings
    include_predictions=True,      # Include prediction analysis data
    include_feature_importance=True, # Include feature importance rankings
    include_model_metadata=True,   # Include model diagnostics
    include_drift_analysis=True,   # Include data drift detection

    # Data quality thresholds
    max_missing_threshold=0.1,     # Max acceptable missing data (10%)
    outlier_std_threshold=3.0,     # Outlier detection threshold (3σ)

    # Business analysis settings
    price_points=[4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
    value_thresholds=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
)
```

### Pipeline Integration Configuration

The `TableauIntegrationConfig` controls automatic integration:

```python
from ml.tableau.pipeline_integration import TableauIntegrationConfig

integration_config = TableauIntegrationConfig(
    # Export triggers
    auto_export_enabled=True,           # Enable automatic exports
    export_after_training=True,         # Export after model training
    export_after_evaluation=True,       # Export after evaluation

    # Output management
    base_output_dir=Path("ml/artifacts/tableau"),
    create_versioned_dirs=True,         # Create versioned directories
    keep_export_history=10,             # Keep last 10 exports

    # Quality gates
    min_model_performance=0.5,          # Minimum R² score required
    max_error_threshold=10.0,           # Maximum MAE allowed
    require_evaluation=True             # Require evaluation results
)
```

## 📊 Data Sources

### Core Data Files

Each dashboard requires specific CSV files:

#### Model Performance Dashboard
- `model_performance_metrics.csv` - Core metrics with confidence intervals
- `position_performance.csv` - Position-specific performance data
- `model_comparison.csv` - Model architecture comparisons
- `cross_validation_results.csv` - Cross-validation stability data
- `feature_importance.csv` - Feature importance rankings

#### Prediction Analysis Dashboard
- `prediction_analysis.csv` - Enhanced prediction data with error metrics
- `calibration_data.csv` - Model calibration analysis
- `error_distribution.csv` - Error pattern analysis
- `model_performance_metrics.csv` - Performance metrics reference

#### Data Quality Dashboard
- `data_coverage.csv` - Missing data and completeness analysis
- `feature_statistics.csv` - Feature distribution statistics
- `outlier_analysis.csv` - Outlier detection results
- `data_drift.csv` - Data drift indicators

#### Business Impact Dashboard
- `player_recommendations.csv` - Player recommendations with value scores
- `value_analysis.csv` - Value analysis by price points
- `risk_assessment.csv` - Risk analysis by position
- `seasonal_trends.csv` - Seasonal performance trends

### Metadata Files

- `dashboard_metadata.json` - Dashboard configuration and metadata
- `calculated_fields.json` - Tableau calculated field definitions
- `dashboard_actions.json` - Filter and parameter configurations
- `export_summary.json` - Export summary and quality report

## 🎯 Key Features

### Automated Data Export

The system automatically exports data in Tableau-optimized formats:
- **Proper data types** for optimal Tableau performance
- **Flattened relationships** for easy dashboard creation
- **Calculated metrics** pre-computed for faster analysis
- **Version control** with model versioning support

### Quality Gates

Built-in quality checks ensure reliable dashboard data:
- **Performance thresholds** (R², MAE validation)
- **Data completeness** checks
- **Outlier detection** and reporting
- **Drift detection** for data quality monitoring

### Business Intelligence Integration

Advanced analytics for FPL decision-making:
- **Player recommendations** with risk-adjusted scoring
- **Value analysis** with points-per-million metrics
- **ROI calculations** for investment decisions
- **Seasonal trend analysis** for timing optimization

### Monitoring and Alerts

Comprehensive monitoring capabilities:
- **Automated quality checks** with configurable thresholds
- **Performance degradation** detection
- **Data drift** monitoring
- **Prediction confidence** tracking

## 🔍 Dashboard Usage Guide

### Opening Dashboards in Tableau

1. **Launch Tableau Desktop**
2. **Connect to Data Sources**
   - File > Open
   - Navigate to exported CSV files
   - Select primary data source for each dashboard

3. **Apply Dashboard Configuration**
   - Each dashboard JSON file contains complete configuration
   - Copy calculated fields from `calculated_fields.json`
   - Configure filters and parameters from `dashboard_actions.json`

### Creating Custom Dashboards

```python
from ml.tableau.export import TableauDataExporter

# Create custom export configuration
custom_config = TableauExportConfig(
    include_predictions=True,
    include_feature_importance=True,
    max_missing_threshold=0.05  # Stricter quality threshold
)

exporter = TableauDataExporter(custom_config)

# Export with custom settings
files = exporter.export_comprehensive_dashboard_data(
    pipeline_result,
    output_dir=Path("custom_dashboard_data"),
    model_version="custom_v1"
)
```

### Performance Optimization

- **Use Tableau extracts** (.hyper files) for large datasets
- **Enable data source filters** to reduce load times
- **Schedule extract refreshes** for regular updates
- **Monitor data source performance** in Tableau

## 🛠 Troubleshooting

### Common Issues

#### 1. Missing Data Files
**Error**: "Cannot find data source file"
**Solution**:
- Verify export completed successfully
- Check file paths in dashboard configurations
- Ensure pipeline ran with evaluation enabled

#### 2. Performance Issues
**Error**: "Dashboard loading slowly"
**Solution**:
- Create Tableau extracts (.hyper files)
- Reduce data granularity if needed
- Enable data source filters
- Check system memory usage

#### 3. Quality Gate Failures
**Error**: "Model does not meet quality thresholds"
**Solution**:
- Review model training parameters
- Check data quality before training
- Adjust quality thresholds in configuration
- Investigate feature engineering

### Debug Mode

Enable debug logging for detailed diagnostics:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run export with debug information
exported_files = exporter.export_comprehensive_dashboard_data(
    pipeline_result, output_dir, model_version, debug=True
)
```

## 📈 Advanced Features

### Custom Metrics Integration

Add custom metrics to dashboards:

```python
# Extend the exporter with custom metrics
class CustomTableauExporter(TableauDataExporter):
    def _create_custom_metrics(self, pipeline_result):
        # Add custom business metrics
        custom_data = {
            'custom_roi': pipeline_result.predictions['test']['actual_points'].sum() / 1000,  # Example
            'market_efficiency': self._calculate_market_efficiency(pipeline_result)
        }
        return custom_data

    def _calculate_market_efficiency(self, pipeline_result):
        # Custom market efficiency calculation
        return 0.85  # Placeholder
```

### Scheduled Exports

Set up automated dashboard updates:

```python
from ml.tableau.pipeline_integration import TableauPipelineIntegration
from ml.config import PipelineConfig

# Configure scheduled exports
integration = TableauPipelineIntegration()
pipeline = PointsPredictionPipeline(PipelineConfig())

# Schedule regular updates
schedule.every().day.at("06:00").do(
    lambda: integration.export_after_training(pipeline.run())
)
```

### Integration with BI Tools

Connect to other business intelligence platforms:

```python
# Export for Power BI
def export_for_powerbi(pipeline_result, output_dir):
    # Power BI optimized export
    pass

# Export for Looker
def export_for_looker(pipeline_result, output_dir):
    # Looker optimized export
    pass
```

## 🤝 Contributing

### Adding New Dashboard Components

1. Create dashboard configuration JSON file
2. Add corresponding data export methods in `export.py`
3. Update documentation and examples
4. Test with sample data

### Extending Export Functionality

1. Subclass `TableauDataExporter`
2. Override specific export methods
3. Add custom configuration options
4. Update integration points

## 📋 Requirements

- **Python**: 3.8+
- **Pandas**: 1.5.0+
- **NumPy**: 1.21.0+
- **Tableau Desktop**: 2023.1+
- **Memory**: 8GB+ recommended for large datasets

## 🔄 Version History

- **v1.0.0** (Current): Initial release with 4 comprehensive dashboards
- **v0.9.0** (Beta): Feature-complete beta with core functionality
- **v0.1.0** (Alpha): Initial prototype with basic export functionality

## 📞 Support

For support and questions:

1. Check this documentation first
2. Review the troubleshooting section
3. Check existing issues in the project repository
4. Create detailed bug reports with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Sample data (if applicable)

## 📄 License

This Tableau integration module is part of the FPL-Elo-Insights project and follows the same licensing terms.

---

**Happy analyzing!** 📊✨
