# Tableau Dashboard Setup Guide

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install pandas numpy pathlib typing dataclasses
```

### 2. Run Your ML Pipeline
```python
from ml.pipeline import PointsPredictionPipeline
from ml.config import PipelineConfig

# Run pipeline with evaluation
config = PipelineConfig()
config.cross_validation_folds = 5  # Enable evaluation
pipeline = PointsPredictionPipeline(config)
result = pipeline.run()
```

### 3. Export Tableau Data
```python
from ml.tableau.pipeline_integration import export_tableau_data_from_pipeline

# Export all dashboard data
exported_files = export_tableau_data_from_pipeline(result)
print(f"Created {len(exported_files)} files for Tableau")
```

### 4. Open in Tableau
1. **Launch Tableau Desktop**
2. **Connect to CSV files** in your export directory
3. **Import dashboard configurations** from the JSON files
4. **Apply calculated fields** from the export

## 🎉 You're Ready!

Your dashboards will include:
- ✅ Model performance monitoring
- ✅ Prediction accuracy analysis
- ✅ Data quality assessment
- ✅ Business impact analysis

---

## Need Help?

See the full documentation in `README.md` for:
- Advanced configuration options
- Troubleshooting guide
- Custom dashboard creation
- Integration examples
