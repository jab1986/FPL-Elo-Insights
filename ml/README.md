# ML Pipeline Overview

This directory now contains the second iteration of the Fantasy Premier League points prediction pipeline.
The implementation focuses on creating a reproducible baseline while layering in contextual team form
features, richer evaluation and persisted predictions for downstream analysis.

## What was built

- **Data loading** – merges the public CSV exports (`merged_gw.csv`) for the configured seasons and filters
  by gameweek using the standard library `csv` module so the pipeline has no runtime dependency on pandas.
- **Feature engineering** – constructs lagged and rolling statistics for points, expected goal involvement
  metrics, ICT index and transfer trends while only using information that would have been available prior
  to kickoff. Team and opponent context is derived via club-level rolling aggregates so each appearance is
  evaluated within recent squad performance trends using pure Python data structures and math utilities.
- **Baseline model** – a lightweight ridge regression implemented with handcrafted linear algebra to avoid
  external dependencies while still supporting coefficient inspection.
- **Pipeline runner** – orchestrates the process end-to-end, stores metrics, learned coefficients and the
  generated predictions for both the training and hold-out splits in `ml/artifacts/`, writing CSV outputs
  directly without relying on pandas.
- **Rolling-origin validation** – optional time-aware cross-validation per season to monitor generalisation
  before production deployment.
- **Diagnostics & baselines** – every run now benchmarks against naïve history-based baselines and produces
  breakdowns by season and position so shortcomings are surfaced early without manual notebook work.
- **Automatic model selection** – provide an `alpha_grid` in `PipelineConfig` to score ridge candidates via
  rolling validation and persist the chosen hyperparameters alongside the diagnostics.
- **Residual uncertainty calibration** – training residuals are transformed into split-aware prediction
  intervals, risk bands and expected error widths so downstream tools can reason about confidence and
  upside rather than single point forecasts.

## Usage

```bash
python -m ml.pipeline
```

The command executes the default configuration (latest two seasons, three gameweeks of history per player
and a two gameweek holdout). Metrics, coefficients and predictions are written to `ml/artifacts`.

To experiment with different settings (e.g., limiting to a single season or adjusting the history
requirement) instantiate `PipelineConfig` manually:

```python
from ml import PipelineConfig, PointsPredictionPipeline

config = PipelineConfig(
    seasons=("2024-2025",),
    max_gameweek=10,
    min_history_games=2,
    holdout_gameweeks=1,
    cross_validation_folds=3,
    cv_min_train_gameweeks=2,
    alpha_grid=(0.5, 1.0, 5.0),
    tuning_metric="rmse",
)
PointsPredictionPipeline(config).run()
```

The `PipelineResult` exposes the trained model, feature list, diagnostics and metrics, making it easy to
plug future experiments (Optuna, MLflow, ensembling, etc.) into this scaffold. Predictions for each split,
baseline comparisons, calibrated prediction intervals and any rolling-origin validation outputs are returned
alongside the trained model for deeper analysis. When `alpha_grid` is provided the pipeline also records the
per-candidate summary in the saved diagnostics to document how the winning regularisation strength was
chosen.

## Next steps

- Expand the feature set with squad context (teammate injuries, fixture congestion) and weather data.
- Integrate probabilistic forecasts (goal, assist, clean sheet probabilities) for richer downstream
  optimisation.
- Swap the handcrafted ridge implementation for dedicated libraries (LightGBM, XGBoost, neural networks)
  once dependency management is in place.
- Persist feature importances and SHAP values to surface which contextual signals drive each prediction.

