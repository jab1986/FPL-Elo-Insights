"""Points prediction pipeline for Fantasy Premier League data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_merged_gameweek_data
from .features import engineer_features
from .model import RidgeRegressionModel
from . import metrics as metrics_module
from .uncertainty import ResidualIntervalEstimator


@dataclass
class PipelineResult:
    """Container holding the artefacts of a pipeline run."""

    config: PipelineConfig
    features: List[str]
    train_rows: int
    test_rows: int
    metrics: Dict[str, Dict[str, float]]
    model: RidgeRegressionModel
    predictions: Dict[str, pd.DataFrame]
    cross_validation: Optional[List[Dict[str, Any]]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

    def save(self, output_dir: Path) -> None:
        """Persist metrics and model coefficients to disk."""

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "points_baseline_metrics.json"
        coefficients_path = output_dir / "points_baseline_coefficients.json"
        predictions_path = output_dir / "points_baseline_predictions.csv"

        metrics_payload = {
            "config": {
                "seasons": list(self.config.seasons),
                "min_gameweek": self.config.min_gameweek,
                "max_gameweek": self.config.max_gameweek,
                "holdout_gameweeks": self.config.holdout_gameweeks,
                "min_history_games": self.config.min_history_games,
                "cross_validation_folds": self.config.cross_validation_folds,
            },
            "metrics": self.metrics,
            "feature_count": len(self.features),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }
        if self.cross_validation:
            metrics_payload["cross_validation"] = self.cross_validation
        if self.diagnostics:
            metrics_payload["diagnostics"] = self.diagnostics
        if self.hyperparameters:
            metrics_payload["hyperparameters"] = self.hyperparameters
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        coefficients_payload = self.model.to_dict(self.features)
        coefficients_path.write_text(json.dumps(coefficients_payload, indent=2))

        prediction_frames = []
        for split_name, frame in self.predictions.items():
            if frame.empty:
                continue
            enriched = frame.copy()
            enriched["split"] = split_name
            prediction_frames.append(enriched)
        if prediction_frames:
            combined = pd.concat(prediction_frames, ignore_index=True)
            combined.to_csv(predictions_path, index=False)


class PointsPredictionPipeline:
    """Builds a lightweight baseline model for predicting FPL points."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self) -> PipelineResult:
        """Execute the end-to-end pipeline."""

        raw = load_merged_gameweek_data(self.config)
        dataset, feature_columns = engineer_features(raw, self.config)

        if dataset.empty:
            raise ValueError("Feature engineering resulted in an empty dataset")

        train_df, test_df = self._split_train_test(dataset)
        if train_df.empty:
            raise ValueError("Training set is empty after applying the holdout strategy")

        selected_alpha, tuning_summary = self._tune_hyperparameters(train_df, feature_columns)
        model = self._train_model(train_df, feature_columns, alpha=selected_alpha)
        predictions: Dict[str, pd.DataFrame] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Any] = {}

        train_prediction_frame, train_metrics, train_diagnostics = self._score_split(
            model,
            train_df,
            feature_columns,
            dataset,
            extra_columns=("position", "team_id", "opponent_team_id"),
        )

        uncertainty_estimator: Optional[ResidualIntervalEstimator] = None
        if not train_prediction_frame.empty:
            uncertainty_estimator = ResidualIntervalEstimator(
                confidence_levels=self.config.uncertainty_confidence_levels
            ).fit(train_prediction_frame)
            train_prediction_frame = uncertainty_estimator.apply(train_prediction_frame)
            summary = uncertainty_estimator.describe()
            if summary:
                diagnostics["uncertainty"] = summary

        predictions["train"] = train_prediction_frame
        if train_metrics:
            metrics["train"] = train_metrics
        if train_diagnostics:
            diagnostics["train"] = train_diagnostics

        test_prediction_frame, test_metrics, test_diagnostics = self._score_split(
            model,
            test_df,
            feature_columns,
            dataset,
            extra_columns=("position", "team_id", "opponent_team_id"),
        )

        if uncertainty_estimator is not None and not test_prediction_frame.empty:
            test_prediction_frame = uncertainty_estimator.apply(test_prediction_frame)

        predictions["test"] = test_prediction_frame
        if test_metrics:
            metrics["test"] = test_metrics
        if test_diagnostics:
            diagnostics["test"] = test_diagnostics

        cross_validation = self._run_cross_validation(dataset, feature_columns)

        if tuning_summary:
            diagnostics["model_selection"] = tuning_summary

        hyperparameters: Dict[str, Any] = {"alpha": float(selected_alpha)}
        if tuning_summary:
            hyperparameters["tuning"] = tuning_summary

        result = PipelineResult(
            config=self.config,
            features=feature_columns,
            train_rows=len(train_df),
            test_rows=len(test_df),
            metrics=metrics,
            model=model,
            predictions=predictions,
            cross_validation=cross_validation,
            diagnostics=diagnostics or None,
            hyperparameters=hyperparameters or None,
        )
        result.save(self.config.resolved_output_dir())
        return result

    def _split_train_test(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform a time-based split, keeping the latest gameweeks for testing."""

        holdout = max(self.config.holdout_gameweeks, 0)
        if holdout == 0:
            return dataset, dataset.iloc[0:0]

        max_gameweek_per_season = dataset.groupby("season")["gameweek"].transform("max")
        test_mask = dataset["gameweek"] >= (max_gameweek_per_season - holdout + 1)
        train_df = dataset.loc[~test_mask].reset_index(drop=True)
        test_df = dataset.loc[test_mask].reset_index(drop=True)
        if train_df.empty:
            return dataset.reset_index(drop=True), dataset.iloc[0:0]
        return train_df, test_df

    def _train_model(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        alpha: Optional[float] = None,
    ) -> RidgeRegressionModel:
        """Fit the ridge regression model using the prepared dataset."""

        model_alpha = self.config.model_alpha if alpha is None else float(alpha)
        model = RidgeRegressionModel(alpha=model_alpha)
        features = dataset[feature_columns].to_numpy(dtype=float)
        target = dataset[self.config.target].to_numpy(dtype=float)
        model.fit(features, target)
        return model

    def _compute_metrics(
        self,
        model: RidgeRegressionModel,
        dataset: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """Calculate MAE, RMSE and R² for a dataset."""

        features = dataset[feature_columns].to_numpy(dtype=float)
        actual = dataset[self.config.target].to_numpy(dtype=float)
        predicted = model.predict(features)

        return metrics_module.regression_metrics(actual, predicted)

    def _tune_hyperparameters(
        self, dataset: pd.DataFrame, feature_columns: List[str]
    ) -> tuple[float, Optional[Dict[str, Any]]]:
        """Select the ridge alpha via lightweight rolling validation."""

        default_alpha = float(self.config.model_alpha)
        if dataset.empty or not self.config.alpha_grid or self.config.tuning_folds <= 0:
            return default_alpha, None

        metric_name = self.config.tuning_metric
        direction = self._metric_orientation(metric_name)
        best_alpha = default_alpha
        best_score: Optional[float] = None
        summary_records: List[Dict[str, Any]] = []

        for alpha in self.config.alpha_grid:
            fold_metrics = self._evaluate_alpha(dataset, feature_columns, alpha)
            metric_values = self._extract_metric_values(fold_metrics, metric_name)
            if not metric_values:
                continue

            aggregated = float(np.mean(metric_values))
            summary_records.append(
                {
                    "alpha": float(alpha),
                    metric_name: aggregated,
                    "fold_count": int(len(metric_values)),
                }
            )

            if best_score is None or self._is_better(aggregated, best_score, direction):
                best_score = aggregated
                best_alpha = float(alpha)

        if not summary_records:
            return default_alpha, None

        summary: Dict[str, Any] = {
            "metric": metric_name,
            "results": summary_records,
        }
        summary["selected_alpha"] = float(best_alpha)
        return best_alpha, summary

    def _extract_metric_values(
        self, metrics: List[Dict[str, Any]], metric_name: str
    ) -> List[float]:
        """Return valid metric values from fold evaluations."""

        values: List[float] = []
        for entry in metrics:
            value = entry.get(metric_name)
            if value is None:
                continue
            numeric = float(value)
            if np.isnan(numeric):
                continue
            if metric_name == "bias":
                numeric = abs(numeric)
            values.append(numeric)
        return values

    def _metric_orientation(self, metric_name: str) -> str:
        """Return whether higher or lower values are preferred for a metric."""

        lower_is_better = {"mae", "rmse", "median_ae", "mape", "bias"}
        higher_is_better = {"r2", "pearson_r"}
        if metric_name in lower_is_better:
            return "lower"
        if metric_name in higher_is_better:
            return "higher"
        raise ValueError(f"Unsupported tuning metric: {metric_name}")

    @staticmethod
    def _is_better(score: float, best_score: float, orientation: str) -> bool:
        """Compare metric scores according to the desired orientation."""

        if orientation == "lower":
            return score < best_score - 1e-9
        if orientation == "higher":
            return score > best_score + 1e-9
        raise ValueError(f"Unknown orientation: {orientation}")

    def _evaluate_alpha(
        self, dataset: pd.DataFrame, feature_columns: List[str], alpha: float
    ) -> List[Dict[str, Any]]:
        """Compute rolling validation metrics for a specific alpha."""

        results: List[Dict[str, Any]] = []
        folds = max(0, self.config.tuning_folds)
        if folds <= 0:
            return results

        for season, season_df in dataset.groupby("season"):
            season_df = season_df.sort_values("gameweek")
            gameweeks = np.sort(season_df["gameweek"].unique())
            if len(gameweeks) <= 1:
                continue

            fold_size = max(1, len(gameweeks) // (folds + 1))
            for fold_index in range(folds):
                train_end = (fold_index + 1) * fold_size
                train_gws = gameweeks[:train_end]
                test_gws = gameweeks[train_end : train_end + fold_size]

                if (
                    len(train_gws) < self.config.tuning_min_train_gameweeks
                    or test_gws.size == 0
                ):
                    continue

                train_df = season_df[season_df["gameweek"].isin(train_gws)]
                test_df = season_df[season_df["gameweek"].isin(test_gws)]
                if train_df.empty or test_df.empty:
                    continue

                model = self._train_model(train_df, feature_columns, alpha=alpha)
                metrics = self._compute_metrics(model, test_df, feature_columns)
                metrics.update(
                    {
                        "fold": fold_index + 1,
                        "season": season,
                        "train_rows": int(len(train_df)),
                        "test_rows": int(len(test_df)),
                        "train_gameweek_max": int(train_gws[-1]),
                        "test_gameweeks": [int(gw) for gw in test_gws],
                        "alpha": float(alpha),
                    }
                )
                results.append(metrics)

        return results

    def _build_prediction_frame(
        self,
        model: RidgeRegressionModel,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        extra_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Create a dataframe containing actual vs predicted points for a split."""

        if dataset.empty:
            return pd.DataFrame(columns=["season", "gameweek", "player_id", "actual_points", "predicted_points"])

        features = dataset[feature_columns].to_numpy(dtype=float)
        predictions = model.predict(features)

        base_columns = ["season", "gameweek", "player_id"]
        if extra_columns:
            for column in extra_columns:
                if column in dataset.columns and column not in base_columns:
                    base_columns.append(column)
        frame = dataset[base_columns].copy()
        frame["actual_points"] = dataset[self.config.target].to_numpy(dtype=float)
        frame["predicted_points"] = predictions
        return frame.reset_index(drop=True)

    def _score_split(
        self,
        model: RidgeRegressionModel,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        history: pd.DataFrame,
        extra_columns: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
        """Generate predictions, metrics and diagnostics for a dataset split."""

        prediction_frame = self._build_prediction_frame(
            model,
            dataset,
            feature_columns,
            extra_columns=extra_columns,
        )
        metrics: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}

        if not prediction_frame.empty:
            metrics = metrics_module.regression_metrics(
                prediction_frame["actual_points"].to_numpy(),
                prediction_frame["predicted_points"].to_numpy(),
            )
            diagnostics = self._build_diagnostics(dataset, prediction_frame, history)

        return prediction_frame, metrics, diagnostics

    def _build_diagnostics(
        self,
        dataset: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compile additional diagnostic information for a dataset split."""

        if dataset.empty or prediction_frame.empty:
            return {}

        diagnostics: Dict[str, Any] = {
            "sample_count": int(len(dataset)),
            "target_mean": float(dataset[self.config.target].mean()),
        }

        baselines = metrics_module.compute_baseline_metrics(dataset, self.config.target, history=history)
        if baselines:
            diagnostics["baselines"] = baselines

        breakdowns: Dict[str, Any] = {}
        for column in ("position", "season"):
            if column in prediction_frame.columns:
                breakdown = metrics_module.grouped_metrics(
                    prediction_frame,
                    actual_column="actual_points",
                    predicted_column="predicted_points",
                    group_column=column,
                )
                if breakdown:
                    breakdowns[column] = breakdown

        if breakdowns:
            diagnostics["breakdowns"] = breakdowns

        return diagnostics

    def _run_cross_validation(
        self, dataset: pd.DataFrame, feature_columns: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Perform rolling-origin cross validation per season."""

        if self.config.cross_validation_folds <= 0:
            return None

        results: List[Dict[str, Any]] = []
        for season, season_df in dataset.groupby("season"):
            season_df = season_df.sort_values("gameweek")
            gameweeks = np.sort(season_df["gameweek"].unique())
            if len(gameweeks) <= 1:
                continue

            fold_size = max(1, len(gameweeks) // (self.config.cross_validation_folds + 1))
            for fold_index in range(self.config.cross_validation_folds):
                train_end = (fold_index + 1) * fold_size
                train_gws = gameweeks[:train_end]
                test_gws = gameweeks[train_end : train_end + fold_size]

                if (
                    len(train_gws) < self.config.cv_min_train_gameweeks
                    or test_gws.size == 0
                ):
                    continue

                train_df = season_df[season_df["gameweek"].isin(train_gws)]
                test_df = season_df[season_df["gameweek"].isin(test_gws)]
                if train_df.empty or test_df.empty:
                    continue

                model = self._train_model(train_df, feature_columns)
                metrics = self._compute_metrics(model, test_df, feature_columns)
                metrics.update(
                    {
                        "fold": fold_index + 1,
                        "season": season,
                        "train_rows": int(len(train_df)),
                        "test_rows": int(len(test_df)),
                        "train_gameweek_max": int(train_gws[-1]),
                        "test_gameweeks": [int(gw) for gw in test_gws],
                    }
                )
                results.append(metrics)

        return results or None


if __name__ == "__main__":
    pipeline = PointsPredictionPipeline()
    result = pipeline.run()
    print(json.dumps(result.metrics, indent=2))

