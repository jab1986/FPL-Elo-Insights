"""Points prediction pipeline for Fantasy Premier League data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_merged_gameweek_data
from .features import engineer_features
from .model import RidgeRegressionModel
from .models import PositionSpecificModel, DeepEnsembleModel
from . import metrics as metrics_module
from .evaluation import EvaluationResult, AdvancedEvaluator


@dataclass
class PipelineResult:
    """Container holding the artefacts of a pipeline run."""

    config: PipelineConfig
    features: List[str]
    train_rows: int
    test_rows: int
    metrics: Dict[str, Dict[str, float]]
    model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel]
    predictions: Dict[str, pd.DataFrame]
    cross_validation: Optional[List[Dict[str, Any]]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    model_comparison: Optional[Dict[str, Dict[str, float]]] = None
    selected_model_type: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None

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
        if self.model_comparison:
            metrics_payload["model_comparison"] = self.model_comparison
        if self.selected_model_type:
            metrics_payload["selected_model_type"] = self.selected_model_type
        if self.evaluation:
            metrics_payload["evaluation"] = self.evaluation.to_dict()
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        # Handle different model types for coefficients
        if hasattr(self.model, 'to_dict'):
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

        model = self._train_model(train_df, feature_columns)
        predictions: Dict[str, pd.DataFrame] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Any] = {}
        model_comparison = None

        if self.config.model_comparison or self.config.model_type == "auto":
            model, model_comparison = self._select_best_model(train_df, test_df, feature_columns, dataset)

        for split_name, split_df in (("train", train_df), ("test", test_df)):
            prediction_frame, split_metrics, split_diagnostics = self._score_split(
                model,
                split_df,
                feature_columns,
                dataset,
                extra_columns=("position", "team_id", "opponent_team_id"),
            )
            predictions[split_name] = prediction_frame
            if split_metrics:
                metrics[split_name] = split_metrics
            if split_diagnostics:
                diagnostics[split_name] = split_diagnostics

        cross_validation = self._run_cross_validation(dataset, feature_columns)

        evaluation = self._run_enhanced_evaluation(
            model=model,
            full_dataset=dataset,
            test_df=test_df,
            feature_columns=feature_columns,
        )

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
            model_comparison=model_comparison if self.config.model_comparison else None,
            selected_model_type=getattr(model, 'model_type', 'ridge') if hasattr(model, 'model_type') else None,
            evaluation=evaluation,
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

    def _compute_metrics(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
        dataset: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """Calculate MAE, RMSE and R² for a dataset."""

        features = self._prepare_features(dataset, feature_columns, model)
        actual = dataset[self.config.target].to_numpy(dtype=float)
        predicted = model.predict(features)

        return metrics_module.regression_metrics(actual, predicted)

    def _prepare_features(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features in a format compatible with the provided model."""

        feature_frame = dataset[feature_columns]
        if isinstance(model, RidgeRegressionModel):
            return feature_frame.to_numpy(dtype=float)
        return feature_frame

    def _build_prediction_frame(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
        dataset: pd.DataFrame,
        feature_columns: List[str],
        extra_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Create a dataframe containing actual vs predicted points for a split."""

        if dataset.empty:
            return pd.DataFrame(columns=["season", "gameweek", "player_id", "actual_points", "predicted_points"])

        features = self._prepare_features(dataset, feature_columns, model)
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
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
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

        if self.config.model_type not in {"ridge", "auto"}:
            # Cross-validation on advanced models can be prohibitively expensive
            return None

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

    def _select_best_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        full_dataset: pd.DataFrame,
    ) -> tuple[Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel], Dict[str, Dict[str, float]]]:
        """Compare different model architectures and select the best performing one."""
        model_candidates = {}
        model_metrics = {}

        # Always include Ridge as baseline
        try:
            ridge_model = self._train_model_by_type(train_df, feature_columns, "ridge")
            test_metrics = self._compute_metrics(ridge_model, test_df, feature_columns)
            model_candidates["ridge"] = ridge_model
            model_metrics["ridge"] = test_metrics
        except Exception as e:
            print(f"Warning: Ridge model failed: {e}")

        # Try PositionSpecific model if dependencies are available
        try:
            pos_model = self._train_model_by_type(train_df, feature_columns, "position_specific")
            test_metrics = self._compute_metrics(pos_model, test_df, feature_columns)
            model_candidates["position_specific"] = pos_model
            model_metrics["position_specific"] = test_metrics
        except ImportError as e:
            print(f"Warning: PositionSpecific model not available (missing dependencies): {e}")
        except Exception as e:
            print(f"Warning: PositionSpecific model failed: {e}")

        # Try DeepEnsemble model if dependencies are available
        try:
            ensemble_model = self._train_model_by_type(train_df, feature_columns, "deep_ensemble")
            test_metrics = self._compute_metrics(ensemble_model, test_df, feature_columns)
            model_candidates["deep_ensemble"] = ensemble_model
            model_metrics["deep_ensemble"] = test_metrics
        except ImportError as e:
            print(f"Warning: DeepEnsemble model not available (missing dependencies): {e}")
        except Exception as e:
            print(f"Warning: DeepEnsemble model failed: {e}")

        if not model_candidates:
            raise ValueError("No models could be trained successfully")

        # Select best model based on MAE
        best_model_type = min(
            model_metrics.keys(),
            key=lambda x: model_metrics[x].get("mae", float("inf")),
        )
        best_model = model_candidates[best_model_type]

        print(
            "Selected model: "
            f"{best_model_type} (MAE: {model_metrics[best_model_type].get('mae', 'N/A')})"
        )

        return best_model, model_metrics

    def _train_model_by_type(
        self, dataset: pd.DataFrame, feature_columns: List[str], model_type: str
    ) -> Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel]:
        """Train a specific model type with proper error handling."""
        feature_frame = dataset[feature_columns]
        target = dataset[self.config.target].to_numpy(dtype=float)

        if model_type == "ridge":
            model = RidgeRegressionModel(alpha=self.config.model_alpha)
            model.fit(feature_frame.to_numpy(dtype=float), target)
            model.model_type = "ridge"
            return model

        elif model_type == "position_specific":
            # Ensure position_code column exists
            if "position_code" not in dataset.columns:
                raise ValueError("position_code column required for position-specific training")

            model = PositionSpecificModel(self.config)
            model.fit(feature_frame, pd.Series(target))
            model.model_type = "position_specific"
            return model

        elif model_type == "deep_ensemble":
            model = DeepEnsembleModel(self.config)
            model.fit(feature_frame, pd.Series(target))
            model.model_type = "deep_ensemble"
            return model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _run_enhanced_evaluation(
        self,
        model: Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel],
        full_dataset: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Optional[EvaluationResult]:
        """Run comprehensive evaluation using the AdvancedEvaluator."""
        try:
            evaluator = AdvancedEvaluator(self.config)

            # Use test set for evaluation if available, otherwise use full dataset
            eval_dataset = test_df if not test_df.empty else full_dataset
            if eval_dataset.empty:
                return None

            # Get target values
            y_true = eval_dataset[self.config.target].to_numpy(dtype=float)
            X_eval = eval_dataset[feature_columns]

            return evaluator.evaluate_model(
                model=model,
                X=X_eval,
                y=pd.Series(y_true),
                feature_names=feature_columns,
                dataset=full_dataset,
                model_name=getattr(model, 'model_type', 'ridge') if hasattr(model, 'model_type') else 'ridge'
            )
        except Exception as e:
            print(f"Warning: Enhanced evaluation failed: {e}")
            return None

    def _train_model(self, dataset: pd.DataFrame, feature_columns: List[str]) -> Union[RidgeRegressionModel, PositionSpecificModel, DeepEnsembleModel]:
        """Fit the specified model type using the prepared dataset."""

        requested_type = self.config.model_type
        if requested_type == "auto":
            requested_type = "ridge"
        return self._train_model_by_type(dataset, feature_columns, requested_type)


if __name__ == "__main__":
    pipeline = PointsPredictionPipeline()
    result = pipeline.run()
    print(json.dumps(result.metrics, indent=2))

