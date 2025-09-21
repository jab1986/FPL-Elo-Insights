"""Points prediction pipeline for Fantasy Premier League data without pandas."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .data import load_merged_gameweek_data
from .features import engineer_features
from .model import RidgeRegressionModel
from . import metrics as metrics_module
from .uncertainty import ResidualIntervalEstimator
from .utils import Dataset, clone_records, extract_matrix, to_float


@dataclass
class PipelineResult:
    """Container holding the artefacts of a pipeline run."""

    config: PipelineConfig
    features: List[str]
    train_rows: int
    test_rows: int
    metrics: Dict[str, Dict[str, float]]
    model: RidgeRegressionModel
    predictions: Dict[str, Dataset]
    cross_validation: Optional[List[Dict[str, Any]]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

    def save(self, output_dir: Path) -> None:
        """Persist metrics and model coefficients to disk."""

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "points_baseline_metrics.json"
        coefficients_path = output_dir / "points_baseline_coefficients.json"
        predictions_path = output_dir / "points_baseline_predictions.csv"

        metrics_payload: Dict[str, Any] = {
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

        combined: List[Dict[str, Any]] = []
        for split_name, records in self.predictions.items():
            for row in records:
                entry = dict(row)
                entry["split"] = split_name
                combined.append(entry)

        if combined:
            fieldnames = sorted({key for row in combined for key in row.keys()})
            with predictions_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(",".join(fieldnames) + "\n")
                for row in combined:
                    values = []
                    for field in fieldnames:
                        value = row.get(field, "")
                        if isinstance(value, float) and math.isnan(value):
                            values.append("")
                        else:
                            values.append(str(value))
                    handle.write(",".join(values) + "\n")


class PointsPredictionPipeline:
    """Builds a lightweight baseline model for predicting FPL points."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self) -> PipelineResult:
        raw = load_merged_gameweek_data(self.config)
        dataset, feature_columns = engineer_features(raw, self.config)

        if not dataset:
            raise ValueError("Feature engineering resulted in an empty dataset")

        train_records, test_records = self._split_train_test(dataset)
        if not train_records:
            raise ValueError("Training set is empty after applying the holdout strategy")

        selected_alpha, tuning_summary = self._tune_hyperparameters(train_records, feature_columns)
        model = self._train_model(train_records, feature_columns, alpha=selected_alpha)
        predictions: Dict[str, Dataset] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        diagnostics: Dict[str, Any] = {}

        train_prediction_frame, train_metrics, train_diagnostics = self._score_split(
            model,
            train_records,
            feature_columns,
            dataset,
            extra_columns=("position", "team_id", "opponent_team_id"),
        )

        uncertainty_estimator: Optional[ResidualIntervalEstimator] = None
        if train_prediction_frame:
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
            test_records,
            feature_columns,
            dataset,
            extra_columns=("position", "team_id", "opponent_team_id"),
        )

        if uncertainty_estimator is not None and test_prediction_frame:
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
            train_rows=len(train_records),
            test_rows=len(test_records),
            metrics=metrics,
            model=model,
            predictions=predictions,
            cross_validation=cross_validation,
            diagnostics=diagnostics or None,
            hyperparameters=hyperparameters or None,
        )
        result.save(self.config.resolved_output_dir())
        return result

    def _split_train_test(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        holdout = max(self.config.holdout_gameweeks, 0)
        if holdout == 0:
            return clone_records(dataset), []

        max_gameweek: Dict[object, int] = {}
        for row in dataset:
            season = row["season"]
            gameweek = int(row["gameweek"])
            max_gameweek[season] = max(max_gameweek.get(season, gameweek), gameweek)

        train: Dataset = []
        test: Dataset = []
        for row in dataset:
            season = row["season"]
            threshold = max_gameweek[season] - holdout + 1
            if row["gameweek"] >= threshold:
                test.append(dict(row))
            else:
                train.append(dict(row))

        if not train:
            return clone_records(dataset), []
        return train, test

    def _train_model(
        self,
        dataset: Dataset,
        feature_columns: List[str],
        alpha: Optional[float] = None,
    ) -> RidgeRegressionModel:
        model_alpha = self.config.model_alpha if alpha is None else float(alpha)
        model = RidgeRegressionModel(alpha=model_alpha)
        features = extract_matrix(dataset, feature_columns)
        target = [to_float(row.get(self.config.target)) for row in dataset]
        model.fit(features, target)
        return model

    def _compute_metrics(
        self,
        model: RidgeRegressionModel,
        dataset: Dataset,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        if not dataset:
            return {}
        features = extract_matrix(dataset, feature_columns)
        actual = [to_float(row.get(self.config.target)) for row in dataset]
        predicted = model.predict(features)
        return metrics_module.regression_metrics(actual, predicted)

    def _tune_hyperparameters(
        self, dataset: Dataset, feature_columns: List[str]
    ) -> tuple[float, Optional[Dict[str, Any]]]:
        default_alpha = float(self.config.model_alpha)
        if not dataset or not self.config.alpha_grid or self.config.tuning_folds <= 0:
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

            aggregated = sum(metric_values) / len(metric_values)
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
            "selected_alpha": float(best_alpha),
        }
        return best_alpha, summary

    def _extract_metric_values(
        self, metrics: List[Dict[str, Any]], metric_name: str
    ) -> List[float]:
        values: List[float] = []
        for entry in metrics:
            value = entry.get(metric_name)
            if value is None:
                continue
            numeric = float(value)
            if math.isnan(numeric):
                continue
            if metric_name == "bias":
                numeric = abs(numeric)
            values.append(numeric)
        return values

    def _metric_orientation(self, metric_name: str) -> str:
        lower_is_better = {"mae", "rmse", "median_ae", "mape", "bias"}
        higher_is_better = {"r2", "pearson_r"}
        if metric_name in lower_is_better:
            return "lower"
        if metric_name in higher_is_better:
            return "higher"
        raise ValueError(f"Unsupported tuning metric: {metric_name}")

    @staticmethod
    def _is_better(score: float, best_score: float, orientation: str) -> bool:
        if orientation == "lower":
            return score < best_score - 1e-9
        if orientation == "higher":
            return score > best_score + 1e-9
        raise ValueError(f"Unknown orientation: {orientation}")

    def _evaluate_alpha(
        self, dataset: Dataset, feature_columns: List[str], alpha: float
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        folds = max(0, self.config.tuning_folds)
        if folds <= 0:
            return results

        seasons = sorted({row["season"] for row in dataset})
        for season in seasons:
            season_records = [row for row in dataset if row["season"] == season]
            gameweeks = sorted({row["gameweek"] for row in season_records})
            if len(gameweeks) <= 1:
                continue

            fold_size = max(1, len(gameweeks) // (folds + 1))
            for fold_index in range(folds):
                train_end = (fold_index + 1) * fold_size
                train_gws = gameweeks[:train_end]
                test_gws = gameweeks[train_end : train_end + fold_size]

                if len(train_gws) < self.config.tuning_min_train_gameweeks or not test_gws:
                    continue

                train_df = [row for row in season_records if row["gameweek"] in train_gws]
                test_df = [row for row in season_records if row["gameweek"] in test_gws]
                if not train_df or not test_df:
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
        dataset: Dataset,
        feature_columns: List[str],
        extra_columns: Sequence[str] | None = None,
    ) -> Dataset:
        if not dataset:
            return []

        features = extract_matrix(dataset, feature_columns)
        predictions = model.predict(features)
        base_columns = ["season", "gameweek", "player_id"]
        frame: Dataset = []
        for row, predicted in zip(dataset, predictions):
            entry: Dict[str, Any] = {column: row[column] for column in base_columns if column in row}
            if extra_columns:
                for column in extra_columns:
                    if column in row and column not in entry:
                        entry[column] = row[column]
            entry["actual_points"] = to_float(row.get(self.config.target))
            entry["predicted_points"] = float(predicted)
            frame.append(entry)
        return frame

    def _score_split(
        self,
        model: RidgeRegressionModel,
        dataset: Dataset,
        feature_columns: List[str],
        history: Dataset,
        extra_columns: Sequence[str] | None = None,
    ) -> tuple[Dataset, Dict[str, float], Dict[str, Any]]:
        prediction_frame = self._build_prediction_frame(
            model,
            dataset,
            feature_columns,
            extra_columns=extra_columns,
        )
        metrics: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}

        if prediction_frame:
            metrics = metrics_module.regression_metrics(
                [row["actual_points"] for row in prediction_frame],
                [row["predicted_points"] for row in prediction_frame],
            )
            diagnostics = self._build_diagnostics(dataset, prediction_frame, history)

        return prediction_frame, metrics, diagnostics

    def _build_diagnostics(
        self,
        dataset: Dataset,
        prediction_frame: Dataset,
        history: Dataset,
    ) -> Dict[str, Any]:
        if not dataset or not prediction_frame:
            return {}

        diagnostics: Dict[str, Any] = {
            "sample_count": int(len(dataset)),
            "target_mean": sum([to_float(row.get(self.config.target)) for row in dataset]) / len(dataset),
        }

        baselines = metrics_module.compute_baseline_metrics(dataset, self.config.target, history=history)
        if baselines:
            diagnostics["baselines"] = baselines

        breakdowns: Dict[str, Any] = {}
        for column in ("position", "season"):
            if any(column in row for row in prediction_frame):
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
        self, dataset: Dataset, feature_columns: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        if self.config.cross_validation_folds <= 0:
            return None

        results: List[Dict[str, Any]] = []
        seasons = sorted({row["season"] for row in dataset})
        for season in seasons:
            season_records = [row for row in dataset if row["season"] == season]
            gameweeks = sorted({row["gameweek"] for row in season_records})
            if len(gameweeks) <= 1:
                continue

            fold_size = max(1, len(gameweeks) // (self.config.cross_validation_folds + 1))
            for fold_index in range(self.config.cross_validation_folds):
                train_end = (fold_index + 1) * fold_size
                train_gws = gameweeks[:train_end]
                test_gws = gameweeks[train_end : train_end + fold_size]

                if len(train_gws) < self.config.cv_min_train_gameweeks or not test_gws:
                    continue

                train_df = [row for row in season_records if row["gameweek"] in train_gws]
                test_df = [row for row in season_records if row["gameweek"] in test_gws]
                if not train_df or not test_df:
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
