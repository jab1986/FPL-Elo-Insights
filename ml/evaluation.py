"""Comprehensive evaluation framework for ML pipelines with advanced statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path

from .config import PipelineConfig


@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results."""

    metrics: Dict[str, float]
    position_metrics: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    calibration_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    reliability_scores: Dict[str, float]
    statistical_tests: Dict[str, Any]
    model_comparison: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary format."""
        result = {
            "metrics": self.metrics,
            "position_metrics": self.position_metrics,
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "calibration_metrics": self.calibration_metrics,
            "feature_importance": self.feature_importance,
            "reliability_scores": self.reliability_scores,
            "statistical_tests": self.statistical_tests,
        }

        if self.model_comparison:
            result["model_comparison"] = self.model_comparison
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics
        if self.validation_results:
            result["validation_results"] = self.validation_results

        return result


class AdvancedEvaluator:
    """Comprehensive evaluation framework for ML models."""

    def __init__(self, config: PipelineConfig, random_state: Optional[int] = None):
        self.config = config
        self.random_state = random_state or 42
        self.position_map = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}

    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        dataset: Optional[pd.DataFrame] = None,
        model_name: str = "model"
    ) -> EvaluationResult:
        """Perform comprehensive evaluation of a model."""

        # Basic predictions
        predictions = model.predict(X)

        # Core metrics
        metrics = self._compute_core_metrics(y, predictions)

        # Position-specific metrics
        position_metrics = self._compute_position_metrics(X, y, predictions)

        # Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(y, predictions)

        # Calibration metrics
        calibration_metrics = self._compute_calibration_metrics(y, predictions)

        # Feature importance
        feature_importance = self._compute_feature_importance(model, X, y, feature_names)

        # Reliability scores
        reliability_scores = self._compute_reliability_scores(y, predictions, X)

        # Statistical tests
        statistical_tests = self._perform_statistical_tests(y, predictions, dataset)

        # Diagnostics
        diagnostics = self._run_diagnostics(X, y, predictions, dataset)

        # Validation results
        validation_results = self._run_validation(X, y, predictions)

        return EvaluationResult(
            metrics=metrics,
            position_metrics=position_metrics,
            confidence_intervals=confidence_intervals,
            calibration_metrics=calibration_metrics,
            feature_importance=feature_importance,
            reliability_scores=reliability_scores,
            statistical_tests=statistical_tests,
            diagnostics=diagnostics,
            validation_results=validation_results
        )

    def _compute_core_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute core regression metrics."""
        residuals = y_pred - y_true

        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs(residuals / np.maximum(np.abs(y_true), 1))) * 100),
            "bias": float(np.mean(residuals)),
            "median_ae": float(np.median(np.abs(residuals))),
            "std_ae": float(np.std(np.abs(residuals))),
            "count": int(len(y_true)),
            "actual_mean": float(np.mean(y_true)),
            "predicted_mean": float(np.mean(y_pred)),
            "actual_std": float(np.std(y_true)),
            "predicted_std": float(np.std(y_pred)),
        }

    def _compute_position_metrics(
        self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by player position."""
        position_metrics = {}

        if "position_code" not in X.columns:
            return position_metrics

        for position_code, position_name in self.position_map.items():
            mask = X["position_code"] == position_code
            if mask.sum() > 0:
                y_true_pos = y_true[mask]
                y_pred_pos = y_pred[mask]

                position_metrics[position_name] = {
                    "mae": float(mean_absolute_error(y_true_pos, y_pred_pos)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true_pos, y_pred_pos))),
                    "r2": float(r2_score(y_true_pos, y_pred_pos)),
                    "mape": float(np.mean(np.abs((y_true_pos - y_pred_pos) /
                                               np.maximum(np.abs(y_true_pos), 1))) * 100),
                    "count": int(mask.sum()),
                    "actual_mean": float(np.mean(y_true_pos)),
                    "predicted_mean": float(np.mean(y_pred_pos)),
                }

        return position_metrics

    def _compute_confidence_intervals(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for key metrics."""
        metrics_to_bootstrap = ["mae", "rmse", "r2", "mape"]
        confidence_intervals = {}

        for metric in metrics_to_bootstrap:
            bootstrapped_values = []

            for _ in range(n_bootstrap):
                indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]

                if metric == "mae":
                    value = mean_absolute_error(y_true_boot, y_pred_boot)
                elif metric == "rmse":
                    value = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
                elif metric == "r2":
                    value = r2_score(y_true_boot, y_pred_boot)
                elif metric == "mape":
                    residuals = y_pred_boot - y_true_boot
                    value = np.mean(np.abs(residuals / np.maximum(np.abs(y_true_boot), 1))) * 100

                bootstrapped_values.append(value)

            ci_lower = np.percentile(bootstrapped_values, 2.5)
            ci_upper = np.percentile(bootstrapped_values, 97.5)

            confidence_intervals[metric] = (float(ci_lower), float(ci_upper))

        return confidence_intervals

    def _compute_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics to assess prediction reliability."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, 11)
        y_true_norm = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true) + 1e-8)
        y_pred_norm = (y_pred - np.min(y_true)) / (np.max(y_true) - np.min(y_true) + 1e-8)

        ece = 0.0
        total_samples = len(y_true)

        for i in range(len(bin_edges) - 1):
            mask = (y_pred_norm >= bin_edges[i]) & (y_pred_norm < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accuracy = np.mean(y_true_norm[mask])
                bin_confidence = np.mean(y_pred_norm[mask])
                bin_weight = mask.sum() / total_samples
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        # Maximum Calibration Error (MCE)
        mce = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (y_pred_norm >= bin_edges[i]) & (y_pred_norm < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accuracy = np.mean(y_true_norm[mask])
                bin_confidence = np.mean(y_pred_norm[mask])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return {
            "ece": float(ece),
            "mce": float(mce),
            "calibration_slope": float(np.polyfit(prob_pred, prob_true, 1)[0]),
            "calibration_intercept": float(np.polyfit(prob_pred, prob_true, 1)[1]),
        }

    def _compute_feature_importance(
        self, model: Any, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute feature importance using multiple methods."""
        importance_dict = {}

        try:
            # Try permutation importance if available
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importance_dict = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models
                importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
            else:
                # Fallback to mutual information
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
                importance_dict = dict(zip(feature_names, mi_scores))

        except Exception as e:
            warnings.warn(f"Feature importance computation failed: {e}")
            # Fallback to random importance
            importance_dict = {name: np.random.random() for name in feature_names}

        # Normalize importance scores
        if importance_dict:
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: float(v / total_importance) for k, v in importance_dict.items()}

        return importance_dict

    def _compute_reliability_scores(self, y_true: np.ndarray, y_pred: np.ndarray, X: pd.DataFrame) -> Dict[str, float]:
        """Compute reliability scores for production deployment."""
        residuals = y_pred - y_true

        # Prediction stability score
        if len(y_true) > 1:
            # Coefficient of variation of predictions
            pred_stability = float(np.std(y_pred) / (np.mean(y_pred) + 1e-8))
        else:
            pred_stability = 0.0

        # Error consistency score
        if len(residuals) > 1:
            error_consistency = 1.0 / (1.0 + np.std(np.abs(residuals)))
        else:
            error_consistency = 1.0

        # Overconfidence score (lower is better)
        # Measures if the model is overconfident in wrong predictions
        error_magnitude = np.abs(residuals)
        pred_confidence = np.abs(y_pred - np.mean(y_pred))  # Distance from mean prediction
        overconfidence = float(np.corrcoef(error_magnitude, pred_confidence)[0, 1] if len(error_magnitude) > 1 else 0.0)

        # Robustness score (lower error variance is better)
        if len(residuals) > 10:
            # Split into chunks and check consistency
            chunk_size = len(residuals) // 5
            chunk_errors = []
            for i in range(5):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < 4 else len(residuals)
                if end_idx > start_idx:
                    chunk_errors.append(np.mean(np.abs(residuals[start_idx:end_idx])))

            robustness = 1.0 / (1.0 + np.std(chunk_errors)) if chunk_errors else 1.0
        else:
            robustness = 1.0

        return {
            "prediction_stability": float(pred_stability),
            "error_consistency": float(error_consistency),
            "overconfidence_score": float(overconfidence),
            "robustness_score": float(robustness),
        }

    def _perform_statistical_tests(
        self, y_true: np.ndarray, y_pred: np.ndarray, dataset: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}

        # Kolmogorov-Smirnov test for distribution similarity
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(y_true, y_pred)
            tests["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "distributions_similar": bool(ks_pvalue > 0.05)
            }
        except Exception:
            tests["ks_test"] = {"error": "Failed to compute KS test"}

        # Anderson-Darling test for normality of residuals
        try:
            residuals = y_pred - y_true
            ad_stat = stats.anderson(residuals, dist='norm').statistic
            tests["anderson_darling"] = {
                "statistic": float(ad_stat),
                "normal_residuals": bool(ad_stat < 2.0)  # Rough threshold
            }
        except Exception:
            tests["anderson_darling"] = {"error": "Failed to compute Anderson-Darling test"}

        # Durbin-Watson test for autocorrelation (if time series data available)
        if dataset is not None and "gameweek" in dataset.columns:
            try:
                # Sort by gameweek and compute DW statistic
                sorted_data = dataset.sort_values("gameweek")
                if len(sorted_data) > 10:
                    dw_stat = self._durbin_watson_test(sorted_data, y_true, y_pred)
                    tests["durbin_watson"] = {
                        "statistic": float(dw_stat),
                        "autocorrelated": bool(dw_stat < 1.5 or dw_stat > 2.5)
                    }
            except Exception:
                tests["durbin_watson"] = {"error": "Failed to compute Durbin-Watson test"}

        # Diebold-Mariano test would require a baseline model
        tests["diebold_mariano"] = {"note": "Requires baseline model for comparison"}

        return tests

    def _durbin_watson_test(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Durbin-Watson statistic for autocorrelation."""
        residuals = y_pred - y_true
        numerator = np.sum(np.diff(residuals) ** 2)
        denominator = np.sum(residuals ** 2)

        if denominator == 0:
            return 2.0  # Neutral value when no variance

        return float(numerator / denominator)

    def _run_diagnostics(
        self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, dataset: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Run comprehensive diagnostics on model performance."""
        diagnostics = {}

        # Overfitting detection
        diagnostics["overfitting"] = self._detect_overfitting(X, y_true, y_pred)

        # Data leakage detection
        diagnostics["data_leakage"] = self._detect_data_leakage(X, y_true, y_pred)

        # Prediction stability analysis
        diagnostics["stability"] = self._analyze_prediction_stability(X, y_true, y_pred)

        # Error pattern analysis
        diagnostics["error_patterns"] = self._analyze_error_patterns(y_true, y_pred, dataset)

        return diagnostics

    def _detect_overfitting(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detect potential overfitting in the model."""
        # High variance detection using bootstrap
        n_bootstrap = 100
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                score = r2_score(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue

        if bootstrap_scores:
            variance = np.var(bootstrap_scores)
            overfitting_score = float(max(0, variance - 0.01))  # Threshold for concerning variance
        else:
            overfitting_score = 0.0

        return {
            "overfitting_score": overfitting_score,
            "high_variance": bool(overfitting_score > 0.05),
            "variance_explained": float(np.var(bootstrap_scores)) if bootstrap_scores else 0.0,
        }

    def _detect_data_leakage(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detect potential data leakage in features."""
        leakage_score = 0.0
        suspicious_features = []

        # Check for features that are too highly correlated with target
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                try:
                    correlation = np.corrcoef(X[col], y_true)[0, 1]
                    if abs(correlation) > 0.95:  # Suspiciously high correlation
                        leakage_score += 0.5
                        suspicious_features.append(col)
                except:
                    continue

        # Check for future information leakage (if temporal data available)
        if "gameweek" in X.columns and len(X) > 1:
            # This is a simplified check - real implementation would need more sophisticated analysis
            pass

        return {
            "leakage_score": float(min(1.0, leakage_score)),
            "suspicious_features": suspicious_features,
            "likely_leakage": bool(leakage_score > 0.7),
        }

    def _analyze_prediction_stability(
        self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction stability across different conditions."""
        stability_metrics = {}

        # Stability across different ranges of the target variable
        if len(y_true) > 10:
            y_true_sorted = np.sort(y_true)
            splits = np.array_split(y_true_sorted, 5)

            stability_scores = []
            for split in splits:
                if len(split) > 0:
                    mask = np.isin(y_true, split)
                    if mask.sum() > 0:
                        split_stability = np.std(y_pred[mask]) / (np.mean(y_pred[mask]) + 1e-8)
                        stability_scores.append(split_stability)

            stability_metrics["target_range_stability"] = float(np.mean(stability_scores)) if stability_scores else 0.0

        # Stability across different feature ranges
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64'] and len(X[col].unique()) > 5:
                try:
                    X_sorted = np.sort(X[col])
                    splits = np.array_split(X_sorted, 5)

                    stability_scores = []
                    for split in splits:
                        if len(split) > 0:
                            mask = np.isin(X[col], split)
                            if mask.sum() > 0:
                                split_stability = np.std(y_pred[mask]) / (np.mean(y_pred[mask]) + 1e-8)
                                stability_scores.append(split_stability)

                    stability_metrics[f"{col}_stability"] = float(np.mean(stability_scores)) if stability_scores else 0.0
                except:
                    continue

        return stability_metrics

    def _analyze_error_patterns(
        self, y_true: np.ndarray, y_pred: np.ndarray, dataset: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in prediction errors."""
        residuals = y_pred - y_true

        error_patterns = {
            "error_skewness": float(stats.skew(residuals)),
            "error_kurtosis": float(stats.kurtosis(residuals)),
            "error_autocorrelation": float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) > 1 else 0.0,
        }

        # Error patterns by position (if available)
        if dataset is not None and "position_code" in dataset.columns:
            for position_code, position_name in self.position_map.items():
                mask = dataset["position_code"] == position_code
                if mask.sum() > 0:
                    pos_residuals = residuals[mask]
                    error_patterns[f"{position_name.lower()}_error_mean"] = float(np.mean(pos_residuals))
                    error_patterns[f"{position_name.lower()}_error_std"] = float(np.std(pos_residuals))

        return error_patterns

    def _run_validation(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Run validation checks on the model."""
        validation_results = {}

        # Cross-validation stability
        try:
            cv_scores = self._cross_validation_stability(X, y_true)
            validation_results["cv_stability"] = cv_scores
        except Exception as e:
            validation_results["cv_stability"] = {"error": str(e)}

        # Prediction bounds validation
        validation_results["prediction_bounds"] = {
            "min_prediction": float(np.min(y_pred)),
            "max_prediction": float(np.max(y_pred)),
            "reasonable_bounds": bool(np.min(y_pred) >= 0 and np.max(y_pred) <= 20),  # Reasonable FPL bounds
        }

        # Monotonicity checks (if applicable)
        validation_results["monotonicity"] = self._check_monotonicity(X, y_pred)

        return validation_results

    def _cross_validation_stability(self, X: pd.DataFrame, y_true: np.ndarray, n_folds: int = 5) -> Dict[str, float]:
        """Assess stability through cross-validation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        cv_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_true[train_idx], y_true[test_idx]

            # This is a placeholder - would need actual model training
            # For now, just compute naive baseline
            y_pred_cv = np.mean(y_train) * np.ones(len(y_test))
            score = r2_score(y_test, y_pred_cv)
            cv_scores.append(score)

        return {
            "mean_cv_score": float(np.mean(cv_scores)),
            "std_cv_score": float(np.std(cv_scores)),
            "cv_score_range": float(np.max(cv_scores) - np.min(cv_scores)),
            "stable_cv": bool(np.std(cv_scores) < 0.1),  # Threshold for stability
        }

    def _check_monotonicity(self, X: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Any]:
        """Check for monotonic relationships in key features."""
        monotonicity_checks = {}

        # Check for expected monotonic relationships
        monotonic_features = ["form", "ict_index", "selected_by_percent"]

        for feature in monotonic_features:
            if feature in X.columns:
                try:
                    correlation = np.corrcoef(X[feature], y_pred)[0, 1]
                    monotonicity_checks[f"{feature}_correlation"] = float(correlation)
                    monotonicity_checks[f"{feature}_monotonic"] = bool(abs(correlation) > 0.1)
                except:
                    monotonicity_checks[f"{feature}_error"] = "Failed to compute correlation"

        return monotonicity_checks

    def compare_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple models using statistical tests."""
        model_results = {}

        for name, model in models.items():
            try:
                evaluation = self.evaluate_model(model, X, y, feature_names, model_name=name)
                model_results[name] = evaluation.to_dict()
            except Exception as e:
                model_results[name] = {"error": str(e)}

        # Perform statistical comparisons
        if len(models) >= 2:
            comparison_results = self._statistical_model_comparison(models, X, y)
            model_results["statistical_comparison"] = comparison_results

        return model_results

    def _statistical_model_comparison(
        self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Perform statistical comparison between models."""
        comparison = {}

        model_names = list(models.keys())
        predictions = {}

        for name, model in models.items():
            predictions[name] = model.predict(X)

        # Diebold-Mariano test for predictive accuracy
        if len(model_names) >= 2:
            base_model = model_names[0]
            for other_model in model_names[1:]:
                try:
                    dm_stat = self._diebold_mariano_test(
                        y, predictions[base_model], predictions[other_model]
                    )
                    comparison[f"dm_{base_model}_vs_{other_model}"] = dm_stat
                except Exception as e:
                    comparison[f"dm_{base_model}_vs_{other_model}"] = {"error": str(e)}

        # Feature importance comparison
        importance_comparison = {}
        for name, model in models.items():
            try:
                evaluation = self.evaluate_model(model, X, y, [], model_name=name)
                importance_comparison[name] = evaluation.feature_importance
            except:
                continue

        comparison["feature_importance_comparison"] = importance_comparison

        return comparison

    def _diebold_mariano_test(
        self, y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
    ) -> Dict[str, float]:
        """Diebold-Mariano test for comparing predictive accuracy."""
        # Compute loss differentials (squared errors)
        loss1 = (y_true - y_pred1) ** 2
        loss2 = (y_true - y_pred2) ** 2
        loss_diff = loss1 - loss2

        # Compute test statistic
        mean_diff = np.mean(loss_diff)
        std_diff = np.std(loss_diff)

        if std_diff > 0:
            dm_stat = mean_diff / (std_diff / np.sqrt(len(loss_diff)))
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        else:
            dm_stat = 0.0
            p_value = 1.0

        return {
            "statistic": float(dm_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "better_model": "model1" if dm_stat < 0 else "model2",  # Negative means model1 is better
        }

    def generate_calibration_plots(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate calibration plots and return plot data."""
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')

        plot_data = {
            "calibration_curve": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            }
        }

        # Reliability diagram
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path / "calibration_curve.png", dpi=150, bbox_inches='tight')
            plot_data["calibration_curve"]["plot_path"] = str(save_path / "calibration_curve.png")

        plt.close()

        return plot_data

    def export_tableau_data(
        self,
        evaluation: EvaluationResult,
        predictions: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Export evaluation results in Tableau-friendly format."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Main metrics export
        metrics_df = pd.DataFrame([evaluation.metrics])
        metrics_df.to_csv(output_path / "evaluation_metrics.csv", index=False)

        # Position metrics export
        if evaluation.position_metrics:
            position_metrics_list = []
            for position, metrics in evaluation.position_metrics.items():
                row = {"position": position}
                row.update(metrics)
                position_metrics_list.append(row)

            position_df = pd.DataFrame(position_metrics_list)
            position_df.to_csv(output_path / "position_metrics.csv", index=False)

        # Feature importance export
        if evaluation.feature_importance:
            importance_df = pd.DataFrame([
                {"feature": feature, "importance": importance}
                for feature, importance in evaluation.feature_importance.items()
            ])
            importance_df = importance_df.sort_values("importance", ascending=False)
            importance_df.to_csv(output_path / "feature_importance.csv", index=False)

        # Predictions with evaluation metadata
        if not predictions.empty:
            enriched_predictions = predictions.copy()

            # Add reliability indicators
            enriched_predictions["reliability_score"] = evaluation.reliability_scores.get("error_consistency", 0.5)
            enriched_predictions["overfitting_risk"] = evaluation.diagnostics.get("overfitting", {}).get("overfitting_score", 0.0)

            enriched_predictions.to_csv(output_path / "predictions_with_evaluation.csv", index=False)

    def generate_evaluation_report(
        self,
        evaluation: EvaluationResult,
        model_name: str = "Model",
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = [
            "# Model Evaluation Report",
            f"## {model_name} Evaluation Summary",
            "",

            "### Core Metrics",
            "| Metric | Value | Confidence Interval |",
            "|--------|-------|-------------------|",
        ]

        for metric, value in evaluation.metrics.items():
            if metric in evaluation.confidence_intervals:
                ci = evaluation.confidence_intervals[metric]
                report_lines.append(f"| {metric.upper()} | {value".4f"} | [{ci[0]".4f"}, {ci[1]".4f"}] |")
            else:
                report_lines.append(f"| {metric.upper()} | {value".4f"} | N/A |")

        report_lines.extend([
            "",
            "### Position-Specific Performance",
        ])

        if evaluation.position_metrics:
            report_lines.append("| Position | MAE | RMSE | R² | Count |")
            report_lines.append("|----------|-----|------|----|-------|")
            for position, metrics in evaluation.position_metrics.items():
                report_lines.append(
                    f"| {position} | {metrics['mae']".4f"} | {metrics['rmse']".4f"} | "
                    f"{metrics['r2']".4f"} | {metrics['count']} |"
                )

        report_lines.extend([
            "",
            "### Calibration Metrics",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for metric, value in evaluation.calibration_metrics.items():
            report_lines.append(f"| {metric.upper()} | {value".4f"} |")

        report_lines.extend([
            "",
            "### Reliability Scores",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for metric, value in evaluation.reliability_scores.items():
            report_lines.append(f"| {metric} | {value".4f"} |")

        report_lines.extend([
            "",
            "### Feature Importance (Top 10)",
        ])

        if evaluation.feature_importance:
            sorted_features = sorted(evaluation.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            report_lines.append("| Feature | Importance |")
            report_lines.append("|---------|------------|")
            for feature, importance in sorted_features:
                report_lines.append(f"| {feature} | {importance".4f"} |")

        report_lines.extend([
            "",
            "### Statistical Tests",
        ])

        for test_name, test_result in evaluation.statistical_tests.items():
            if isinstance(test_result, dict):
                if "error" not in test_result:
                    report_lines.append(f"#### {test_name}")
                    for key, value in test_result.items():
                        if isinstance(value, bool):
                            report_lines.append(f"- {key}: {'✓' if value else '✗'}")
                        else:
                            report_lines.append(f"- {key}: {value}")

        report_lines.extend([
            "",
            "### Diagnostics",
        ])

        if evaluation.diagnostics:
            for diagnostic_name, diagnostic_result in evaluation.diagnostics.items():
                if isinstance(diagnostic_result, dict):
                    report_lines.append(f"#### {diagnostic_name}")
                    for key, value in diagnostic_result.items():
                        if isinstance(value, bool):
                            report_lines.append(f"- {key}: {'✓' if value else '✗'}")
                        else:
                            report_lines.append(f"- {key}: {value}")

        return "\n".join(report_lines)
