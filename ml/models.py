"""Enhanced ML models with ensemble learning and position-specific optimisations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:  # Optional dependency for traditional ML components
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold
except ImportError:  # pragma: no cover - allow operation without scikit-learn
    class _BaseEstimatorFallback:
        pass

    class _RegressorMixinFallback:
        pass

    BaseEstimator = _BaseEstimatorFallback  # type: ignore[assignment]
    RegressorMixin = _RegressorMixinFallback  # type: ignore[assignment]
    GradientBoostingRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]

    def mean_absolute_error(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    def mean_squared_error(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true_arr - y_pred_arr) ** 2))

    def r2_score(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    class KFold:  # type: ignore[override]
        def __init__(self, n_splits: int, shuffle: bool = False, random_state: Optional[int] = None):
            if n_splits < 2:
                raise ValueError("n_splits must be at least 2")
            self.n_splits = n_splits

        def split(self, X: Union[pd.DataFrame, np.ndarray]):
            n_samples = len(X)
            indices = np.arange(n_samples)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: n_samples % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                val_indices = indices[start:stop]
                train_indices = np.concatenate([indices[:start], indices[stop:]])
                yield train_indices, val_indices
                current = stop

try:  # Optional dependency
    import lightgbm as lgb
except ImportError:  # pragma: no cover - handled gracefully at runtime
    lgb = None  # type: ignore[assignment]

try:  # Optional dependency
    import xgboost as xgb
except ImportError:  # pragma: no cover - handled gracefully at runtime
    xgb = None  # type: ignore[assignment]

try:  # Optional dependency
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, InputLayer
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Sequential = None  # type: ignore[assignment]
    BatchNormalization = None  # type: ignore[assignment]
    Dense = None  # type: ignore[assignment]
    Dropout = None  # type: ignore[assignment]
    InputLayer = None  # type: ignore[assignment]
    Adam = None  # type: ignore[assignment]
    EarlyStopping = None  # type: ignore[assignment]

from .config import PipelineConfig


class PositionSpecificModel(BaseEstimator, RegressorMixin):
    """Ensemble model with position-specific optimizations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        if xgb is None or lgb is None:
            missing = []
            if xgb is None:
                missing.append("xgboost")
            if lgb is None:
                missing.append("lightgbm")
            raise ImportError(
                "Optional dependencies required for PositionSpecificModel are missing: "
                + ", ".join(missing)
            )

        if GradientBoostingRegressor is None or Ridge is None:
            raise ImportError("scikit-learn is required for PositionSpecificModel")

        self.models: Dict[str, Any] = {}
        self.meta_model: Optional[GradientBoostingRegressor] = None
        self.position_map = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}
        self.feature_columns: Optional[List[str]] = None
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize models for each position and meta-learner."""

        # Position-specific models with optimized hyperparameters
        for position_code, position_name in self.position_map.items():
            if position_code == 0:  # Goalkeeper
                self.models[position_name] = Ridge(
                    alpha=1.0, random_state=self.config.random_state
                )
            elif position_code == 1:  # Defender
                self.models[position_name] = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                )
            elif position_code == 2:  # Midfielder
                self.models[position_name] = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                )
            else:  # Forward
                self.models[position_name] = lgb.LGBMRegressor(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.config.random_state,
                )

        # Meta-learner for ensemble predictions
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.config.random_state,
        )

    def _get_position_from_code(self, position_code: int) -> str:
        """Get position name from position code."""
        return self.position_map.get(position_code, "Midfielder")

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply position-specific feature preprocessing."""
        X_processed = X.copy()

        # Add position-specific interaction features
        if "position_code" in X_processed.columns:
            for col in X_processed.columns:
                if col not in ["position_code", "player_id", "gameweek", "season"]:
                    # Create position-specific features
                    X_processed[f"{col}_gk"] = X_processed[col] * (X_processed["position_code"] == 0).astype(int)
                    X_processed[f"{col}_def"] = X_processed[col] * (X_processed["position_code"] == 1).astype(int)
                    X_processed[f"{col}_mid"] = X_processed[col] * (X_processed["position_code"] == 2).astype(int)
                    X_processed[f"{col}_fwd"] = X_processed[col] * (X_processed["position_code"] == 3).astype(int)

        return X_processed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PositionSpecificModel:
        """Fit the ensemble model with position-specific training."""
        if "position_code" not in X.columns:
            raise ValueError("position_code column required for position-specific training")

        self.feature_columns = list(X.columns)
        X_processed = self._preprocess_features(X)

        # Train position-specific models
        for position_name, model in self.models.items():
            mask = X["position_code"] == list(self.position_map.keys())[list(self.position_map.values()).index(position_name)]
            if mask.sum() > 0:
                X_pos = X_processed[mask]
                y_pos = y[mask]
                model.fit(X_pos, y_pos)

        # Create meta-features for stacking
        meta_features = self._create_meta_features(X_processed)

        # Train meta-learner
        if self.meta_model is not None:
            self.meta_model.fit(meta_features, y)

        return self

    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create meta-features for stacking ensemble."""
        meta_features_list = []

        for position_name, model in self.models.items():
            # Get predictions from each position-specific model
            predictions = model.predict(X)
            meta_features_list.append(predictions)

        # Add original features that are strong predictors
        important_features = [
            "event_points_lag_1",
            "expected_goal_involvements",
            "form",
            "selected_by_percent",
            "ict_index"
        ]

        for feature in important_features:
            if feature in X.columns:
                meta_features_list.append(X[feature].values)

        return np.column_stack(meta_features_list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the ensemble model."""
        if isinstance(X, np.ndarray):
            if not self.feature_columns:
                raise ValueError("Model has not been fitted with feature metadata")
            X = pd.DataFrame(X, columns=self.feature_columns)

        if "position_code" not in X.columns:
            raise ValueError("position_code column required for prediction")

        X_processed = self._preprocess_features(X)
        meta_features = self._create_meta_features(X_processed)

        if self.meta_model is not None:
            return self.meta_model.predict(meta_features)
        else:
            # Fallback to simple averaging if meta-model not trained
            predictions = []
            for position_name, model in self.models.items():
                pred = model.predict(X_processed)
                predictions.append(pred)

            return np.mean(predictions, axis=0)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "mape": np.mean(np.abs((y - predictions) / np.maximum(np.abs(y), 1))) * 100,
            "bias": np.mean(predictions - y),
            "median_ae": np.median(np.abs(y - predictions)),
        }

        # Position-specific metrics
        for position_code, position_name in self.position_map.items():
            mask = X["position_code"] == position_code
            if mask.sum() > 0:
                y_pos = y[mask]
                pred_pos = predictions[mask]
                metrics[f"{position_name.lower()}_mae"] = mean_absolute_error(y_pos, pred_pos)
                metrics[f"{position_name.lower()}_count"] = mask.sum()

        return metrics


class DeepEnsembleModel(BaseEstimator, RegressorMixin):
    """Deep learning ensemble model for advanced predictions."""

    def __init__(self, config: PipelineConfig):
        if None in {Sequential, Dense, Dropout, BatchNormalization, InputLayer, Adam}:
            raise ImportError(
                "TensorFlow with Keras layers is required to use DeepEnsembleModel"
            )

        self.config = config
        self.models: List[Sequential] = []
        self.meta_model: Optional[Sequential] = None
        self._base_architectures: List[List[Tuple[str, Dict[str, Any]]]] = [
            [
                ("dense", {"units": 128}),
                ("batchnorm", {}),
                ("dropout", {"rate": 0.3}),
                ("dense", {"units": 64}),
                ("batchnorm", {}),
                ("dropout", {"rate": 0.2}),
                ("dense", {"units": 32}),
            ],
            [
                ("dense", {"units": 256}),
                ("batchnorm", {}),
                ("dropout", {"rate": 0.4}),
                ("dense", {"units": 128}),
                ("batchnorm", {}),
                ("dropout", {"rate": 0.3}),
                ("dense", {"units": 64}),
            ],
            [
                ("dense", {"units": 64}),
                ("batchnorm", {}),
                ("dense", {"units": 128}),
                ("batchnorm", {}),
                ("dropout", {"rate": 0.2}),
                ("dense", {"units": 64}),
                ("batchnorm", {}),
                ("dense", {"units": 32}),
            ],
        ]
        self._base_epochs = 100
        self._meta_epochs = 60
        self._base_patience = 10
        self._meta_patience = 8

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DeepEnsembleModel:
        """Fit the deep ensemble model."""

        X_array = np.asarray(X, dtype=float)
        y_array = y.to_numpy(dtype=float)

        if X_array.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional")

        n_samples, n_features = X_array.shape
        if n_samples == 0:
            raise ValueError("Cannot train DeepEnsembleModel on an empty dataset")

        meta_features = self._generate_oof_meta_features(X_array, y_array)

        # Train final base models on full data for inference
        self.models = []
        validation_split = 0.2 if n_samples >= 20 else 0.0
        for architecture in self._base_architectures:
            model = self._build_base_model(n_features, architecture)
            self._compile_model(model)
            callbacks = self._get_callbacks(self._base_patience, validation_split > 0)
            model.fit(
                X_array,
                y_array,
                epochs=self._base_epochs,
                batch_size=32,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0,
            )
            self.models.append(model)

        # Train meta-learner on stacked predictions
        self.meta_model = self._build_meta_model(meta_features.shape[1])
        self._compile_model(self.meta_model)
        meta_callbacks = self._get_callbacks(self._meta_patience, validation_split > 0)
        self.meta_model.fit(
            meta_features,
            y_array,
            epochs=self._meta_epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=meta_callbacks,
            verbose=0,
        )

        return self

    def _generate_oof_meta_features(self, X_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Create out-of-fold predictions for the meta-learner to avoid leakage."""

        n_samples, n_features = X_array.shape
        n_models = len(self._base_architectures)
        meta_features = np.zeros((n_samples, n_models), dtype=float)

        n_splits = min(5, n_samples)
        if n_splits < 2:
            # Fallback: train quick models on full data to obtain features
            for idx, architecture in enumerate(self._base_architectures):
                temp_model = self._build_base_model(n_features, architecture)
                self._compile_model(temp_model)
                callbacks = self._get_callbacks(max(1, self._base_patience // 2), False)
                temp_model.fit(
                    X_array,
                    y_array,
                    epochs=max(1, self._base_epochs // 2),
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0,
                )
                meta_features[:, idx] = temp_model.predict(X_array, verbose=0).flatten()
            return meta_features

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        for train_idx, val_idx in kf.split(X_array):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            for model_idx, architecture in enumerate(self._base_architectures):
                temp_model = self._build_base_model(n_features, architecture)
                self._compile_model(temp_model)
                callbacks = self._get_callbacks(self._base_patience, True)
                temp_model.fit(
                    X_train,
                    y_train,
                    epochs=self._base_epochs,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0,
                )
                meta_features[val_idx, model_idx] = temp_model.predict(X_val, verbose=0).flatten()

        return meta_features

    def _build_base_model(
        self, input_dim: int, architecture: List[Tuple[str, Dict[str, Any]]]
    ) -> Sequential:
        model = Sequential()
        model.add(InputLayer(input_shape=(input_dim,)))
        for layer_type, params in architecture:
            if layer_type == "dense":
                model.add(Dense(params["units"], activation=params.get("activation", "relu")))
            elif layer_type == "batchnorm":
                model.add(BatchNormalization())
            elif layer_type == "dropout":
                model.add(Dropout(params["rate"]))
        model.add(Dense(1))
        return model

    def _build_meta_model(self, input_dim: int) -> Sequential:
        model = Sequential()
        model.add(InputLayer(input_shape=(input_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        return model

    def _compile_model(self, model: Sequential) -> None:
        if Adam is None:
            raise ImportError("TensorFlow optimiser unavailable; ensure tensorflow is installed")
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error'],
        )

    def _get_callbacks(self, patience: int, use_validation: bool) -> List[Any]:
        if EarlyStopping is None or patience <= 0:
            return []
        monitor = 'val_loss' if use_validation else 'loss'
        return [EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)]

    def _create_meta_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Base models have not been trained")
        X_array = np.asarray(X, dtype=float)
        predictions = [model.predict(X_array, verbose=0).flatten() for model in self.models]
        return np.column_stack(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using deep ensemble."""
        if self.meta_model is None:
            raise RuntimeError("Meta-learner has not been trained")
        meta_features = self._create_meta_features(X)
        return self.meta_model.predict(meta_features, verbose=0).flatten()

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate deep ensemble performance."""
        predictions = self.predict(X)

        return {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "mape": np.mean(np.abs((y - predictions) / np.maximum(np.abs(y), 1))) * 100,
        }
