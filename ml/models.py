"""Enhanced ML models with ensemble learning and position-specific optimizations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .config import PipelineConfig


class PositionSpecificModel(BaseEstimator, RegressorMixin):
    """Ensemble model with position-specific optimizations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models: Dict[str, Union[Ridge, xgb.XGBRegressor, lgb.LGBMRegressor]] = {}
        self.meta_model: Optional[GradientBoostingRegressor] = None
        self.position_map = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}
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

        X_processed = self._preprocess_features(X)

        # Train position-specific models
        for position_name, model in self.models.items():
            mask = X["position_code"] == list(self.position_map.keys())[list(self.position_map.values()).index(position_name)]
            if mask.sum() > 0:
                X_pos = X_processed[mask]
                y_pos = y[mask]
                model.fit(X_pos, y_pos)

        # Create meta-features for stacking
        meta_features = self._create_meta_features(X_processed, y)

        # Train meta-learner
        if self.meta_model is not None:
            self.meta_model.fit(meta_features, y)

        return self

    def _create_meta_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
        if "position_code" not in X.columns:
            raise ValueError("position_code column required for prediction")

        X_processed = self._preprocess_features(X)
        meta_features = self._create_meta_features(X_processed, None)

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
        self.config = config
        self.models: List[Sequential] = []
        self.meta_model: Optional[Sequential] = None
        self._build_models()

    def _build_models(self) -> None:
        """Build multiple neural network architectures."""

        # Model 1: Dense network for general patterns
        model1 = Sequential([
            Dense(128, activation='relu', input_dim=None),  # Will be set dynamically
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Model 2: Wider network for complex interactions
        model2 = Sequential([
            Dense(256, activation='relu', input_dim=None),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        # Model 3: Deeper network for hierarchical features
        model3 = Sequential([
            Dense(64, activation='relu', input_dim=None),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        self.models = [model1, model2, model3]

        # Meta-learner
        self.meta_model = Sequential([
            Dense(64, activation='relu', input_dim=4),  # 3 models + 1 target
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DeepEnsembleModel:
        """Fit the deep ensemble model."""
        # Compile models
        for model in self.models + [self.meta_model]:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )

        # Update input dimensions
        input_dim = X.shape[1]
        for model in self.models:
            model.layers[0].input_dim = input_dim

        self.meta_model.layers[0].input_dim = len(self.models) + 1

        # Train individual models
        for i, model in enumerate(self.models):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )

            model.fit(
                X, y,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

        # Create meta-features and train meta-learner
        meta_features = self._create_meta_features(X)
        meta_target = np.column_stack([meta_features, y.values])

        self.meta_model.fit(
            meta_target,
            y,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
            verbose=0
        )

        return self

    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create meta-features from individual model predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0).flatten()
            predictions.append(pred)
        return np.column_stack(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using deep ensemble."""
        meta_features = self._create_meta_features(X)
        meta_input = np.column_stack([meta_features, np.zeros((X.shape[0], 1))])  # Placeholder for target
        return self.meta_model.predict(meta_input, verbose=0).flatten()

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate deep ensemble performance."""
        predictions = self.predict(X)

        return {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "mape": np.mean(np.abs((y - predictions) / np.maximum(np.abs(y), 1))) * 100,
        }
