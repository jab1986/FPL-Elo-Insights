"""Lightweight regression models used by the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RidgeRegressionModel:
    """Simple ridge regression implemented with numpy."""

    alpha: float = 1.0
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RidgeRegressionModel":
        """Fit the model using the normal equation with L2 regularisation."""

        if features.ndim != 2:
            raise ValueError("Features must be a 2D array")

        n_samples, n_features = features.shape
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)

        x_augmented = np.hstack([x, np.ones((n_samples, 1))])
        identity = np.eye(n_features + 1)
        identity[-1, -1] = 0.0  # do not regularise the intercept

        regularised = x_augmented.T @ x_augmented + self.alpha * identity
        solution = np.linalg.solve(regularised, x_augmented.T @ y)

        self.coef_ = solution[:-1]
        self.intercept_ = float(solution[-1])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for the provided feature matrix."""

        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted")

        x = np.asarray(features, dtype=float)
        return x @ self.coef_ + self.intercept_

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, float]:
        """Return a serialisable representation of the learned parameters."""

        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted")

        return {name: float(weight) for name, weight in zip(feature_names, self.coef_)}

