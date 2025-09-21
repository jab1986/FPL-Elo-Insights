"""Lightweight regression models used by the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .compat import RidgeClass, numpy as np


def _gaussian_elimination(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    for i in range(size):
        pivot = max(range(i, size), key=lambda row: abs(matrix[row][i]))
        if abs(matrix[pivot][i]) < 1e-12:
            raise ValueError("Singular matrix encountered during ridge solve")
        if pivot != i:
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            vector[i], vector[pivot] = vector[pivot], vector[i]

        pivot_value = matrix[i][i]
        for j in range(i, size):
            matrix[i][j] /= pivot_value
        vector[i] /= pivot_value

        for row in range(i + 1, size):
            factor = matrix[row][i]
            if abs(factor) < 1e-12:
                continue
            for col in range(i, size):
                matrix[row][col] -= factor * matrix[i][col]
            vector[row] -= factor * vector[i]

    solution = [0.0] * size
    for i in range(size - 1, -1, -1):
        value = vector[i]
        for col in range(i + 1, size):
            value -= matrix[i][col] * solution[col]
        solution[i] = value
    return solution


@dataclass
class RidgeRegressionModel:
    """Ridge regression with optional numpy/sklearn backends."""

    alpha: float = 1.0
    coef_: List[float] | None = None
    intercept_: float = 0.0
    _backend_model: object | None = None

    def fit(self, features: Sequence[Sequence[float]], target: Sequence[float]) -> "RidgeRegressionModel":
        if not features:
            raise ValueError("Features must not be empty")
        n_samples = len(features)
        n_features = len(features[0])
        if any(len(row) != n_features for row in features):
            raise ValueError("All feature rows must have the same length")
        if len(target) != n_samples:
            raise ValueError("Target length must match feature rows")

        if RidgeClass is not None:
            return self._fit_with_sklearn(features, target)
        if np is not None:
            return self._fit_with_numpy(features, target)
        return self._fit_with_pure_python(features, target)

    def _fit_with_sklearn(
        self, features: Sequence[Sequence[float]], target: Sequence[float]
    ) -> "RidgeRegressionModel":
        assert RidgeClass is not None  # for type checkers
        model = RidgeClass(alpha=self.alpha, fit_intercept=True)
        model.fit(features, target)
        self.coef_ = [float(value) for value in getattr(model, "coef_", [])]
        self.intercept_ = float(getattr(model, "intercept_", 0.0))
        self._backend_model = model
        return self

    def _fit_with_numpy(
        self, features: Sequence[Sequence[float]], target: Sequence[float]
    ) -> "RidgeRegressionModel":
        assert np is not None  # for type checkers
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Features must be a 2D structure")
        target_vector = np.asarray(target, dtype=float)
        if target_vector.shape[0] != matrix.shape[0]:
            raise ValueError("Target length must match feature rows")

        ones = np.ones((matrix.shape[0], 1), dtype=float)
        augmented = np.concatenate([matrix, ones], axis=1)
        gram = augmented.T @ augmented
        ridge = np.eye(augmented.shape[1], dtype=float)
        ridge[-1, -1] = 0.0
        gram += self.alpha * ridge
        rhs = augmented.T @ target_vector

        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise ValueError("Singular matrix encountered during ridge solve") from exc

        self.coef_ = [float(value) for value in solution[:-1]]
        self.intercept_ = float(solution[-1])
        self._backend_model = None
        return self

    def _fit_with_pure_python(
        self, features: Sequence[Sequence[float]], target: Sequence[float]
    ) -> "RidgeRegressionModel":
        n_features = len(features[0])
        augmented_size = n_features + 1
        matrix = [[0.0 for _ in range(augmented_size)] for _ in range(augmented_size)]
        vector = [0.0 for _ in range(augmented_size)]

        for row, y in zip(features, target):
            extended = list(row) + [1.0]
            for i in range(augmented_size):
                vector[i] += extended[i] * y
                for j in range(augmented_size):
                    matrix[i][j] += extended[i] * extended[j]

        for i in range(n_features):
            matrix[i][i] += self.alpha

        solution = _gaussian_elimination(matrix, vector)
        self.coef_ = solution[:-1]
        self.intercept_ = solution[-1]
        self._backend_model = None
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> List[float]:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted")
        predictions: List[float] = []
        for row in features:
            if len(row) != len(self.coef_):
                raise ValueError("Feature row has unexpected length")
            value = sum(weight * feature for weight, feature in zip(self.coef_, row)) + self.intercept_
            predictions.append(value)
        return predictions

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, float]:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted")
        return {name: float(weight) for name, weight in zip(feature_names, self.coef_)}
