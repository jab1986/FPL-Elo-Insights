import math

import pytest

from ml import model as model_module
from ml.model import RidgeRegressionModel


def test_ridge_manual_backend_produces_predictions(monkeypatch):
    features = [
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ]
    target = [2.0, 1.5, 3.5, 4.0]

    model = RidgeRegressionModel(alpha=0.5)
    monkeypatch.setattr(model_module, "RidgeClass", None)
    monkeypatch.setattr(model_module, "np", None)

    fitted = model.fit(features, target)
    predictions = fitted.predict(features)

    assert len(predictions) == len(target)
    assert model.coef_ is not None
    assert model._backend_model is None
    for value in predictions:
        assert not math.isnan(value)


def test_ridge_prefers_sklearn_when_available(monkeypatch):
    class DummyRidge:
        def __init__(self, alpha: float, fit_intercept: bool = True):
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None

        def fit(self, features, target):
            self.coef_ = [float(sum(col) / len(features)) for col in zip(*features)]
            self.intercept_ = float(sum(target) / len(target))

    monkeypatch.setattr(model_module, "RidgeClass", DummyRidge)
    monkeypatch.setattr(model_module, "np", None)

    model = RidgeRegressionModel(alpha=2.0)
    features = [[1.0, 2.0], [0.5, 1.5], [2.0, 0.5]]
    target = [4.0, 3.0, 5.5]

    fitted = model.fit(features, target)

    assert isinstance(fitted._backend_model, DummyRidge)
    expected_coef = [sum(column) / len(features) for column in zip(*features)]
    expected_intercept = sum(target) / len(target)
    assert model.coef_ == pytest.approx(expected_coef)
    assert model.intercept_ == pytest.approx(expected_intercept)

    predictions = model.predict(features)
    assert len(predictions) == len(target)
