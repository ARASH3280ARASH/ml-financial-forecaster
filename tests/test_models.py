"""Tests for model architectures."""

import numpy as np
import pandas as pd
import pytest

from src.models.base_model import BaseModel, ModelMetrics
from src.models.gradient_boosting import GradientBoostingModel
from src.models.ensemble_model import EnsembleModel, EnsembleMethod
from src.models.model_registry import ModelRegistry


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    rng = np.random.default_rng(42)
    n = 1000
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3 + rng.standard_normal(n) * 0.1)
    return X, y


class TestBaseModel:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModel("test")

    def test_model_metrics_summary(self):
        metrics = ModelMetrics(rmse=0.01, mae=0.008, r2=0.85, directional_accuracy=0.55)
        assert "RMSE" in metrics.summary()
        assert "0.85" in metrics.summary()


class TestGradientBoosting:
    def test_xgboost_fit_predict(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel("xgb_test", backend="xgboost")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert model.is_fitted

    def test_lightgbm_fit_predict(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel("lgb_test", backend="lightgbm")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_evaluate_returns_metrics(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel("xgb_eval", backend="xgboost")
        model.fit(X[:800], y[:800])
        metrics = model.evaluate(X[800:], y[800:])
        assert isinstance(metrics, ModelMetrics)
        assert metrics.rmse >= 0

    def test_feature_importance(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel("xgb_imp", backend="xgboost")
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        model = GradientBoostingModel("unfitted")
        with pytest.raises(RuntimeError):
            model.predict(X)


class TestEnsembleModel:
    def test_mean_ensemble(self, sample_data):
        X, y = sample_data
        m1 = GradientBoostingModel("xgb", backend="xgboost")
        m2 = GradientBoostingModel("lgb", backend="lightgbm")
        ensemble = EnsembleModel("ens", models=[m1, m2], method=EnsembleMethod.MEAN)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(X)

    def test_median_ensemble(self, sample_data):
        X, y = sample_data
        m1 = GradientBoostingModel("xgb", backend="xgboost")
        m2 = GradientBoostingModel("lgb", backend="lightgbm")
        ensemble = EnsembleModel("ens_med", models=[m1, m2], method=EnsembleMethod.MEDIAN)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(X)

    def test_model_contributions(self, sample_data):
        X, y = sample_data
        m1 = GradientBoostingModel("xgb", backend="xgboost")
        m2 = GradientBoostingModel("lgb", backend="lightgbm")
        ensemble = EnsembleModel("ens", models=[m1, m2])
        ensemble.fit(X, y)
        contribs = ensemble.get_model_contributions(X)
        assert contribs.shape == (len(X), 2)

    def test_empty_ensemble_raises(self, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel("empty")
        with pytest.raises(ValueError):
            ensemble.fit(X, y)


class TestModelRegistry:
    def test_register_and_retrieve(self, sample_data, tmp_path):
        X, y = sample_data
        registry = ModelRegistry(str(tmp_path / "registry"))

        model = GradientBoostingModel("xgb_reg", backend="xgboost")
        model.fit(X, y)
        model.evaluate(X, y)

        mv = registry.register(model, version="1.0.0", tags=["test"])
        assert mv.version == "1.0.0"
        assert registry.version_count == 1

    def test_get_best_model(self, sample_data, tmp_path):
        X, y = sample_data
        registry = ModelRegistry(str(tmp_path / "registry"))

        for i, depth in enumerate([3, 6, 9]):
            model = GradientBoostingModel(f"xgb_{depth}", backend="xgboost", params={"max_depth": depth})
            model.fit(X[:800], y[:800])
            model.evaluate(X[800:], y[800:])
            registry.register(model, version=f"1.{i}.0")

        best = registry.get_best(f"xgb_{3}", metric="rmse")
        assert best is not None

    def test_unfitted_model_raises(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "registry"))
        model = GradientBoostingModel("unfitted")
        with pytest.raises(ValueError):
            registry.register(model, version="1.0")
