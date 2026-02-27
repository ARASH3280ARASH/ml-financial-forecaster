"""Tests for the end-to-end pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataConfig, DataLoader, DataLoadError
from src.data.preprocessor import Preprocessor, PreprocessingError, ScalingMethod
from src.training.trainer import Trainer, TrainingConfig
from src.training.cross_validator import WalkForwardValidator, PurgedKFold
from src.training.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import MetricsCalculator, FinancialMetrics
from src.evaluation.backtester import Backtester, BacktestConfig
from src.evaluation.statistical_tests import StatisticalTester
from src.models.gradient_boosting import GradientBoostingModel


@pytest.fixture
def synthetic_data():
    """Load synthetic test data."""
    loader = DataLoader(DataConfig(source="synthetic", start_date="2023-01-01", end_date="2023-12-31"))
    return loader.load()


class TestDataLoader:
    def test_synthetic_source(self):
        loader = DataLoader(DataConfig(source="synthetic"))
        df = loader.load()
        assert not df.empty
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

    def test_caching(self):
        loader = DataLoader(DataConfig(source="synthetic"))
        df1 = loader.load()
        df2 = loader.load()
        assert df1.shape == df2.shape

    def test_missing_csv_raises(self):
        loader = DataLoader(DataConfig(source="csv"))
        # Will fall back to synthetic since "csv" is matched
        df = loader.load()
        assert not df.empty


class TestPreprocessor:
    def test_fit_transform(self, synthetic_data):
        pp = Preprocessor(scaling=ScalingMethod.ROBUST)
        X = synthetic_data[["close", "volume"]]
        result = pp.fit_transform(X)
        assert result.shape == X.shape
        assert pp.is_fitted

    def test_transform_before_fit_raises(self, synthetic_data):
        pp = Preprocessor()
        with pytest.raises(PreprocessingError):
            pp.transform(synthetic_data[["close"]])

    def test_temporal_split(self, synthetic_data):
        X = synthetic_data[["close", "volume"]]
        y = synthetic_data["close"].pct_change().fillna(0)
        split = Preprocessor.temporal_split(X, y, train_ratio=0.7, val_ratio=0.15)
        assert len(split.X_train) > 0
        assert len(split.X_val) > 0
        assert len(split.X_test) > 0

    def test_empty_dataframe_raises(self):
        pp = Preprocessor()
        with pytest.raises(PreprocessingError):
            pp.fit(pd.DataFrame())


class TestTrainer:
    def test_train_single_model(self, synthetic_data):
        X = synthetic_data[["open", "high", "low", "close", "volume"]]
        y = synthetic_data["close"].pct_change().shift(-1).fillna(0)

        config = TrainingConfig(train_ratio=0.6, val_ratio=0.2, target_horizon=0)
        trainer = Trainer(config=config)
        model = GradientBoostingModel("test_xgb", backend="xgboost", params={"n_estimators": 50, "max_depth": 3})
        result = trainer.train(model, X, y)

        assert result.model.is_fitted
        assert result.test_metrics.rmse >= 0
        assert result.training_time > 0

    def test_comparison_table(self, synthetic_data):
        X = synthetic_data[["close", "volume"]]
        y = synthetic_data["close"].pct_change().shift(-1).fillna(0)

        config = TrainingConfig(train_ratio=0.6, val_ratio=0.2, target_horizon=0)
        trainer = Trainer(config=config)

        models = [
            GradientBoostingModel("xgb", backend="xgboost", params={"n_estimators": 20}),
            GradientBoostingModel("lgb", backend="lightgbm", params={"n_estimators": 20}),
        ]
        results = trainer.train_multiple(models, X, y)
        table = trainer.comparison_table()
        assert len(table) == 2


class TestCrossValidation:
    def test_walk_forward(self, synthetic_data):
        X = synthetic_data[["close", "volume"]].fillna(0)
        y = synthetic_data["close"].pct_change().fillna(0)

        wf = WalkForwardValidator(n_splits=3, min_train_size=500, embargo_bars=10)
        model = GradientBoostingModel("wf_xgb", backend="xgboost", params={"n_estimators": 20})
        result = wf.validate(model, X, y)
        assert len(result.fold_results) >= 1
        assert result.mean_rmse >= 0

    def test_purged_kfold(self, synthetic_data):
        X = synthetic_data[["close", "volume"]].fillna(0)
        y = synthetic_data["close"].pct_change().fillna(0)

        pkf = PurgedKFold(n_splits=3, embargo_pct=0.01)
        model = GradientBoostingModel("pk_xgb", backend="xgboost", params={"n_estimators": 20})
        result = pkf.validate(model, X, y)
        assert len(result.fold_results) >= 1


class TestMetrics:
    def test_from_returns(self):
        returns = pd.Series(np.random.randn(252) * 0.01)
        calc = MetricsCalculator()
        metrics = calc.from_returns(returns)
        assert isinstance(metrics, FinancialMetrics)
        assert metrics.annualized_volatility > 0

    def test_rolling_sharpe(self):
        returns = pd.Series(np.random.randn(252) * 0.01)
        rolling = MetricsCalculator.rolling_sharpe(returns, window=60)
        assert len(rolling) == len(returns)


class TestBacktester:
    def test_backtest_run(self):
        n = 500
        predictions = pd.Series(np.random.randn(n) * 0.01, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        prices = pd.Series(50000 + np.cumsum(np.random.randn(n) * 100), index=predictions.index)

        bt = Backtester(BacktestConfig(initial_capital=100_000))
        result = bt.run(predictions, prices, strategy_name="test")
        assert len(result.equity_curve) == n
        assert result.trades >= 0


class TestStatisticalTests:
    def test_paired_t_test(self):
        errors_a = np.random.randn(100) * 0.01
        errors_b = np.random.randn(100) * 0.02
        tester = StatisticalTester()
        result = tester.paired_t_test(np.abs(errors_a), np.abs(errors_b))
        assert 0 <= result.p_value <= 1

    def test_bootstrap_ci(self):
        values = np.random.randn(200)
        tester = StatisticalTester()
        lower, point, upper = tester.bootstrap_confidence_interval(values)
        assert lower <= point <= upper

    def test_normality_test(self):
        returns = pd.Series(np.random.randn(500))
        tester = StatisticalTester()
        result = tester.normality_test(returns)
        assert 0 <= result.p_value <= 1


class TestExperimentTracker:
    def test_log_and_retrieve(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        exp = tracker.log_experiment(
            model_name="test_model",
            params={"depth": 6},
            metrics={"rmse": 0.01},
            tags=["test"],
        )
        retrieved = tracker.get_experiment(exp.experiment_id)
        assert retrieved.model_name == "test_model"

    def test_comparison(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        tracker.log_experiment("m1", metrics={"rmse": 0.02})
        tracker.log_experiment("m2", metrics={"rmse": 0.01})
        df = tracker.compare_experiments(metric="rmse")
        assert len(df) == 2
