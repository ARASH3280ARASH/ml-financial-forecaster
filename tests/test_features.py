"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataConfig, DataLoader
from src.features.technical_features import TechnicalFeatureEngine
from src.features.statistical_features import StatisticalFeatureEngine
from src.features.sentiment_features import SentimentFeatureEngine, SentimentScorer
from src.features.feature_selector import FeatureSelector
from src.data.feature_store import FeatureStore


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    loader = DataLoader(DataConfig(source="synthetic", start_date="2023-01-01", end_date="2023-06-30"))
    return loader.load()


class TestTechnicalFeatures:
    def test_compute_all_returns_dataframe(self, sample_ohlcv):
        engine = TechnicalFeatureEngine()
        features = engine.compute_all(sample_ohlcv)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)

    def test_feature_count_exceeds_50(self, sample_ohlcv):
        engine = TechnicalFeatureEngine()
        features = engine.compute_all(sample_ohlcv)
        assert features.shape[1] >= 50

    def test_rsi_bounded(self, sample_ohlcv):
        engine = TechnicalFeatureEngine()
        features = engine.compute_all(sample_ohlcv)
        rsi = features["rsi_14"].dropna()
        assert rsi.min() >= -1  # Approximately 0-100
        assert rsi.max() <= 101

    def test_custom_periods(self, sample_ohlcv):
        engine = TechnicalFeatureEngine(periods=[5, 10])
        features = engine.compute_all(sample_ohlcv)
        assert "sma_5" in features.columns
        assert "sma_10" in features.columns


class TestStatisticalFeatures:
    def test_compute_all_returns_dataframe(self, sample_ohlcv):
        engine = StatisticalFeatureEngine()
        features = engine.compute_all(sample_ohlcv)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)

    def test_moment_features_present(self, sample_ohlcv):
        engine = StatisticalFeatureEngine(windows=[10, 20])
        features = engine.compute_all(sample_ohlcv)
        assert "return_mean_10" in features.columns
        assert "return_skew_20" in features.columns

    def test_hurst_exponent_range(self, sample_ohlcv):
        engine = StatisticalFeatureEngine()
        features = engine.compute_all(sample_ohlcv)
        hurst = features["hurst_50"].dropna()
        assert hurst.min() >= -1
        assert hurst.max() <= 2


class TestSentimentFeatures:
    def test_scorer_positive(self):
        scorer = SentimentScorer()
        score = scorer.score("bullish rally strong momentum")
        assert score > 0

    def test_scorer_negative(self):
        scorer = SentimentScorer()
        score = scorer.score("bearish crash selloff decline")
        assert score < 0

    def test_scorer_neutral(self):
        scorer = SentimentScorer()
        score = scorer.score("the weather is nice today")
        assert score == 0.0

    def test_scorer_negation(self):
        scorer = SentimentScorer()
        positive = scorer.score("bullish rally")
        negated = scorer.score("not bullish not rally")
        assert negated < positive

    def test_simulated_features(self, sample_ohlcv):
        engine = SentimentFeatureEngine()
        features = engine.generate_simulated(sample_ohlcv)
        assert isinstance(features, pd.DataFrame)
        assert "sentiment_raw" in features.columns


class TestFeatureSelector:
    def test_select_reduces_features(self, sample_ohlcv):
        tech = TechnicalFeatureEngine(periods=[7, 14]).compute_all(sample_ohlcv)
        X = tech.dropna().fillna(0).replace([np.inf, -np.inf], 0)
        y = sample_ohlcv["close"].pct_change().shift(-1).loc[X.index].fillna(0)

        selector = FeatureSelector(max_features=10, correlation_threshold=0.9)
        report = selector.select(X, y)
        assert report.n_selected <= 10
        assert report.n_selected > 0

    def test_transform_applies_selection(self, sample_ohlcv):
        tech = TechnicalFeatureEngine(periods=[7, 14]).compute_all(sample_ohlcv)
        X = tech.dropna().fillna(0).replace([np.inf, -np.inf], 0)
        y = sample_ohlcv["close"].pct_change().shift(-1).loc[X.index].fillna(0)

        selector = FeatureSelector(max_features=10)
        selector.select(X, y)
        X_sel = selector.transform(X)
        assert X_sel.shape[1] <= 10


class TestFeatureStore:
    def test_register_and_retrieve(self):
        store = FeatureStore()
        data = pd.Series(np.random.randn(100), name="test_feat")
        store.register("test_feat", group="test", data=data)
        retrieved = store.get("test_feat")
        assert len(retrieved) == 100

    def test_feature_matrix(self):
        store = FeatureStore()
        for name in ["feat_a", "feat_b", "feat_c"]:
            store.register(name, group="test", data=pd.Series(np.random.randn(50)))
        matrix = store.get_feature_matrix()
        assert matrix.shape == (50, 3)

    def test_list_groups(self):
        store = FeatureStore()
        store.register("f1", group="tech", data=pd.Series([1.0, 2.0]))
        store.register("f2", group="stat", data=pd.Series([1.0, 2.0]))
        groups = store.list_groups()
        assert groups["tech"] == 1
        assert groups["stat"] == 1

    def test_empty_name_raises(self):
        store = FeatureStore()
        with pytest.raises(ValueError):
            store.register("", group="test", data=pd.Series([1.0]))

    def test_missing_feature_raises(self):
        store = FeatureStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")
