"""Statistical features — moments, rolling statistics, and distributional metrics.

Computes higher-order statistics, entropy measures, and
autocorrelation features useful for regime detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class StatisticalFeatureEngine:
    """Compute statistical features from price and return series.

    Focuses on distributional properties that capture market regime
    shifts and tail risk behaviour.

    Args:
        windows: Rolling window sizes.

    Example:
        >>> engine = StatisticalFeatureEngine()
        >>> features = engine.compute_all(ohlcv_df)
    """

    DEFAULT_WINDOWS = [10, 20, 50, 100]

    def __init__(self, windows: Optional[List[int]] = None) -> None:
        self._windows = windows or self.DEFAULT_WINDOWS

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all statistical features.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with statistical features.
        """
        features = pd.DataFrame(index=df.index)

        groups = [
            self._moment_features(df),
            self._autocorrelation_features(df),
            self._distribution_features(df),
            self._entropy_features(df),
            self._regime_features(df),
        ]

        for group in groups:
            features = pd.concat([features, group], axis=1)

        logger.info("Computed %d statistical features", features.shape[1])
        return features

    def _moment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling moments — mean, variance, skewness, kurtosis."""
        returns = df["close"].pct_change()
        features: Dict[str, pd.Series] = {}

        for w in self._windows:
            features[f"return_mean_{w}"] = returns.rolling(w).mean()
            features[f"return_std_{w}"] = returns.rolling(w).std()
            features[f"return_skew_{w}"] = returns.rolling(w).skew()
            features[f"return_kurt_{w}"] = returns.rolling(w).kurt()

            # Coefficient of variation
            mean = returns.rolling(w).mean()
            std = returns.rolling(w).std()
            features[f"return_cv_{w}"] = std / (mean.abs() + 1e-10)

        return pd.DataFrame(features, index=df.index)

    def _autocorrelation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling autocorrelation at various lags."""
        returns = df["close"].pct_change()
        features: Dict[str, pd.Series] = {}

        for lag in [1, 5, 10]:
            features[f"autocorr_lag{lag}_50"] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0.0, raw=False
            )

        # Partial autocorrelation proxy (difference of autocorrelations)
        if "autocorr_lag1_50" in features and "autocorr_lag5_50" in features:
            features["pacf_diff_1_5"] = features["autocorr_lag1_50"] - features["autocorr_lag5_50"]

        return pd.DataFrame(features, index=df.index)

    def _distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distributional metrics — quantiles, tail ratios, normality."""
        returns = df["close"].pct_change()
        features: Dict[str, pd.Series] = {}

        for w in [20, 50]:
            # Quantile spread
            features[f"iqr_{w}"] = returns.rolling(w).quantile(0.75) - returns.rolling(w).quantile(0.25)

            # Tail ratios
            q95 = returns.rolling(w).quantile(0.95).abs()
            q05 = returns.rolling(w).quantile(0.05).abs()
            features[f"tail_ratio_{w}"] = q95 / (q05 + 1e-10)

            # VaR and CVaR (historical)
            features[f"var_5pct_{w}"] = returns.rolling(w).quantile(0.05)
            features[f"cvar_5pct_{w}"] = returns.rolling(w).apply(
                lambda x: x[x <= np.percentile(x, 5)].mean() if len(x) > 0 else 0.0,
                raw=True,
            )

            # Positive/negative return ratio
            features[f"pos_ratio_{w}"] = returns.rolling(w).apply(
                lambda x: (x > 0).sum() / len(x), raw=True
            )

        return pd.DataFrame(features, index=df.index)

    def _entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Information-theoretic features."""
        returns = df["close"].pct_change()
        features: Dict[str, pd.Series] = {}

        for w in [20, 50]:
            features[f"approx_entropy_{w}"] = returns.rolling(w).apply(
                self._approximate_entropy, raw=True
            )

            # Binned Shannon entropy
            features[f"shannon_entropy_{w}"] = returns.rolling(w).apply(
                self._shannon_entropy, raw=True
            )

        return pd.DataFrame(features, index=df.index)

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features useful for regime detection."""
        close = df["close"]
        returns = close.pct_change()
        features: Dict[str, pd.Series] = {}

        # Hurst exponent approximation (R/S method)
        features["hurst_50"] = returns.rolling(50).apply(self._hurst_rs, raw=True)

        # Variance ratio (mean-reversion vs trend indicator)
        for short, long in [(5, 20), (10, 50)]:
            var_short = returns.rolling(short).var()
            var_long = returns.rolling(long).var()
            features[f"variance_ratio_{short}_{long}"] = (var_long / (long / short)) / (var_short + 1e-10)

        # Trend strength (linear regression R²)
        for w in [20, 50]:
            features[f"trend_r2_{w}"] = close.rolling(w).apply(self._linear_r2, raw=True)

        # Volatility of volatility
        vol_20 = returns.rolling(20).std()
        features["vol_of_vol_50"] = vol_20.rolling(50).std()

        return pd.DataFrame(features, index=df.index)

    @staticmethod
    def _approximate_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
        """Simplified approximate entropy."""
        n = len(x)
        if n < m + 1:
            return 0.0

        r = r_mult * np.std(x)
        if r == 0:
            return 0.0

        def phi(m_val: int) -> float:
            templates = np.array([x[i: i + m_val] for i in range(n - m_val + 1)])
            count = 0.0
            for i in range(len(templates)):
                dists = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(dists <= r)
            count /= len(templates)
            return np.log(count + 1e-10)

        return abs(phi(m) - phi(m + 1))

    @staticmethod
    def _shannon_entropy(x: np.ndarray, bins: int = 10) -> float:
        """Shannon entropy of binned distribution."""
        if len(x) < 2:
            return 0.0
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def _hurst_rs(x: np.ndarray) -> float:
        """Hurst exponent via rescaled range."""
        n = len(x)
        if n < 10:
            return 0.5

        mean = np.mean(x)
        deviations = np.cumsum(x - mean)
        r = np.max(deviations) - np.min(deviations)
        s = np.std(x, ddof=1)
        if s == 0:
            return 0.5

        rs = r / s
        return np.log(rs + 1e-10) / np.log(n)

    @staticmethod
    def _linear_r2(x: np.ndarray) -> float:
        """R² of linear regression fit."""
        n = len(x)
        if n < 3:
            return 0.0
        t = np.arange(n)
        slope, intercept, r_value, _, _ = sp_stats.linregress(t, x)
        return r_value ** 2
