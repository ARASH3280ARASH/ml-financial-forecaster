"""Technical indicator features for financial time series.

Computes 50+ technical indicators including trend, momentum, volatility,
and volume-based features. All indicators are computed without look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatureEngine:
    """Compute technical analysis features from OHLCV data.

    Generates a comprehensive set of indicators used in quantitative
    finance. Every feature is computed causally (using only past data).

    Args:
        periods: Look-back windows for multi-period indicators.

    Example:
        >>> engine = TechnicalFeatureEngine()
        >>> features = engine.compute_all(ohlcv_df)
        >>> features.shape[1]  # 50+ columns
    """

    DEFAULT_PERIODS = [7, 14, 21, 50, 100, 200]

    def __init__(self, periods: Optional[List[int]] = None) -> None:
        self._periods = periods or self.DEFAULT_PERIODS

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].

        Returns:
            DataFrame with all computed features.
        """
        features = pd.DataFrame(index=df.index)

        feature_groups = [
            self._trend_features(df),
            self._momentum_features(df),
            self._volatility_features(df),
            self._volume_features(df),
            self._price_features(df),
            self._pattern_features(df),
        ]

        for group in feature_groups:
            features = pd.concat([features, group], axis=1)

        logger.info("Computed %d technical features", features.shape[1])
        return features

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving averages and trend indicators."""
        close = df["close"]
        features: Dict[str, pd.Series] = {}

        # Simple Moving Averages
        for p in self._periods:
            features[f"sma_{p}"] = close.rolling(p).mean()
            features[f"sma_ratio_{p}"] = close / close.rolling(p).mean()

        # Exponential Moving Averages
        for p in self._periods:
            features[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
            features[f"ema_ratio_{p}"] = close / close.ewm(span=p, adjust=False).mean()

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features["macd"] = ema12 - ema26
        features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # ADX (Average Directional Index)
        features["adx_14"] = self._compute_adx(df, 14)

        # Ichimoku components
        high9 = df["high"].rolling(9).max()
        low9 = df["low"].rolling(9).min()
        features["tenkan_sen"] = (high9 + low9) / 2

        high26 = df["high"].rolling(26).max()
        low26 = df["low"].rolling(26).min()
        features["kijun_sen"] = (high26 + low26) / 2

        features["ichimoku_diff"] = features["tenkan_sen"] - features["kijun_sen"]

        return pd.DataFrame(features, index=df.index)

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and oscillator indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        features: Dict[str, pd.Series] = {}

        # RSI
        for p in [7, 14, 21]:
            features[f"rsi_{p}"] = self._compute_rsi(close, p)

        # Stochastic Oscillator
        for p in [14, 21]:
            lowest = low.rolling(p).min()
            highest = high.rolling(p).max()
            features[f"stoch_k_{p}"] = 100 * (close - lowest) / (highest - lowest + 1e-10)
            features[f"stoch_d_{p}"] = features[f"stoch_k_{p}"].rolling(3).mean()

        # Williams %R
        for p in [14, 21]:
            highest = high.rolling(p).max()
            lowest = low.rolling(p).min()
            features[f"williams_r_{p}"] = -100 * (highest - close) / (highest - lowest + 1e-10)

        # Rate of Change
        for p in [5, 10, 20]:
            features[f"roc_{p}"] = close.pct_change(p) * 100

        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        for p in [14, 20]:
            sma_tp = tp.rolling(p).mean()
            mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            features[f"cci_{p}"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # MFI (Money Flow Index)
        features["mfi_14"] = self._compute_mfi(df, 14)

        return pd.DataFrame(features, index=df.index)

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        features: Dict[str, pd.Series] = {}

        # Bollinger Bands
        for p in [20]:
            sma = close.rolling(p).mean()
            std = close.rolling(p).std()
            features[f"bb_upper_{p}"] = sma + 2 * std
            features[f"bb_lower_{p}"] = sma - 2 * std
            features[f"bb_width_{p}"] = (4 * std) / (sma + 1e-10)
            features[f"bb_position_{p}"] = (close - features[f"bb_lower_{p}"]) / (
                features[f"bb_upper_{p}"] - features[f"bb_lower_{p}"] + 1e-10
            )

        # ATR (Average True Range)
        for p in [7, 14, 21]:
            features[f"atr_{p}"] = self._compute_atr(df, p)
            features[f"atr_ratio_{p}"] = features[f"atr_{p}"] / (close + 1e-10)

        # Historical Volatility
        returns = close.pct_change()
        for p in [10, 20, 60]:
            features[f"volatility_{p}"] = returns.rolling(p).std() * np.sqrt(252)

        # Garman-Klass Volatility
        log_hl = np.log(high / (low + 1e-10)) ** 2
        log_co = np.log(close / (df["open"] + 1e-10)) ** 2
        features["gk_volatility_20"] = (
            0.5 * log_hl.rolling(20).mean() - (2 * np.log(2) - 1) * log_co.rolling(20).mean()
        ).apply(np.sqrt)

        return pd.DataFrame(features, index=df.index)

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators."""
        close = df["close"]
        volume = df["volume"]
        features: Dict[str, pd.Series] = {}

        # Volume Moving Averages
        for p in [10, 20, 50]:
            features[f"volume_sma_{p}"] = volume.rolling(p).mean()
            features[f"volume_ratio_{p}"] = volume / (volume.rolling(p).mean() + 1e-10)

        # OBV (On-Balance Volume)
        direction = np.sign(close.diff())
        features["obv"] = (volume * direction).cumsum()
        features["obv_sma_20"] = features["obv"].rolling(20).mean()

        # VWAP approximation (rolling)
        typical_price = (df["high"] + df["low"] + close) / 3
        for p in [20]:
            cum_vol = volume.rolling(p).sum()
            cum_vp = (typical_price * volume).rolling(p).sum()
            features[f"vwap_{p}"] = cum_vp / (cum_vol + 1e-10)
            features[f"vwap_ratio_{p}"] = close / (features[f"vwap_{p}"] + 1e-10)

        # Accumulation/Distribution
        mfm = ((close - df["low"]) - (df["high"] - close)) / (df["high"] - df["low"] + 1e-10)
        features["ad_line"] = (mfm * volume).cumsum()

        return pd.DataFrame(features, index=df.index)

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw price-derived features."""
        close = df["close"]
        features: Dict[str, pd.Series] = {}

        # Returns at multiple horizons
        for p in [1, 2, 5, 10, 20]:
            features[f"return_{p}d"] = close.pct_change(p)

        # Log returns
        features["log_return_1d"] = np.log(close / close.shift(1))

        # Price distance from recent high/low
        for p in [20, 50]:
            features[f"dist_high_{p}"] = close / (df["high"].rolling(p).max() + 1e-10) - 1
            features[f"dist_low_{p}"] = close / (df["low"].rolling(p).min() + 1e-10) - 1

        # Candle body and shadow ratios
        body = abs(close - df["open"])
        full_range = df["high"] - df["low"] + 1e-10
        features["body_ratio"] = body / full_range
        features["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / full_range
        features["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / full_range

        return pd.DataFrame(features, index=df.index)

    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple candlestick pattern detection."""
        close = df["close"]
        open_ = df["open"]
        features: Dict[str, pd.Series] = {}

        # Consecutive direction
        direction = np.sign(close - open_)
        features["consecutive_green"] = direction.rolling(5).sum()

        # Gap detection
        features["gap_up"] = (open_ > close.shift(1)).astype(float)
        features["gap_down"] = (open_ < close.shift(1)).astype(float)

        # Higher highs / lower lows
        features["higher_high"] = (df["high"] > df["high"].shift(1)).astype(float).rolling(5).sum()
        features["lower_low"] = (df["low"] < df["low"].shift(1)).astype(float).rolling(5).sum()

        return pd.DataFrame(features, index=df.index)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average Directional Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        return dx.rolling(period).mean()

    @staticmethod
    def _compute_mfi(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Money Flow Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]
        direction = tp.diff()

        pos_mf = mf.where(direction > 0, 0.0).rolling(period).sum()
        neg_mf = mf.where(direction <= 0, 0.0).rolling(period).sum()

        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        return mfi
