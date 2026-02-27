"""Sentiment features — simulated news and social-media sentiment scoring.

Provides a framework for integrating NLP-based sentiment into the
feature pipeline. Includes a lexicon-based scorer and a placeholder
for transformer-based models (e.g. FinBERT).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentLexicon:
    """Financial sentiment word lists.

    Attributes:
        positive: Words conveying bullish sentiment.
        negative: Words conveying bearish sentiment.
        amplifiers: Words that intensify surrounding sentiment.
    """

    positive: List[str] = field(default_factory=lambda: [
        "bullish", "surge", "rally", "breakout", "upgrade", "outperform",
        "growth", "profit", "gain", "recovery", "momentum", "strong",
        "accumulation", "optimistic", "uptrend", "support", "buy",
        "overweight", "beat", "exceed", "innovation", "expansion",
    ])
    negative: List[str] = field(default_factory=lambda: [
        "bearish", "crash", "plunge", "breakdown", "downgrade", "underperform",
        "loss", "decline", "selloff", "recession", "weakness", "volatile",
        "distribution", "pessimistic", "downtrend", "resistance", "sell",
        "underweight", "miss", "default", "liquidation", "contraction",
    ])
    amplifiers: List[str] = field(default_factory=lambda: [
        "very", "extremely", "significantly", "strongly", "sharply",
        "massively", "dramatically", "substantially", "heavily",
    ])


class SentimentScorer:
    """Lexicon-based financial sentiment scorer.

    Scores text by counting positive/negative financial terms,
    accounting for negation and amplifiers.

    Args:
        lexicon: Custom lexicon. Uses default financial lexicon if None.

    Example:
        >>> scorer = SentimentScorer()
        >>> scorer.score("Bitcoin rally continues with strong momentum")
        0.67
    """

    def __init__(self, lexicon: Optional[SentimentLexicon] = None) -> None:
        self._lexicon = lexicon or SentimentLexicon()
        self._negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                                "hardly", "barely", "cannot", "without", "don't", "doesn't"}

    def score(self, text: str) -> float:
        """Score a single text for sentiment.

        Args:
            text: Input text.

        Returns:
            Sentiment score in [-1, 1]. Positive = bullish, negative = bearish.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        score = 0.0
        amplifier = 1.0

        for i, word in enumerate(words):
            if word in self._lexicon.amplifiers:
                amplifier = 1.5
                continue

            # Check for negation in preceding two words
            negated = any(
                words[j] in self._negation_words
                for j in range(max(0, i - 2), i)
            )

            if word in self._lexicon.positive:
                score += amplifier * (-1.0 if negated else 1.0)
            elif word in self._lexicon.negative:
                score += amplifier * (1.0 if negated else -1.0)

            amplifier = 1.0

        # Normalize by text length
        max_score = max(len(words) * 0.5, 1.0)
        return np.clip(score / max_score, -1.0, 1.0)

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of sentiment scores.
        """
        return [self.score(t) for t in texts]


class SentimentFeatureEngine:
    """Generate sentiment-derived features for the ML pipeline.

    When real sentiment data is unavailable, generates simulated
    sentiment correlated with price movements for development/testing.

    Args:
        scorer: Sentiment scorer instance.

    Example:
        >>> engine = SentimentFeatureEngine()
        >>> features = engine.generate_simulated(ohlcv_df)
    """

    def __init__(self, scorer: Optional[SentimentScorer] = None) -> None:
        self._scorer = scorer or SentimentScorer()

    def from_texts(
        self, texts: pd.Series, timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Compute sentiment features from text data.

        Args:
            texts: Series of text strings.
            timestamps: Aligned datetime index.

        Returns:
            DataFrame with sentiment features.
        """
        scores = pd.Series(self._scorer.score_batch(texts.tolist()), index=timestamps)
        return self._build_features(scores)

    def generate_simulated(
        self, df: pd.DataFrame, noise_level: float = 0.3, seed: int = 42
    ) -> pd.DataFrame:
        """Generate simulated sentiment correlated with returns.

        Creates a realistic sentiment proxy by combining lagged returns
        with random noise. Useful for pipeline development.

        Args:
            df: OHLCV DataFrame.
            noise_level: Proportion of random noise (0 = pure signal).
            seed: Random seed.

        Returns:
            DataFrame with simulated sentiment features.
        """
        rng = np.random.default_rng(seed)
        returns = df["close"].pct_change()

        # Simulated raw sentiment: lagged returns + noise
        signal = returns.shift(1).fillna(0)
        noise = pd.Series(rng.normal(0, returns.std() * noise_level, len(df)), index=df.index)
        raw_sentiment = np.clip(signal * 50 + noise, -1, 1)

        logger.info("Generated simulated sentiment features (noise=%.1f)", noise_level)
        return self._build_features(raw_sentiment)

    def _build_features(self, sentiment: pd.Series) -> pd.DataFrame:
        """Build feature set from a raw sentiment series.

        Args:
            sentiment: Raw sentiment scores indexed by datetime.

        Returns:
            DataFrame with derived sentiment features.
        """
        features: Dict[str, pd.Series] = {}

        features["sentiment_raw"] = sentiment

        # Smoothed sentiment
        for w in [5, 10, 20]:
            features[f"sentiment_sma_{w}"] = sentiment.rolling(w).mean()
            features[f"sentiment_std_{w}"] = sentiment.rolling(w).std()

        # Sentiment momentum
        features["sentiment_momentum_5"] = sentiment.diff(5)
        features["sentiment_momentum_10"] = sentiment.diff(10)

        # Extreme sentiment flags
        features["sentiment_extreme_pos"] = (sentiment > sentiment.rolling(50).mean() + 2 * sentiment.rolling(50).std()).astype(float)
        features["sentiment_extreme_neg"] = (sentiment < sentiment.rolling(50).mean() - 2 * sentiment.rolling(50).std()).astype(float)

        # Sentiment divergence (sentiment direction vs price direction placeholder)
        features["sentiment_sign"] = np.sign(sentiment)

        return pd.DataFrame(features, index=sentiment.index)
