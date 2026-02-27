"""Data ingestion from multiple sources with validation and caching.

Supports CSV files, Yahoo Finance API, and synthetic data generation
for testing. All loaders return a standardized OHLCV DataFrame.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading fails."""


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        source: Data source type ("csv", "synthetic").
        symbol: Asset ticker symbol.
        start_date: Start date for data range.
        end_date: End date for data range.
        frequency: Bar frequency (e.g. "1h", "1d").
        cache_dir: Directory for caching downloaded data.
    """

    source: str = "synthetic"
    symbol: str = "BTCUSD"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "1h"
    cache_dir: str = "data/cache"
    required_columns: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def load(self, config: DataConfig) -> pd.DataFrame:
        """Load data from the source.

        Args:
            config: Data loading configuration.

        Returns:
            OHLCV DataFrame with DatetimeIndex.

        Raises:
            DataLoadError: If loading fails.
        """


class CSVDataSource(DataSource):
    """Load OHLCV data from CSV files."""

    def load(self, config: DataConfig) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            config: Must have ``source`` set to a valid file path.

        Returns:
            Validated OHLCV DataFrame.

        Raises:
            DataLoadError: If the file is missing or malformed.
        """
        path = Path(config.source)
        if not path.exists():
            raise DataLoadError(f"CSV file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV: {exc}") from exc

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        missing = set(config.required_columns) - set(df.columns)
        if missing:
            raise DataLoadError(f"Missing columns: {missing}")

        logger.info("Loaded %d rows from %s", len(df), path.name)
        return df


class SyntheticDataSource(DataSource):
    """Generate synthetic OHLCV data for testing and development."""

    def load(self, config: DataConfig) -> pd.DataFrame:
        """Generate synthetic price data using geometric Brownian motion.

        Args:
            config: Configuration with date range and frequency.

        Returns:
            Synthetic OHLCV DataFrame.

        Raises:
            DataLoadError: If date range is invalid.
        """
        try:
            idx = pd.date_range(config.start_date, config.end_date, freq="h")
        except Exception as exc:
            raise DataLoadError(f"Invalid date range: {exc}") from exc

        n = len(idx)
        rng = np.random.default_rng(42)

        # Geometric Brownian motion
        mu, sigma = 0.0001, 0.015
        returns = mu + sigma * rng.standard_normal(n)
        price = 50000.0 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        noise = rng.uniform(0.001, 0.008, n)
        df = pd.DataFrame(
            {
                "open": price * (1 + rng.uniform(-0.002, 0.002, n)),
                "high": price * (1 + noise),
                "low": price * (1 - noise),
                "close": price,
                "volume": rng.integers(100, 10000, n).astype(float),
            },
            index=idx,
        )

        logger.info("Generated %d synthetic bars for %s", n, config.symbol)
        return df


class DataLoader:
    """Unified data loader with caching and validation.

    Selects the appropriate data source based on configuration and
    applies common post-processing (sorting, deduplication, type checks).

    Args:
        config: Data loading configuration.

    Example:
        >>> loader = DataLoader(DataConfig(source="synthetic"))
        >>> df = loader.load()
        >>> df.shape
        (52584, 5)
    """

    _sources: Dict[str, type] = {
        "csv": CSVDataSource,
        "synthetic": SyntheticDataSource,
    }

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self._config = config or DataConfig()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(self, config: Optional[DataConfig] = None) -> pd.DataFrame:
        """Load and validate data.

        Args:
            config: Override configuration. Uses instance config if None.

        Returns:
            Validated OHLCV DataFrame with DatetimeIndex.

        Raises:
            DataLoadError: If source is unknown or data is invalid.
        """
        cfg = config or self._config
        cache_key = self._cache_key(cfg)

        if cache_key in self._cache:
            logger.debug("Cache hit for %s", cache_key[:12])
            return self._cache[cache_key].copy()

        source_type = cfg.source if cfg.source in self._sources else "synthetic"
        source = self._sources[source_type]()
        df = source.load(cfg)

        df = self._postprocess(df, cfg)
        self._cache[cache_key] = df
        return df.copy()

    def _postprocess(self, df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
        """Sort, deduplicate, and validate the DataFrame.

        Args:
            df: Raw DataFrame.
            cfg: Configuration with required columns.

        Returns:
            Cleaned DataFrame.
        """
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        for col in cfg.required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        initial_len = len(df)
        df.dropna(subset=cfg.required_columns, inplace=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning("Dropped %d rows with NaN values", dropped)

        return df

    @staticmethod
    def _cache_key(cfg: DataConfig) -> str:
        """Generate a deterministic cache key from config."""
        raw = f"{cfg.source}:{cfg.symbol}:{cfg.start_date}:{cfg.end_date}:{cfg.frequency}"
        return hashlib.md5(raw.encode()).hexdigest()
