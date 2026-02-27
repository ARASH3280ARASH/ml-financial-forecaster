"""Data preprocessing — normalization, missing-value handling, and splitting.

Ensures financial data is ML-ready while preserving temporal ordering
and preventing look-ahead bias during train/test splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class ScalingMethod(str, Enum):
    """Supported feature scaling methods."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""


@dataclass
class SplitResult:
    """Container for temporal train/validation/test splits.

    Attributes:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        split_indices: Dict with split boundary indices.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    split_indices: Dict[str, int]


class Preprocessor:
    """Financial data preprocessor with temporal-aware operations.

    Handles missing values, outliers, normalization, and chronological
    train/validation/test splitting with optional embargo periods.

    Args:
        scaling: Scaling method to apply.
        outlier_std: Standard deviations beyond which values are clipped.
        fill_method: How to fill missing values ("ffill", "interpolate").

    Example:
        >>> pp = Preprocessor(scaling=ScalingMethod.ROBUST)
        >>> X_clean = pp.fit_transform(features)
        >>> X_new = pp.transform(new_features)
    """

    def __init__(
        self,
        scaling: ScalingMethod = ScalingMethod.ROBUST,
        outlier_std: float = 5.0,
        fill_method: str = "ffill",
    ) -> None:
        self._scaling = scaling
        self._outlier_std = outlier_std
        self._fill_method = fill_method
        self._scaler: Optional[object] = None
        self._feature_means: Optional[pd.Series] = None
        self._feature_stds: Optional[pd.Series] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessor has been fitted."""
        return self._is_fitted

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        """Fit scaling parameters from training data.

        Args:
            X: Training feature matrix.

        Returns:
            Self for method chaining.

        Raises:
            PreprocessingError: If X is empty.
        """
        if X.empty:
            raise PreprocessingError("Cannot fit on empty DataFrame")

        self._feature_means = X.mean()
        self._feature_stds = X.std().replace(0, 1.0)

        if self._scaling == ScalingMethod.STANDARD:
            self._scaler = StandardScaler()
            self._scaler.fit(X.values)
        elif self._scaling == ScalingMethod.ROBUST:
            self._scaler = RobustScaler()
            self._scaler.fit(X.values)

        self._is_fitted = True
        logger.info("Preprocessor fitted on %d features × %d samples", X.shape[1], X.shape[0])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to data.

        Args:
            X: Feature matrix to transform.

        Returns:
            Preprocessed DataFrame.

        Raises:
            PreprocessingError: If not fitted.
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor not fitted — call fit() first")

        result = X.copy()

        # Fill missing values
        if self._fill_method == "ffill":
            result = result.ffill().bfill()
        elif self._fill_method == "interpolate":
            result = result.interpolate(method="linear").ffill().bfill()

        result = result.fillna(0.0)

        # Clip outliers
        if self._outlier_std > 0 and self._feature_means is not None:
            lower = self._feature_means - self._outlier_std * self._feature_stds
            upper = self._feature_means + self._outlier_std * self._feature_stds
            result = result.clip(lower=lower, upper=upper, axis=1)

        # Replace infinities
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0.0, inplace=True)

        # Scale
        if self._scaler is not None:
            scaled = self._scaler.transform(result.values)
            result = pd.DataFrame(scaled, index=result.index, columns=result.columns)

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature matrix.

        Returns:
            Preprocessed DataFrame.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def temporal_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        embargo_bars: int = 10,
    ) -> SplitResult:
        """Split data chronologically with embargo gaps.

        An embargo period between splits prevents information leakage
        from auto-correlated financial time series.

        Args:
            X: Feature matrix.
            y: Target series (aligned with X).
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            embargo_bars: Gap between splits to prevent leakage.

        Returns:
            SplitResult with train/val/test partitions.

        Raises:
            PreprocessingError: If ratios are invalid.
        """
        if train_ratio + val_ratio >= 1.0:
            raise PreprocessingError("train_ratio + val_ratio must be < 1.0")

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        val_start = train_end + embargo_bars
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        test_start = val_end + embargo_bars
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]

        logger.info(
            "Split: train=%d, val=%d, test=%d (embargo=%d)",
            len(X_train), len(X_val), len(X_test), embargo_bars,
        )

        return SplitResult(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            split_indices={"train_end": train_end, "val_end": val_end, "test_start": test_start},
        )
