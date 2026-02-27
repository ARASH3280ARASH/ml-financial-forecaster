"""Cross-validation — walk-forward and purged CV for financial time series.

Prevents information leakage by respecting temporal ordering
and adding embargo gaps between train/test folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold.

    Attributes:
        fold: Fold index.
        train_size: Number of training samples.
        test_size: Number of test samples.
        metrics: Model performance on this fold.
    """

    fold: int
    train_size: int
    test_size: int
    metrics: ModelMetrics


@dataclass
class CVResult:
    """Aggregated cross-validation results.

    Attributes:
        fold_results: Per-fold results.
        mean_rmse: Average RMSE across folds.
        std_rmse: RMSE standard deviation.
        mean_r2: Average R-squared.
        mean_directional_accuracy: Average directional accuracy.
    """

    fold_results: List[CVFoldResult] = field(default_factory=list)
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    mean_r2: float = 0.0
    mean_directional_accuracy: float = 0.0

    def summary(self) -> str:
        return (
            f"CV({len(self.fold_results)} folds): "
            f"RMSE={self.mean_rmse:.6f}±{self.std_rmse:.6f} | "
            f"R²={self.mean_r2:.4f} | "
            f"DirAcc={self.mean_directional_accuracy:.2%}"
        )


class WalkForwardValidator:
    """Walk-forward cross-validation for time series.

    Expands the training window forward through time, testing
    on the next unseen segment. Respects temporal ordering.

    Args:
        n_splits: Number of expanding folds.
        min_train_size: Minimum training samples.
        embargo_bars: Gap between train and test.
        expanding: If True, training window expands; if False, slides.

    Example:
        >>> wf = WalkForwardValidator(n_splits=5)
        >>> result = wf.validate(model, X, y)
        >>> print(result.summary())
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 252,
        embargo_bars: int = 10,
        expanding: bool = True,
    ) -> None:
        self._n_splits = n_splits
        self._min_train_size = min_train_size
        self._embargo = embargo_bars
        self._expanding = expanding

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for each fold.

        Args:
            X: Feature matrix.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n = len(X)
        test_size = (n - self._min_train_size) // self._n_splits

        for fold in range(self._n_splits):
            if self._expanding:
                train_start = 0
            else:
                train_start = max(0, self._min_train_size + fold * test_size - self._min_train_size)

            train_end = self._min_train_size + fold * test_size
            test_start = train_end + self._embargo
            test_end = min(test_start + test_size, n)

            if test_start >= n or test_end <= test_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CVResult:
        """Run walk-forward cross-validation.

        Args:
            model: Model to validate (will be re-fitted each fold).
            X: Feature matrix.
            y: Target series.

        Returns:
            CVResult with aggregated metrics.
        """
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Clean data
            X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
            X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)

            model.fit(X_train_clean, y_train)
            metrics = model.evaluate(X_test_clean, y_test)

            fold_result = CVFoldResult(
                fold=fold,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
            )
            fold_results.append(fold_result)

            logger.info(
                "Fold %d: train=%d, test=%d, RMSE=%.6f",
                fold, len(train_idx), len(test_idx), metrics.rmse,
            )

        result = self._aggregate(fold_results)
        logger.info("Walk-forward CV: %s", result.summary())
        return result

    def _aggregate(self, fold_results: List[CVFoldResult]) -> CVResult:
        """Aggregate metrics across folds."""
        if not fold_results:
            return CVResult()

        rmses = [f.metrics.rmse for f in fold_results]
        r2s = [f.metrics.r2 for f in fold_results]
        dir_accs = [f.metrics.directional_accuracy for f in fold_results]

        return CVResult(
            fold_results=fold_results,
            mean_rmse=float(np.mean(rmses)),
            std_rmse=float(np.std(rmses)),
            mean_r2=float(np.mean(r2s)),
            mean_directional_accuracy=float(np.mean(dir_accs)),
        )


class PurgedKFold:
    """Purged K-Fold cross-validation for financial data.

    Removes samples from the training set that are within an
    embargo window of the test set, preventing information leakage
    from autocorrelated features.

    Args:
        n_splits: Number of folds.
        embargo_pct: Fraction of data to purge around test boundaries.

    Example:
        >>> pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
        >>> result = pkf.validate(model, X, y)
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self._n_splits = n_splits
        self._embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits.

        Args:
            X: Feature matrix.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        n = len(X)
        embargo = int(n * self._embargo_pct)
        fold_size = n // self._n_splits

        for fold in range(self._n_splits):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)

            # Purge: remove embargo samples around test boundaries
            purge_start = max(0, test_start - embargo)
            purge_end = min(n, test_end + embargo)

            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n),
            ])
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CVResult:
        """Run purged cross-validation."""
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            X_train = X.iloc[train_idx].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = X.iloc[test_idx].fillna(0).replace([np.inf, -np.inf], 0)

            model.fit(X_train, y.iloc[train_idx])
            metrics = model.evaluate(X_test, y.iloc[test_idx])

            fold_results.append(CVFoldResult(
                fold=fold,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
            ))

        rmses = [f.metrics.rmse for f in fold_results]
        r2s = [f.metrics.r2 for f in fold_results]
        dir_accs = [f.metrics.directional_accuracy for f in fold_results]

        return CVResult(
            fold_results=fold_results,
            mean_rmse=float(np.mean(rmses)),
            std_rmse=float(np.std(rmses)),
            mean_r2=float(np.mean(r2s)),
            mean_directional_accuracy=float(np.mean(dir_accs)),
        )
