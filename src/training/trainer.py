"""Training orchestrator — end-to-end model training pipeline.

Coordinates data loading, preprocessing, feature engineering,
model training, and evaluation into a single reproducible workflow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import Preprocessor, SplitResult
from src.models.base_model import BaseModel, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline.

    Attributes:
        train_ratio: Fraction of data for training.
        val_ratio: Fraction for validation.
        embargo_bars: Gap between splits.
        target_column: Column to predict.
        target_horizon: Forward-looking prediction horizon.
        early_stopping_patience: Epochs to wait before stopping.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    embargo_bars: int = 10
    target_column: str = "close"
    target_horizon: int = 1
    early_stopping_patience: int = 10


@dataclass
class TrainingResult:
    """Results from a training run.

    Attributes:
        model: Trained model instance.
        train_metrics: Performance on training set.
        val_metrics: Performance on validation set.
        test_metrics: Performance on test set.
        training_time: Total wall-clock time.
        feature_importance: Top feature importances.
    """

    model: BaseModel
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    test_metrics: ModelMetrics
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        lines = [
            f"Model: {self.model.name}",
            f"  Train: {self.train_metrics.summary()}",
            f"  Val:   {self.val_metrics.summary()}",
            f"  Test:  {self.test_metrics.summary()}",
            f"  Time:  {self.training_time:.1f}s",
        ]
        return "\n".join(lines)


class Trainer:
    """End-to-end training orchestrator.

    Manages the complete workflow from raw features to evaluated model,
    including preprocessing, splitting, training, and evaluation.

    Args:
        config: Training configuration.
        preprocessor: Data preprocessor.

    Example:
        >>> trainer = Trainer()
        >>> result = trainer.train(model, features, target)
        >>> print(result.test_metrics.summary())
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        self._config = config or TrainingConfig()
        self._preprocessor = preprocessor or Preprocessor()
        self._history: List[TrainingResult] = []

    @property
    def history(self) -> List[TrainingResult]:
        return self._history

    def train(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TrainingResult:
        """Execute the full training pipeline.

        Args:
            model: Model instance to train.
            X: Feature matrix.
            y: Target series.

        Returns:
            TrainingResult with metrics and trained model.
        """
        start = time.time()

        # Create target with forward horizon
        y_shifted = self._create_target(y)

        # Align X and y
        valid_idx = y_shifted.dropna().index.intersection(X.index)
        X_aligned = X.loc[valid_idx]
        y_aligned = y_shifted.loc[valid_idx]

        # Temporal split
        split = Preprocessor.temporal_split(
            X_aligned, y_aligned,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            embargo_bars=self._config.embargo_bars,
        )

        # Fit preprocessor on training data only
        self._preprocessor.fit(split.X_train)

        # Transform all splits
        X_train = self._preprocessor.transform(split.X_train)
        X_val = self._preprocessor.transform(split.X_val)
        X_test = self._preprocessor.transform(split.X_test)

        # Train
        model.fit(X_train, split.y_train)

        # Evaluate on all splits
        train_metrics = model.evaluate(X_train, split.y_train)
        val_metrics = model.evaluate(X_val, split.y_val)
        test_metrics = model.evaluate(X_test, split.y_test)

        elapsed = time.time() - start

        result = TrainingResult(
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            training_time=elapsed,
            feature_importance=model.get_feature_importance(),
        )

        self._history.append(result)
        logger.info("Training complete:\n%s", result.summary())
        return result

    def train_multiple(
        self,
        models: List[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[TrainingResult]:
        """Train and compare multiple models.

        Args:
            models: List of model instances.
            X: Feature matrix.
            y: Target series.

        Returns:
            List of TrainingResults, sorted by test RMSE.
        """
        results = []
        for model in models:
            logger.info("Training model: %s", model.name)
            result = self.train(model, X, y)
            results.append(result)

        results.sort(key=lambda r: r.test_metrics.rmse)

        logger.info("Model comparison (by test RMSE):")
        for i, r in enumerate(results):
            logger.info("  %d. %s — RMSE=%.6f", i + 1, r.model.name, r.test_metrics.rmse)

        return results

    def _create_target(self, y: pd.Series) -> pd.Series:
        """Create forward-shifted target for prediction.

        Args:
            y: Raw target series.

        Returns:
            Shifted target (predicting horizon steps ahead).
        """
        horizon = self._config.target_horizon
        if horizon == 0:
            return y

        # Predict future returns
        return y.pct_change(horizon).shift(-horizon)

    def get_best_model(self) -> Optional[TrainingResult]:
        """Get the best model from training history.

        Returns:
            TrainingResult with lowest test RMSE, or None.
        """
        if not self._history:
            return None
        return min(self._history, key=lambda r: r.test_metrics.rmse)

    def comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all trained models.

        Returns:
            DataFrame with model names and metrics.
        """
        rows = []
        for result in self._history:
            rows.append({
                "model": result.model.name,
                "train_rmse": result.train_metrics.rmse,
                "val_rmse": result.val_metrics.rmse,
                "test_rmse": result.test_metrics.rmse,
                "test_mae": result.test_metrics.mae,
                "test_r2": result.test_metrics.r2,
                "test_dir_acc": result.test_metrics.directional_accuracy,
                "time_s": result.training_time,
            })
        return pd.DataFrame(rows).sort_values("test_rmse")
