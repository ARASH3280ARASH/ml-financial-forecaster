"""Ensemble model — voting and stacking strategies for combining forecasters.

Combines multiple base models to improve robustness and reduce
variance in financial predictions.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel, ModelMetrics

logger = logging.getLogger(__name__)


class EnsembleMethod(str, Enum):
    """Supported ensemble combination strategies."""

    MEAN = "mean"
    WEIGHTED = "weighted"
    MEDIAN = "median"
    STACKING = "stacking"


class EnsembleModel(BaseModel):
    """Ensemble forecaster combining multiple base models.

    Supports simple averaging, weighted combination, median,
    and meta-learner stacking.

    Args:
        name: Ensemble identifier.
        models: List of base models to combine.
        method: Combination strategy.
        weights: Model weights (for WEIGHTED method).

    Example:
        >>> xgb = GradientBoostingModel("xgb", backend="xgboost")
        >>> lgb = GradientBoostingModel("lgb", backend="lightgbm")
        >>> ensemble = EnsembleModel("ens", models=[xgb, lgb])
        >>> ensemble.fit(X_train, y_train)
    """

    def __init__(
        self,
        name: str = "ensemble",
        models: Optional[List[BaseModel]] = None,
        method: EnsembleMethod = EnsembleMethod.MEAN,
        weights: Optional[List[float]] = None,
    ) -> None:
        self._models = models or []
        self._method = method
        self._weights = weights
        self._meta_model: Any = None
        super().__init__(name)

    @property
    def n_models(self) -> int:
        return len(self._models)

    def add_model(self, model: BaseModel) -> None:
        """Add a model to the ensemble."""
        self._models.append(model)
        logger.debug("Added %s to ensemble (total: %d)", model.name, self.n_models)

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all base models (and meta-learner if stacking)."""
        if not self._models:
            raise ValueError("Ensemble has no models — add models first")

        for model in self._models:
            logger.info("Training ensemble member: %s", model.name)
            model.fit(X, y)

        if self._method == EnsembleMethod.STACKING:
            self._fit_stacking(X, y)

        if self._method == EnsembleMethod.WEIGHTED and self._weights is None:
            self._weights = self._optimize_weights(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = np.column_stack([m.predict(X) for m in self._models])

        if self._method == EnsembleMethod.MEAN:
            return np.mean(predictions, axis=1)

        elif self._method == EnsembleMethod.MEDIAN:
            return np.median(predictions, axis=1)

        elif self._method == EnsembleMethod.WEIGHTED:
            weights = np.array(self._weights or [1.0 / self.n_models] * self.n_models)
            return predictions @ weights

        elif self._method == EnsembleMethod.STACKING:
            if self._meta_model is None:
                return np.mean(predictions, axis=1)
            return self._meta_model.predict(predictions)

        return np.mean(predictions, axis=1)

    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train a meta-learner on base model predictions."""
        from sklearn.linear_model import Ridge

        # Generate out-of-fold predictions for meta-learner training
        n = len(X)
        split = int(n * 0.7)

        # Use temporal split (no shuffling)
        X_base, X_meta = X.iloc[:split], X.iloc[split:]
        y_base, y_meta = y.iloc[:split], y.iloc[split:]

        # Retrain base models on first portion
        for model in self._models:
            model.fit(X_base, y_base)

        # Get base predictions on second portion
        meta_features = np.column_stack([m.predict(X_meta) for m in self._models])

        # Train meta-learner
        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(meta_features, y_meta)

        # Retrain base models on full data
        for model in self._models:
            model.fit(X, y)

        logger.info("Stacking meta-learner fitted with %d base models", self.n_models)

    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Find optimal ensemble weights via validation performance."""
        n = len(X)
        val_start = int(n * 0.8)

        X_tr, X_val = X.iloc[:val_start], X.iloc[val_start:]
        y_val = y.iloc[val_start:]

        # Re-fit on training portion
        for model in self._models:
            model.fit(X_tr, y.iloc[:val_start])

        # Score each model on validation
        val_predictions = np.column_stack([m.predict(X_val) for m in self._models])
        errors = np.array([
            np.mean((y_val.values - val_predictions[:, i]) ** 2)
            for i in range(self.n_models)
        ])

        # Inverse-error weighting
        inv_errors = 1.0 / (errors + 1e-10)
        weights = inv_errors / inv_errors.sum()

        # Re-fit on full data
        for model in self._models:
            model.fit(X, y)

        logger.info("Optimized weights: %s", dict(zip([m.name for m in self._models], weights)))
        return weights.tolist()

    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get individual model predictions for analysis.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with one column per model.
        """
        return pd.DataFrame(
            {m.name: m.predict(X) for m in self._models},
            index=X.index,
        )

    def evaluate_members(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelMetrics]:
        """Evaluate each member model individually.

        Args:
            X: Test features.
            y: True values.

        Returns:
            Dict mapping model names to their metrics.
        """
        results = {}
        for model in self._models:
            results[model.name] = model.evaluate(X, y)
        return results
