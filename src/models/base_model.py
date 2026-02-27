"""Abstract base model — defines the interface for all forecasting models.

Every model in the registry must implement this contract to ensure
interchangeability in ensembles and cross-validation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics.

    Attributes:
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        r2: R-squared score.
        directional_accuracy: Fraction of correct direction predictions.
        training_time: Seconds spent training.
    """

    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0
    training_time: float = 0.0

    def summary(self) -> str:
        return (
            f"RMSE={self.rmse:.6f} | MAE={self.mae:.6f} | "
            f"R²={self.r2:.4f} | DirAcc={self.directional_accuracy:.2%}"
        )


class BaseModel(ABC):
    """Abstract base class for all forecasting models.

    Provides a consistent interface for training, prediction,
    and evaluation. Subclasses must implement ``_fit`` and ``_predict``.

    Args:
        name: Human-readable model name.
        params: Model-specific hyperparameters.

    Example:
        >>> class MyModel(BaseModel):
        ...     def _fit(self, X, y): ...
        ...     def _predict(self, X): ...
        >>> model = MyModel("my_model")
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params = params or {}
        self._is_fitted = False
        self._metrics: Optional[ModelMetrics] = None
        self._feature_names: Optional[list] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def metrics(self) -> Optional[ModelMetrics]:
        return self._metrics

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Train the model.

        Args:
            X: Training features.
            y: Training target.

        Returns:
            Self for method chaining.
        """
        logger.info("Training %s on %d samples × %d features", self.name, *X.shape)
        self._feature_names = list(X.columns)

        start = time.time()
        self._fit(X, y)
        elapsed = time.time() - start

        self._is_fitted = True
        logger.info("%s trained in %.2fs", self.name, elapsed)

        if self._metrics:
            self._metrics.training_time = elapsed

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} is not fitted — call fit() first")
        return self._predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model on held-out data.

        Args:
            X: Test features.
            y: True values.

        Returns:
            ModelMetrics with evaluation scores.
        """
        preds = self.predict(X)
        y_arr = y.values

        mse = float(np.mean((y_arr - preds) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_arr - preds)))

        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        # Directional accuracy
        if len(y_arr) > 1:
            actual_dir = np.sign(np.diff(y_arr))
            pred_dir = np.sign(np.diff(preds))
            dir_acc = float(np.mean(actual_dir == pred_dir))
        else:
            dir_acc = 0.0

        self._metrics = ModelMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2,
            directional_accuracy=dir_acc,
        )

        logger.info("%s evaluation: %s", self.name, self._metrics.summary())
        return self._metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance scores if supported.

        Returns:
            Dict mapping feature names to importance scores, or None.
        """
        return None

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Model-specific training logic."""

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Model-specific prediction logic."""

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"
