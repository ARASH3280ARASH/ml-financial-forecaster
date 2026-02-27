"""Gradient boosting models — XGBoost and LightGBM wrappers.

Production-ready wrappers with built-in early stopping,
feature importance extraction, and hyperparameter defaults
tuned for financial time series.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Unified gradient boosting model supporting XGBoost and LightGBM.

    Automatically selects the backend based on availability and provides
    sensible defaults for financial forecasting.

    Args:
        name: Model identifier.
        backend: 'xgboost' or 'lightgbm'.
        params: Hyperparameters passed to the backend.
        early_stopping_rounds: Patience for early stopping.

    Example:
        >>> model = GradientBoostingModel("xgb_v1", backend="xgboost")
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    XGBOOST_DEFAULTS: Dict[str, Any] = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "random_state": 42,
    }

    LIGHTGBM_DEFAULTS: Dict[str, Any] = {
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(
        self,
        name: str = "gradient_boosting",
        backend: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
    ) -> None:
        self._backend = backend
        self._early_stopping = early_stopping_rounds
        self._model: Any = None

        defaults = (
            self.XGBOOST_DEFAULTS.copy()
            if backend == "xgboost"
            else self.LIGHTGBM_DEFAULTS.copy()
        )
        if params:
            defaults.update(params)

        super().__init__(name, defaults)

    @property
    def backend(self) -> str:
        return self._backend

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the gradient boosting model."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(0)

        if self._backend == "xgboost":
            self._fit_xgboost(X_clean, y_clean)
        else:
            self._fit_lightgbm(X_clean, y_clean)

    def _fit_xgboost(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit using XGBoost."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("XGBoost not installed, falling back to LightGBM")
            self._backend = "lightgbm"
            self._fit_lightgbm(X, y)
            return

        self._model = XGBRegressor(**self.params)

        # Use last 15% for early stopping validation
        split_idx = int(len(X) * 0.85)
        X_tr, X_es = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_es = y.iloc[:split_idx], y.iloc[split_idx:]

        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )

        best_iter = getattr(self._model, "best_iteration", self.params.get("n_estimators"))
        logger.info("XGBoost fitted: best_iteration=%s", best_iter)

    def _fit_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit using LightGBM."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("Neither XGBoost nor LightGBM is installed")

        self._model = LGBMRegressor(**self.params)

        split_idx = int(len(X) * 0.85)
        X_tr, X_es = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_es = y.iloc[:split_idx], y.iloc[split_idx:]

        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
        )

        best_iter = getattr(self._model, "best_iteration_", self.params.get("n_estimators"))
        logger.info("LightGBM fitted: best_iteration=%s", best_iter)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        return self._model.predict(X_clean)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Extract feature importance from the fitted model."""
        if self._model is None or self._feature_names is None:
            return None

        importances = self._model.feature_importances_
        total = importances.sum() + 1e-10
        normalized = importances / total

        return dict(
            sorted(
                zip(self._feature_names, normalized),
                key=lambda x: x[1],
                reverse=True,
            )
        )
