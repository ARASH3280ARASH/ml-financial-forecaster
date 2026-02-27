"""Automated feature selection — mutual information, RFE, and correlation filtering.

Reduces dimensionality while retaining the most predictive features,
preventing overfitting on noisy financial data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class SelectionReport:
    """Results of feature selection.

    Attributes:
        selected_features: Features that passed selection.
        removed_features: Features that were removed.
        scores: Importance scores per feature.
        method: Selection method used.
    """

    selected_features: List[str]
    removed_features: List[str]
    scores: Dict[str, float]
    method: str

    @property
    def n_selected(self) -> int:
        return len(self.selected_features)

    @property
    def n_removed(self) -> int:
        return len(self.removed_features)

    def summary(self) -> str:
        return (
            f"FeatureSelection({self.method}): "
            f"kept {self.n_selected}, removed {self.n_removed}"
        )


class FeatureSelector:
    """Multi-method feature selection for financial ML.

    Combines correlation filtering, mutual information ranking,
    and recursive feature elimination to identify the most
    informative features.

    Args:
        correlation_threshold: Max allowed pairwise correlation.
        min_importance: Minimum mutual information score.
        max_features: Maximum features to retain.
        task: 'classification' or 'regression'.

    Example:
        >>> selector = FeatureSelector(max_features=50)
        >>> report = selector.select(X_train, y_train)
        >>> X_selected = selector.transform(X_train)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        min_importance: float = 0.01,
        max_features: int = 100,
        task: str = "regression",
    ) -> None:
        self._corr_threshold = correlation_threshold
        self._min_importance = min_importance
        self._max_features = max_features
        self._task = task
        self._selected: Optional[List[str]] = None
        self._report: Optional[SelectionReport] = None

    @property
    def selected_features(self) -> List[str]:
        if self._selected is None:
            raise RuntimeError("Selector not fitted — call select() first")
        return self._selected

    def select(self, X: pd.DataFrame, y: pd.Series) -> SelectionReport:
        """Run the full feature selection pipeline.

        Steps:
            1. Remove zero-variance features.
            2. Remove highly correlated features.
            3. Rank by mutual information.
            4. (Optional) Recursive Feature Elimination.

        Args:
            X: Feature matrix.
            y: Target variable.

        Returns:
            SelectionReport with results.
        """
        all_features = list(X.columns)
        logger.info("Starting feature selection on %d features", len(all_features))

        # Step 1: Remove zero-variance
        variances = X.var()
        zero_var = variances[variances < 1e-10].index.tolist()
        X_filtered = X.drop(columns=zero_var)
        if zero_var:
            logger.info("Removed %d zero-variance features", len(zero_var))

        # Step 2: Correlation filtering
        X_filtered, corr_removed = self._remove_correlated(X_filtered)
        logger.info("Removed %d correlated features", len(corr_removed))

        # Step 3: Mutual information ranking
        mi_scores = self._mutual_information(X_filtered, y)
        mi_selected = [f for f, s in mi_scores.items() if s >= self._min_importance]

        if len(mi_selected) > self._max_features:
            sorted_features = sorted(mi_scores, key=mi_scores.get, reverse=True)
            mi_selected = sorted_features[: self._max_features]

        X_filtered = X_filtered[mi_selected]

        # Step 4: RFE if still too many features
        if len(mi_selected) > self._max_features:
            rfe_selected = self._recursive_elimination(X_filtered, y, self._max_features)
            X_filtered = X_filtered[rfe_selected]
            mi_selected = rfe_selected

        self._selected = mi_selected
        removed = [f for f in all_features if f not in mi_selected]

        self._report = SelectionReport(
            selected_features=mi_selected,
            removed_features=removed,
            scores=mi_scores,
            method="correlation+mi+rfe",
        )

        logger.info(self._report.summary())
        return self._report

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply selection to a DataFrame.

        Args:
            X: Feature matrix.

        Returns:
            Filtered DataFrame.
        """
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]

    def _remove_correlated(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with pairwise correlation above threshold."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        to_drop = [
            col for col in upper.columns if any(upper[col] > self._corr_threshold)
        ]
        return X.drop(columns=to_drop), to_drop

    def _mutual_information(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Compute mutual information scores."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(0)

        if self._task == "classification":
            scores = mutual_info_classif(X_clean, y_clean, random_state=42)
        else:
            scores = mutual_info_regression(X_clean, y_clean, random_state=42)

        return dict(zip(X.columns, scores))

    def _recursive_elimination(
        self, X: pd.DataFrame, y: pd.Series, n_features: int
    ) -> List[str]:
        """Recursive Feature Elimination with a tree-based estimator."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(0)

        if self._task == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        rfe = RFE(estimator, n_features_to_select=n_features, step=10)
        rfe.fit(X_clean, y_clean)

        return [col for col, mask in zip(X.columns, rfe.support_) if mask]
