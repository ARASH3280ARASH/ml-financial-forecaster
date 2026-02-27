"""Model visualization — learning curves, feature importance, prediction analysis.

Generates publication-quality plots for model diagnostics
and presentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPlotter:
    """Generate model diagnostic and analysis plots.

    Creates a suite of visualizations for understanding model
    behaviour, feature importance, and prediction quality.

    Args:
        style: Matplotlib style to apply.
        figsize: Default figure size.
        save_dir: Directory to save generated plots.

    Example:
        >>> plotter = ModelPlotter(save_dir="plots/")
        >>> plotter.feature_importance(importances, top_n=20)
        >>> plotter.learning_curve(train_losses, val_losses)
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: tuple = (12, 6),
        save_dir: Optional[str] = None,
    ) -> None:
        self._figsize = figsize
        self._save_dir = Path(save_dir) if save_dir else None
        self._style = style

        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def feature_importance(
        self,
        importances: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> None:
        """Plot horizontal bar chart of feature importances.

        Args:
            importances: Feature name → importance score.
            top_n: Number of top features to show.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names, values = zip(*reversed(sorted_items))

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        bars = ax.barh(range(len(names)), values, color="#2196F3", edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, "feature_importance")

    def learning_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Learning Curve",
    ) -> None:
        """Plot training and validation loss over epochs.

        Args:
            train_losses: Training loss per epoch.
            val_losses: Validation loss per epoch.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train", linewidth=2, color="#2196F3")
        ax.plot(epochs, val_losses, label="Validation", linewidth=2, color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, "learning_curve")

    def prediction_vs_actual(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
        title: str = "Predicted vs Actual",
    ) -> None:
        """Scatter plot and time series of predictions vs actuals.

        Args:
            actual: True values.
            predicted: Model predictions.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(actual, predicted, alpha=0.3, s=10, color="#2196F3")
        lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        ax1.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Scatter: Predicted vs Actual")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Time series
        ax2 = axes[1]
        ax2.plot(actual.index, actual.values, label="Actual", linewidth=1, alpha=0.8)
        ax2.plot(actual.index, predicted[:len(actual)], label="Predicted", linewidth=1, alpha=0.8)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.set_title("Time Series: Predicted vs Actual")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, "prediction_vs_actual")

    def residual_analysis(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
        title: str = "Residual Analysis",
    ) -> None:
        """Plot residual distribution and time series.

        Args:
            actual: True values.
            predicted: Model predictions.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        residuals = actual.values - predicted[:len(actual)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Residual time series
        axes[0].plot(actual.index, residuals, linewidth=0.5, alpha=0.7)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Residuals Over Time")
        axes[0].grid(alpha=0.3)

        # Histogram
        axes[1].hist(residuals, bins=50, color="#2196F3", edgecolor="white", density=True)
        axes[1].set_title("Residual Distribution")
        axes[1].grid(alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        axes[2].grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, "residual_analysis")

    def model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str = "rmse",
        title: str = "Model Comparison",
    ) -> None:
        """Bar chart comparing models on a single metric.

        Args:
            metrics: Dict of model_name → metrics dict.
            metric_name: Which metric to plot.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        models = list(metrics.keys())
        values = [m.get(metric_name, 0) for m in metrics.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

        ax.set_ylabel(metric_name.upper())
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_or_show(fig, "model_comparison")

    def _save_or_show(self, fig: object, name: str) -> None:
        """Save figure to disk or show it."""
        import matplotlib.pyplot as plt

        if self._save_dir:
            path = self._save_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved plot: %s", path)
        else:
            plt.show()
