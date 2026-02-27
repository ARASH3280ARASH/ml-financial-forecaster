"""Performance visualization — equity curves, drawdown, rolling metrics.

Publication-quality plots for strategy performance analysis
and risk reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """Generate strategy performance visualizations.

    Creates equity curves, drawdown charts, rolling metric plots,
    and risk analysis visualizations.

    Args:
        figsize: Default figure size.
        save_dir: Directory to save plots.

    Example:
        >>> plotter = PerformancePlotter(save_dir="reports/plots/")
        >>> plotter.equity_curve(returns, benchmark_returns)
        >>> plotter.drawdown_chart(returns)
    """

    def __init__(
        self,
        figsize: tuple = (14, 7),
        save_dir: Optional[str] = None,
    ) -> None:
        self._figsize = figsize
        self._save_dir = Path(save_dir) if save_dir else None

        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def equity_curve(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        initial_capital: float = 100_000.0,
    ) -> None:
        """Plot cumulative equity curve with optional benchmark.

        Args:
            returns: Strategy returns.
            benchmark_returns: Benchmark returns for comparison.
            title: Plot title.
            initial_capital: Starting capital.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)

        equity = initial_capital * (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, label="Strategy", linewidth=2, color="#2196F3")

        if benchmark_returns is not None:
            bench_equity = initial_capital * (1 + benchmark_returns).cumprod()
            ax.plot(bench_equity.index, bench_equity.values,
                    label="Benchmark", linewidth=1.5, color="#9E9E9E", linestyle="--")

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, fontsize=11)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        self._save_or_show(fig, "equity_curve")

    def drawdown_chart(
        self,
        returns: pd.Series,
        title: str = "Underwater Plot (Drawdown)",
    ) -> None:
        """Plot drawdown over time.

        Args:
            returns: Strategy returns.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4)
        ax.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, "drawdown")

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60,
        title: str = "Rolling Performance Metrics",
    ) -> None:
        """Plot rolling Sharpe, volatility, and returns.

        Args:
            returns: Strategy returns.
            window: Rolling window size.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Rolling Sharpe
        rf_daily = 0.02 / 252
        excess = returns - rf_daily
        rolling_sharpe = (
            excess.rolling(window).mean() /
            (excess.rolling(window).std() + 1e-10) *
            np.sqrt(252)
        )
        axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, color="#2196F3", linewidth=1.5)
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].set_title(f"Rolling {window}-Period Sharpe Ratio")
        axes[0].grid(alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol.values, color="#FF9800", linewidth=1.5)
        axes[1].set_ylabel("Annualized Volatility")
        axes[1].set_title(f"Rolling {window}-Period Volatility")
        axes[1].grid(alpha=0.3)

        # Rolling return
        rolling_return = returns.rolling(window).mean() * 252
        axes[2].plot(rolling_return.index, rolling_return.values, color="#4CAF50", linewidth=1.5)
        axes[2].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[2].set_ylabel("Annualized Return")
        axes[2].set_title(f"Rolling {window}-Period Annualized Return")
        axes[2].grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        self._save_or_show(fig, "rolling_metrics")

    def return_distribution(
        self,
        returns: pd.Series,
        title: str = "Return Distribution",
    ) -> None:
        """Plot return distribution with VaR/CVaR annotations.

        Args:
            returns: Strategy returns.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._figsize)

        ax.hist(returns.dropna(), bins=100, density=True, color="#2196F3",
                edgecolor="white", alpha=0.7, label="Returns")

        # VaR and CVaR lines
        var_95 = float(returns.quantile(0.05))
        cvar_mask = returns <= var_95
        cvar_95 = float(returns[cvar_mask].mean()) if cvar_mask.any() else var_95

        ax.axvline(var_95, color="#FF5722", linestyle="--", linewidth=2, label=f"VaR 95% ({var_95:.4f})")
        ax.axvline(cvar_95, color="#F44336", linestyle="-.", linewidth=2, label=f"CVaR 95% ({cvar_95:.4f})")
        ax.axvline(float(returns.mean()), color="#4CAF50", linestyle="-", linewidth=2, label=f"Mean ({float(returns.mean()):.4f})")

        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=True)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, "return_distribution")

    def monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
    ) -> None:
        """Heatmap of monthly returns by year.

        Args:
            returns: Daily return series.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_table = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })

        pivot = monthly_table.pivot_table(values="return", index="year", columns="month")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots(figsize=self._figsize)
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, format="%.1%%")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, "monthly_heatmap")

    def _save_or_show(self, fig: object, name: str) -> None:
        """Save or display the figure."""
        import matplotlib.pyplot as plt

        if self._save_dir:
            path = self._save_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved plot: %s", path)
        else:
            plt.show()
