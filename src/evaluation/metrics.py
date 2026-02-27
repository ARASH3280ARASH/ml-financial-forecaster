"""Financial evaluation metrics — Sharpe, Sortino, Calmar, drawdown, VaR/CVaR.

Extends standard ML metrics with finance-specific performance
measures used in quantitative research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
HOURS_PER_YEAR = 252 * 6.5


@dataclass
class FinancialMetrics:
    """Comprehensive financial performance metrics.

    Attributes:
        total_return: Cumulative return.
        annualized_return: Annualized return.
        annualized_volatility: Annualized standard deviation.
        sharpe_ratio: Risk-adjusted return (excess return / volatility).
        sortino_ratio: Downside risk-adjusted return.
        calmar_ratio: Return / max drawdown.
        max_drawdown: Largest peak-to-trough decline.
        max_drawdown_duration: Longest drawdown period (bars).
        var_95: Value at Risk at 95% confidence.
        cvar_95: Conditional VaR (Expected Shortfall).
        win_rate: Fraction of positive return periods.
        profit_factor: Gross profit / gross loss.
        skewness: Return distribution skewness.
        kurtosis: Return distribution excess kurtosis.
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    def summary(self) -> str:
        return (
            f"Sharpe={self.sharpe_ratio:.3f} | Sortino={self.sortino_ratio:.3f} | "
            f"Calmar={self.calmar_ratio:.3f} | MaxDD={self.max_drawdown:.2%} | "
            f"WinRate={self.win_rate:.2%}"
        )


class MetricsCalculator:
    """Calculate financial performance metrics from returns or predictions.

    Supports both raw returns and model prediction-based metrics.

    Args:
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.
        periods_per_year: Trading periods per year (252 for daily).

    Example:
        >>> calc = MetricsCalculator()
        >>> metrics = calc.from_returns(daily_returns)
        >>> print(metrics.sharpe_ratio)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self._rf = risk_free_rate
        self._periods = periods_per_year

    def from_returns(self, returns: pd.Series) -> FinancialMetrics:
        """Calculate all metrics from a return series.

        Args:
            returns: Period returns (not cumulative).

        Returns:
            FinancialMetrics dataclass.
        """
        if len(returns) < 2:
            return FinancialMetrics()

        returns = returns.dropna()

        total_return = float((1 + returns).prod() - 1)
        ann_return = float((1 + total_return) ** (self._periods / len(returns)) - 1)
        ann_vol = float(returns.std() * np.sqrt(self._periods))

        # Sharpe ratio
        excess_return = ann_return - self._rf
        sharpe = excess_return / (ann_vol + 1e-10)

        # Sortino ratio
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(self._periods)) if len(downside) > 0 else 1e-10
        sortino = excess_return / (downside_vol + 1e-10)

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = float(drawdown.min())
        dd_duration = self._max_drawdown_duration(drawdown)

        # Calmar ratio
        calmar = ann_return / (abs(max_dd) + 1e-10)

        # VaR and CVaR
        var_95 = float(returns.quantile(0.05))
        cvar_mask = returns <= var_95
        cvar_95 = float(returns[cvar_mask].mean()) if cvar_mask.any() else var_95

        # Win rate and profit factor
        win_rate = float((returns > 0).mean())
        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(abs(returns[returns < 0].sum()))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        return FinancialMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurtosis()),
        )

    def from_predictions(
        self,
        predictions: np.ndarray,
        actual_returns: pd.Series,
        threshold: float = 0.0,
    ) -> FinancialMetrics:
        """Calculate metrics from model predictions used as trading signals.

        Args:
            predictions: Model output (predicted returns).
            actual_returns: Realized returns.
            threshold: Minimum prediction to take a position.

        Returns:
            FinancialMetrics for the prediction-based strategy.
        """
        # Generate positions: long if predicted return > threshold, else flat
        positions = pd.Series(
            np.where(predictions > threshold, 1.0, 0.0),
            index=actual_returns.index[-len(predictions):],
        )

        strategy_returns = positions * actual_returns.iloc[-len(predictions):]
        return self.from_returns(strategy_returns)

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Find the longest drawdown period in bars."""
        is_dd = drawdown < 0
        if not is_dd.any():
            return 0

        durations = []
        current = 0
        for val in is_dd:
            if val:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)

        return max(durations) if durations else 0

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 60,
        risk_free_rate: float = 0.02,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Compute rolling Sharpe ratio.

        Args:
            returns: Return series.
            window: Rolling window size.
            risk_free_rate: Annualized risk-free rate.
            periods_per_year: Periods per year.

        Returns:
            Rolling Sharpe ratio series.
        """
        rf_per_period = risk_free_rate / periods_per_year
        excess = returns - rf_per_period
        return (
            excess.rolling(window).mean()
            / (excess.rolling(window).std() + 1e-10)
            * np.sqrt(periods_per_year)
        )
