"""ML-aware backtester — evaluate model predictions as trading strategies.

Simulates strategy execution with realistic constraints including
transaction costs, slippage, and position sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import FinancialMetrics, MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration.

    Attributes:
        initial_capital: Starting capital.
        transaction_cost_bps: Cost per trade in basis points.
        slippage_bps: Slippage per trade in basis points.
        max_position_size: Maximum position as fraction of capital.
        risk_per_trade: Maximum risk per trade as fraction of capital.
    """

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_position_size: float = 1.0
    risk_per_trade: float = 0.02


@dataclass
class BacktestResult:
    """Results from a backtest run.

    Attributes:
        equity_curve: Portfolio value over time.
        returns: Period returns.
        positions: Position history.
        trades: Number of trades executed.
        metrics: Financial performance metrics.
        strategy_name: Name of the strategy tested.
    """

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: int
    metrics: FinancialMetrics
    strategy_name: str = ""

    def summary(self) -> str:
        return (
            f"Backtest({self.strategy_name}): "
            f"Return={self.metrics.total_return:.2%} | "
            f"Trades={self.trades} | "
            f"{self.metrics.summary()}"
        )


class Backtester:
    """Backtest ML predictions as trading strategies.

    Converts model predictions into positions, applies transaction
    costs and slippage, and computes performance metrics.

    Args:
        config: Backtesting configuration.

    Example:
        >>> bt = Backtester()
        >>> result = bt.run(predictions, actual_prices)
        >>> print(result.summary())
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self._config = config or BacktestConfig()
        self._metrics_calc = MetricsCalculator()

    def run(
        self,
        predictions: pd.Series,
        prices: pd.Series,
        strategy_name: str = "ml_strategy",
    ) -> BacktestResult:
        """Run a backtest on model predictions.

        Args:
            predictions: Model predictions (positive = bullish signal).
            prices: Actual price series (aligned with predictions).
            strategy_name: Label for the strategy.

        Returns:
            BacktestResult with equity curve and metrics.
        """
        # Align series
        common_idx = predictions.index.intersection(prices.index)
        predictions = predictions.loc[common_idx]
        prices = prices.loc[common_idx]

        # Generate positions from predictions
        positions = self._generate_positions(predictions)

        # Calculate returns with costs
        price_returns = prices.pct_change().fillna(0)
        strategy_returns = self._apply_costs(positions, price_returns)

        # Build equity curve
        equity = self._config.initial_capital * (1 + strategy_returns).cumprod()

        # Count trades
        n_trades = int((positions.diff().abs() > 0).sum())

        # Calculate metrics
        metrics = self._metrics_calc.from_returns(strategy_returns)

        result = BacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            positions=positions,
            trades=n_trades,
            metrics=metrics,
            strategy_name=strategy_name,
        )

        logger.info("Backtest complete: %s", result.summary())
        return result

    def run_comparison(
        self,
        strategies: Dict[str, pd.Series],
        prices: pd.Series,
    ) -> Dict[str, BacktestResult]:
        """Compare multiple prediction strategies.

        Args:
            strategies: Dict mapping strategy names to prediction series.
            prices: Actual price series.

        Returns:
            Dict mapping strategy names to their results.
        """
        results = {}
        for name, preds in strategies.items():
            results[name] = self.run(preds, prices, strategy_name=name)

        # Log comparison
        logger.info("Strategy comparison:")
        for name, result in sorted(results.items(), key=lambda x: x[1].metrics.sharpe_ratio, reverse=True):
            logger.info("  %s: Sharpe=%.3f, Return=%.2%%", name, result.metrics.sharpe_ratio, result.metrics.total_return * 100)

        return results

    def _generate_positions(self, predictions: pd.Series) -> pd.Series:
        """Convert predictions to position sizes.

        Args:
            predictions: Raw model predictions.

        Returns:
            Position series in [-1, 1].
        """
        # Normalize predictions to [-1, 1]
        pred_std = predictions.std()
        if pred_std > 0:
            normalized = predictions / (3 * pred_std)
        else:
            normalized = predictions

        positions = normalized.clip(-self._config.max_position_size, self._config.max_position_size)
        return positions

    def _apply_costs(
        self, positions: pd.Series, returns: pd.Series
    ) -> pd.Series:
        """Apply transaction costs and slippage to returns.

        Args:
            positions: Position series.
            returns: Raw price returns.

        Returns:
            Net returns after costs.
        """
        # Strategy returns (position × market return)
        strategy_returns = positions.shift(1).fillna(0) * returns

        # Transaction costs on position changes
        turnover = positions.diff().abs().fillna(0)
        cost_rate = (self._config.transaction_cost_bps + self._config.slippage_bps) / 10_000
        costs = turnover * cost_rate

        return strategy_returns - costs

    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int = 252,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation of future paths.

        Args:
            returns: Historical return series to resample from.
            n_simulations: Number of simulation paths.
            n_periods: Number of periods to simulate.
            seed: Random seed.

        Returns:
            DataFrame with simulated equity paths (columns = simulations).
        """
        rng = np.random.default_rng(seed)
        returns_arr = returns.dropna().values

        simulations = np.zeros((n_periods, n_simulations))
        for sim in range(n_simulations):
            sampled = rng.choice(returns_arr, size=n_periods, replace=True)
            simulations[:, sim] = self._config.initial_capital * np.cumprod(1 + sampled)

        return pd.DataFrame(
            simulations,
            columns=[f"sim_{i}" for i in range(n_simulations)],
        )
