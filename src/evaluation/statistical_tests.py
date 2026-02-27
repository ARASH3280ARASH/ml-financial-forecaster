"""Statistical tests — hypothesis testing for model validation.

Provides rigorous statistical tests to determine whether model
performance is statistically significant or due to chance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a statistical hypothesis test.

    Attributes:
        test_name: Name of the test.
        statistic: Test statistic value.
        p_value: p-value.
        significant: Whether result is significant at alpha level.
        alpha: Significance level used.
        interpretation: Human-readable interpretation.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    interpretation: str = ""

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} ({sig} at α={self.alpha})"
        )


class StatisticalTester:
    """Statistical hypothesis testing for model evaluation.

    Tests whether model predictions are statistically better than
    benchmarks and validates key assumptions.

    Args:
        alpha: Default significance level.

    Example:
        >>> tester = StatisticalTester()
        >>> result = tester.paired_t_test(model_errors, baseline_errors)
        >>> print(result.summary())
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def paired_t_test(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        alternative: str = "less",
    ) -> TestResult:
        """Paired t-test comparing two models' errors.

        Tests H0: mean(errors_a) >= mean(errors_b) vs
        H1: mean(errors_a) < mean(errors_b).

        Args:
            errors_a: Absolute errors from model A.
            errors_b: Absolute errors from model B.
            alternative: 'less', 'greater', or 'two-sided'.

        Returns:
            TestResult.
        """
        stat, p = stats.ttest_rel(errors_a, errors_b, alternative=alternative)

        significant = p < self._alpha
        interp = (
            f"Model A {'significantly' if significant else 'does not significantly'} "
            f"{'outperform' if significant else 'outperform'} Model B"
        )

        return TestResult(
            test_name="Paired t-test",
            statistic=float(stat),
            p_value=float(p),
            significant=significant,
            alpha=self._alpha,
            interpretation=interp,
        )

    def diebold_mariano(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        h: int = 1,
    ) -> TestResult:
        """Diebold-Mariano test for predictive accuracy.

        Tests whether two forecasts have equal predictive accuracy.

        Args:
            errors_a: Squared errors from model A.
            errors_b: Squared errors from model B.
            h: Forecast horizon.

        Returns:
            TestResult.
        """
        d = errors_a ** 2 - errors_b ** 2
        n = len(d)
        mean_d = np.mean(d)

        # Newey-West variance estimate
        var_d = np.var(d, ddof=1)
        for k in range(1, h):
            gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
            var_d += 2 * (1 - k / h) * gamma_k

        dm_stat = mean_d / np.sqrt(var_d / n + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return TestResult(
            test_name="Diebold-Mariano",
            statistic=float(dm_stat),
            p_value=float(p_value),
            significant=p_value < self._alpha,
            alpha=self._alpha,
            interpretation="Tests equal predictive accuracy between two models",
        )

    def bootstrap_confidence_interval(
        self,
        metric_values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for a metric.

        Args:
            metric_values: Sample values.
            confidence: Confidence level.
            n_bootstrap: Number of bootstrap samples.
            seed: Random seed.

        Returns:
            Tuple of (lower_bound, point_estimate, upper_bound).
        """
        rng = np.random.default_rng(seed)
        n = len(metric_values)

        boot_means = np.array([
            np.mean(rng.choice(metric_values, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])

        alpha = 1 - confidence
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        point = float(np.mean(metric_values))

        logger.info(
            "Bootstrap CI (%.0f%%): %.6f [%.6f, %.6f]",
            confidence * 100, point, lower, upper,
        )

        return lower, point, upper

    def sharpe_ratio_test(
        self, returns: pd.Series, benchmark_sharpe: float = 0.0
    ) -> TestResult:
        """Test whether Sharpe ratio is significantly different from benchmark.

        Uses the Lo (2002) adjustment for autocorrelated returns.

        Args:
            returns: Return series.
            benchmark_sharpe: Benchmark Sharpe ratio to test against.

        Returns:
            TestResult.
        """
        n = len(returns)
        sr = float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252))

        # Standard error with autocorrelation correction
        rho1 = float(returns.autocorr(lag=1)) if n > 1 else 0.0
        se = np.sqrt((1 + 2 * rho1 ** 2) / n) * np.sqrt(252)

        z_stat = (sr - benchmark_sharpe) / (se + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return TestResult(
            test_name="Sharpe Ratio Test (Lo 2002)",
            statistic=float(z_stat),
            p_value=float(p_value),
            significant=p_value < self._alpha,
            alpha=self._alpha,
            interpretation=f"SR={sr:.3f} vs benchmark={benchmark_sharpe:.3f}",
        )

    def normality_test(self, returns: pd.Series) -> TestResult:
        """Jarque-Bera test for return normality.

        Args:
            returns: Return series.

        Returns:
            TestResult.
        """
        stat, p = stats.jarque_bera(returns.dropna())

        return TestResult(
            test_name="Jarque-Bera Normality",
            statistic=float(stat),
            p_value=float(p),
            significant=p < self._alpha,
            alpha=self._alpha,
            interpretation="Rejects normality" if p < self._alpha else "Cannot reject normality",
        )

    def stationarity_test(self, series: pd.Series) -> TestResult:
        """Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series to test.

        Returns:
            TestResult.
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), autolag="AIC")
        stat, p = result[0], result[1]

        return TestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=float(stat),
            p_value=float(p),
            significant=p < self._alpha,
            alpha=self._alpha,
            interpretation="Stationary" if p < self._alpha else "Non-stationary",
        )

    def run_all_tests(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> List[TestResult]:
        """Run a comprehensive test suite.

        Args:
            returns: Strategy returns.
            benchmark_returns: Benchmark returns for comparison.

        Returns:
            List of TestResults.
        """
        results = [
            self.normality_test(returns),
            self.stationarity_test(returns),
            self.sharpe_ratio_test(returns),
        ]

        if benchmark_returns is not None:
            errors_a = returns.values ** 2
            errors_b = benchmark_returns.values[:len(errors_a)] ** 2
            min_len = min(len(errors_a), len(errors_b))
            results.append(
                self.diebold_mariano(errors_a[:min_len], errors_b[:min_len])
            )

        for r in results:
            logger.info(r.summary())

        return results
