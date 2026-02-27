"""Hyperparameter tuner — Optuna-based Bayesian optimization for model tuning.

Supports temporal cross-validation to prevent overfitting on
financial time series data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Results from hyperparameter optimization.

    Attributes:
        best_params: Optimal hyperparameters found.
        best_score: Best objective value achieved.
        n_trials: Total number of trials run.
        elapsed_time: Wall-clock tuning time.
        trial_history: Per-trial scores.
    """

    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    elapsed_time: float
    trial_history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Tuning: best_score={self.best_score:.6f}, "
            f"trials={self.n_trials}, time={self.elapsed_time:.1f}s"
        )


class HyperparameterTuner:
    """Bayesian hyperparameter optimization using Optuna.

    Searches the hyperparameter space while respecting temporal
    ordering constraints.

    Args:
        n_trials: Number of optimization trials.
        metric: Metric to optimize ('rmse', 'mae', 'r2').
        direction: 'minimize' or 'maximize'.
        cv_splits: Number of temporal CV splits.
        timeout: Max seconds for the entire study.

    Example:
        >>> tuner = HyperparameterTuner(n_trials=100)
        >>> result = tuner.tune(model_factory, param_space, X, y)
        >>> print(result.best_params)
    """

    def __init__(
        self,
        n_trials: int = 50,
        metric: str = "rmse",
        direction: str = "minimize",
        cv_splits: int = 5,
        timeout: Optional[int] = None,
    ) -> None:
        self._n_trials = n_trials
        self._metric = metric
        self._direction = direction
        self._cv_splits = cv_splits
        self._timeout = timeout

    def tune(
        self,
        model_factory: Callable[[Dict[str, Any]], BaseModel],
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TuningResult:
        """Run hyperparameter optimization.

        Args:
            model_factory: Callable that creates a model from params.
            param_space: Parameter search space definition.
            X: Feature matrix.
            y: Target series.

        Returns:
            TuningResult with best parameters and history.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed — running random search fallback")
            return self._random_search(model_factory, param_space, X, y)

        start = time.time()
        trial_history: List[Dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)
            score = self._evaluate_params(model_factory, params, X, y)
            trial_history.append({"params": params, "score": score})
            return score

        study = optuna.create_study(direction=self._direction)
        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
        )

        elapsed = time.time() - start

        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            elapsed_time=elapsed,
            trial_history=trial_history,
        )

        logger.info("Hyperparameter tuning complete: %s", result.summary())
        return result

    def _sample_params(self, trial: Any, space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from the search space."""
        params = {}
        for name, spec in space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                log = spec.get("log", False)
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _evaluate_params(
        self,
        model_factory: Callable,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Evaluate parameters using temporal cross-validation."""
        n = len(X)
        fold_size = n // (self._cv_splits + 1)
        scores = []

        for fold in range(self._cv_splits):
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                continue

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_val = X.iloc[val_start:val_end]
            y_val = y.iloc[val_start:val_end]

            model = model_factory(params)
            X_clean = X_tr.fillna(0).replace([np.inf, -np.inf], 0)
            X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)

            model.fit(X_clean, y_tr)
            metrics = model.evaluate(X_val_clean, y_val)

            if self._metric == "rmse":
                scores.append(metrics.rmse)
            elif self._metric == "mae":
                scores.append(metrics.mae)
            elif self._metric == "r2":
                scores.append(metrics.r2)

        return float(np.mean(scores)) if scores else float("inf")

    def _random_search(
        self,
        model_factory: Callable,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TuningResult:
        """Fallback random search when Optuna is unavailable."""
        rng = np.random.default_rng(42)
        start = time.time()
        best_score = float("inf") if self._direction == "minimize" else float("-inf")
        best_params: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []

        for i in range(self._n_trials):
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "int":
                    params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
                elif spec["type"] == "float":
                    if spec.get("log"):
                        params[name] = float(np.exp(
                            rng.uniform(np.log(spec["low"]), np.log(spec["high"]))
                        ))
                    else:
                        params[name] = float(rng.uniform(spec["low"], spec["high"]))
                elif spec["type"] == "categorical":
                    params[name] = rng.choice(spec["choices"])

            score = self._evaluate_params(model_factory, params, X, y)
            history.append({"params": params, "score": score})

            is_better = (
                score < best_score if self._direction == "minimize"
                else score > best_score
            )
            if is_better:
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=self._n_trials,
            elapsed_time=time.time() - start,
            trial_history=history,
        )
