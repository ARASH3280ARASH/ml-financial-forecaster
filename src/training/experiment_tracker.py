"""Experiment tracker — logs, compares, and persists training experiments.

Provides a lightweight local alternative to MLflow/W&B for
tracking model training experiments and their results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A single training experiment record.

    Attributes:
        experiment_id: Unique experiment identifier.
        model_name: Name of the model trained.
        params: Hyperparameters used.
        metrics: Evaluation metrics.
        tags: Experiment tags for filtering.
        notes: Free-form notes.
        created_at: Unix timestamp.
    """

    experiment_id: str
    model_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: float = field(default_factory=time.time)


class ExperimentTracker:
    """Track and compare ML experiments.

    Logs experiment parameters, metrics, and metadata to disk
    for reproducibility and comparison.

    Args:
        log_dir: Directory for experiment logs.

    Example:
        >>> tracker = ExperimentTracker("experiments/")
        >>> exp = tracker.log_experiment(
        ...     model_name="xgboost_v1",
        ...     params={"max_depth": 6},
        ...     metrics={"rmse": 0.012, "r2": 0.85},
        ... )
        >>> tracker.compare_experiments(metric="rmse")
    """

    def __init__(self, log_dir: str = "experiments") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    @property
    def n_experiments(self) -> int:
        return len(self._experiments)

    def log_experiment(
        self,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> Experiment:
        """Log a new experiment.

        Args:
            model_name: Model identifier.
            params: Hyperparameters.
            metrics: Performance metrics.
            tags: Experiment tags.
            notes: Free-form notes.

        Returns:
            The logged Experiment.
        """
        exp_id = f"{model_name}_{int(time.time())}_{self.n_experiments}"

        exp = Experiment(
            experiment_id=exp_id,
            model_name=model_name,
            params=params or {},
            metrics=metrics or {},
            tags=tags or [],
            notes=notes,
        )

        self._experiments[exp_id] = exp
        self._save_experiment(exp)

        logger.info("Logged experiment: %s (metrics: %s)", exp_id, exp.metrics)
        return exp

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Experiment record.

        Raises:
            KeyError: If not found.
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]

    def list_experiments(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Experiment]:
        """List experiments with optional filtering.

        Args:
            model_name: Filter by model name.
            tags: Filter by tags (any match).

        Returns:
            Filtered list of experiments.
        """
        result = list(self._experiments.values())

        if model_name:
            result = [e for e in result if e.model_name == model_name]

        if tags:
            tag_set = set(tags)
            result = [e for e in result if tag_set.intersection(e.tags)]

        return sorted(result, key=lambda e: e.created_at, reverse=True)

    def compare_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        metric: str = "rmse",
    ) -> pd.DataFrame:
        """Create a comparison table of experiments.

        Args:
            experiment_ids: Specific experiments. Uses all if None.
            metric: Primary sort metric.

        Returns:
            DataFrame with experiment comparison.
        """
        if experiment_ids:
            exps = [self._experiments[eid] for eid in experiment_ids if eid in self._experiments]
        else:
            exps = list(self._experiments.values())

        rows = []
        for exp in exps:
            row = {
                "experiment_id": exp.experiment_id,
                "model": exp.model_name,
                "tags": ", ".join(exp.tags),
            }
            row.update(exp.metrics)
            row.update({f"param_{k}": v for k, v in exp.params.items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        if metric in df.columns:
            df = df.sort_values(metric)

        return df

    def get_best_experiment(
        self,
        model_name: Optional[str] = None,
        metric: str = "rmse",
        minimize: bool = True,
    ) -> Optional[Experiment]:
        """Get the best experiment by a metric.

        Args:
            model_name: Filter by model name.
            metric: Metric to optimize.
            minimize: Whether lower is better.

        Returns:
            Best experiment, or None.
        """
        exps = self.list_experiments(model_name=model_name)
        scored = [e for e in exps if metric in e.metrics]

        if not scored:
            return None

        return sorted(
            scored,
            key=lambda e: e.metrics[metric],
            reverse=not minimize,
        )[0]

    def _save_experiment(self, exp: Experiment) -> None:
        """Persist experiment to disk."""
        path = self._log_dir / f"{exp.experiment_id}.json"
        path.write_text(json.dumps(asdict(exp), indent=2, default=str))

    def _load_experiments(self) -> None:
        """Load all experiments from disk."""
        for path in self._log_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                exp = Experiment(**data)
                self._experiments[exp.experiment_id] = exp
            except Exception as e:
                logger.warning("Failed to load %s: %s", path.name, e)

        if self._experiments:
            logger.info("Loaded %d experiments from %s", len(self._experiments), self._log_dir)

    def summary(self) -> str:
        """Human-readable summary."""
        models = set(e.model_name for e in self._experiments.values())
        return (
            f"ExperimentTracker: {self.n_experiments} experiments, "
            f"{len(models)} unique models"
        )
