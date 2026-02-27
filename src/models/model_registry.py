"""Model registry — versioned storage and retrieval of trained models.

Provides a centralized catalog for managing model versions,
metadata, and serialization/deserialization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.base_model import BaseModel, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a registered model version.

    Attributes:
        model_name: Model identifier.
        version: Semantic version string.
        metrics: Performance metrics at registration time.
        params: Hyperparameters used.
        created_at: Unix timestamp.
        tags: Arbitrary tags (e.g. 'production', 'experimental').
        artifact_path: Path to serialized model file.
    """

    model_name: str
    version: str
    metrics: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    artifact_path: Optional[str] = None

    @property
    def version_id(self) -> str:
        raw = f"{self.model_name}:{self.version}:{self.created_at}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class ModelRegistry:
    """Centralized model version registry.

    Tracks model versions, their performance, and provides
    model loading/saving with metadata persistence.

    Args:
        storage_dir: Directory for model artifacts and registry.

    Example:
        >>> registry = ModelRegistry("models/")
        >>> registry.register(model, version="1.0", tags=["production"])
        >>> best = registry.get_best("gradient_boosting", metric="rmse")
        >>> loaded = registry.load(best.version_id)
    """

    def __init__(self, storage_dir: str = "models/registry") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, ModelVersion] = {}
        self._models: Dict[str, BaseModel] = {}
        self._load_registry()

    @property
    def model_names(self) -> List[str]:
        """Unique model names in registry."""
        return sorted(set(v.model_name for v in self._versions.values()))

    @property
    def version_count(self) -> int:
        return len(self._versions)

    def register(
        self,
        model: BaseModel,
        version: str,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """Register a trained model.

        Args:
            model: Fitted model instance.
            version: Version string (e.g. "1.0.0").
            tags: Metadata tags.

        Returns:
            ModelVersion metadata.

        Raises:
            ValueError: If model is not fitted.
        """
        if not model.is_fitted:
            raise ValueError(f"Cannot register unfitted model: {model.name}")

        metrics_dict = None
        if model.metrics:
            metrics_dict = {
                "rmse": model.metrics.rmse,
                "mae": model.metrics.mae,
                "r2": model.metrics.r2,
                "directional_accuracy": model.metrics.directional_accuracy,
            }

        mv = ModelVersion(
            model_name=model.name,
            version=version,
            metrics=metrics_dict,
            params=model.params,
            tags=tags or [],
        )

        # Save model artifact
        artifact_path = self._storage_dir / f"{mv.version_id}.pkl"
        self._save_artifact(model, artifact_path)
        mv.artifact_path = str(artifact_path)

        self._versions[mv.version_id] = mv
        self._models[mv.version_id] = model
        self._save_registry()

        logger.info("Registered %s v%s (id=%s)", model.name, version, mv.version_id)
        return mv

    def get_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions for a model name."""
        return sorted(
            [v for v in self._versions.values() if v.model_name == model_name],
            key=lambda v: v.created_at,
            reverse=True,
        )

    def get_best(
        self,
        model_name: str,
        metric: str = "rmse",
        minimize: bool = True,
    ) -> Optional[ModelVersion]:
        """Get the best version by a metric.

        Args:
            model_name: Model name to search.
            metric: Metric key (e.g. 'rmse', 'r2').
            minimize: Whether lower is better.

        Returns:
            Best ModelVersion, or None.
        """
        versions = self.get_versions(model_name)
        scored = [
            v for v in versions
            if v.metrics and metric in v.metrics
        ]

        if not scored:
            return None

        return sorted(
            scored,
            key=lambda v: v.metrics[metric],
            reverse=not minimize,
        )[0]

    def load(self, version_id: str) -> BaseModel:
        """Load a model by version ID.

        Args:
            version_id: Unique version identifier.

        Returns:
            Deserialized BaseModel.

        Raises:
            KeyError: If version not found.
        """
        if version_id in self._models:
            return self._models[version_id]

        if version_id not in self._versions:
            raise KeyError(f"Version not found: {version_id}")

        mv = self._versions[version_id]
        if mv.artifact_path and Path(mv.artifact_path).exists():
            model = self._load_artifact(Path(mv.artifact_path))
            self._models[version_id] = model
            return model

        raise FileNotFoundError(f"Artifact not found for {version_id}")

    def compare(self, version_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare metrics across versions.

        Args:
            version_ids: List of version IDs to compare.

        Returns:
            Dict mapping version_id to metrics dict.
        """
        results = {}
        for vid in version_ids:
            if vid in self._versions and self._versions[vid].metrics:
                results[vid] = self._versions[vid].metrics
        return results

    def summary(self) -> str:
        """Human-readable registry summary."""
        lines = [f"ModelRegistry: {self.version_count} versions"]
        for name in self.model_names:
            versions = self.get_versions(name)
            lines.append(f"  {name}: {len(versions)} version(s)")
        return "\n".join(lines)

    def _save_artifact(self, model: BaseModel, path: Path) -> None:
        """Serialize model to disk."""
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def _load_artifact(self, path: Path) -> BaseModel:
        """Deserialize model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_registry(self) -> None:
        """Persist registry metadata."""
        registry_path = self._storage_dir / "registry.json"
        data = {vid: asdict(v) for vid, v in self._versions.items()}
        registry_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_registry(self) -> None:
        """Load registry metadata from disk."""
        registry_path = self._storage_dir / "registry.json"
        if not registry_path.exists():
            return

        try:
            data = json.loads(registry_path.read_text())
            for vid, v_dict in data.items():
                self._versions[vid] = ModelVersion(**v_dict)
            logger.info("Loaded registry with %d versions", len(self._versions))
        except Exception as e:
            logger.warning("Failed to load registry: %s", e)
