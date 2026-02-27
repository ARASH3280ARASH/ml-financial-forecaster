"""Feature Store — Registry for computed features with versioning and metadata.

Tracks which features have been computed, their lineage, and provides
a unified interface for assembling feature matrices from registered features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a registered feature.

    Attributes:
        name: Unique feature name.
        group: Feature group (e.g. "technical", "statistical").
        description: Human-readable description.
        dependencies: Input columns required to compute this feature.
        version: Feature version string.
        created_at: Unix timestamp of creation.
    """

    name: str
    group: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)


class FeatureStore:
    """Registry and cache for computed features.

    Provides a single source of truth for feature definitions and
    their computed values. Supports persistence to disk for
    reproducibility.

    Args:
        cache_dir: Directory for persisting feature data.

    Example:
        >>> store = FeatureStore()
        >>> store.register("rsi_14", group="technical", data=rsi_series)
        >>> matrix = store.get_feature_matrix(["rsi_14", "macd_hist"])
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._registry: Dict[str, FeatureMetadata] = {}
        self._data: Dict[str, pd.Series] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def feature_names(self) -> List[str]:
        """List all registered feature names."""
        return sorted(self._registry.keys())

    @property
    def feature_count(self) -> int:
        """Number of registered features."""
        return len(self._registry)

    def register(
        self,
        name: str,
        group: str,
        data: pd.Series,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a new feature with its computed values.

        Args:
            name: Unique feature identifier.
            group: Feature category.
            data: Computed feature values as a pandas Series.
            description: Human-readable description.
            dependencies: Input columns used to compute this feature.

        Raises:
            ValueError: If name is empty or data is not a Series.
        """
        if not name:
            raise ValueError("Feature name cannot be empty")
        if not isinstance(data, pd.Series):
            raise ValueError(f"Expected pd.Series, got {type(data)}")

        self._registry[name] = FeatureMetadata(
            name=name,
            group=group,
            description=description,
            dependencies=dependencies or [],
        )
        self._data[name] = data.copy()
        logger.debug("Registered feature: %s (group=%s, length=%d)", name, group, len(data))

    def get(self, name: str) -> pd.Series:
        """Retrieve a single feature's data.

        Args:
            name: Feature name.

        Returns:
            Feature values.

        Raises:
            KeyError: If feature is not registered.
        """
        if name not in self._data:
            raise KeyError(f"Feature not found: {name}")
        return self._data[name].copy()

    def get_feature_matrix(
        self,
        names: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Assemble a feature matrix from registered features.

        Args:
            names: Specific feature names. Uses all if None.
            groups: Filter by groups. Ignored if names is provided.

        Returns:
            DataFrame with selected features as columns.
        """
        if names:
            selected = names
        elif groups:
            selected = [
                n for n, m in self._registry.items() if m.group in groups
            ]
        else:
            selected = self.feature_names

        missing = [n for n in selected if n not in self._data]
        if missing:
            logger.warning("Features not found (skipped): %s", missing)
            selected = [n for n in selected if n in self._data]

        if not selected:
            return pd.DataFrame()

        return pd.DataFrame({n: self._data[n] for n in selected})

    def get_metadata(self, name: str) -> FeatureMetadata:
        """Get metadata for a feature.

        Args:
            name: Feature name.

        Returns:
            FeatureMetadata dataclass.

        Raises:
            KeyError: If not found.
        """
        if name not in self._registry:
            raise KeyError(f"Feature not registered: {name}")
        return self._registry[name]

    def list_groups(self) -> Dict[str, int]:
        """Count features per group.

        Returns:
            Dict mapping group names to feature counts.
        """
        groups: Dict[str, int] = {}
        for meta in self._registry.values():
            groups[meta.group] = groups.get(meta.group, 0) + 1
        return groups

    def save_registry(self, path: Optional[str] = None) -> Path:
        """Persist feature registry metadata to JSON.

        Args:
            path: Output file path. Defaults to cache_dir/registry.json.

        Returns:
            Path to saved file.
        """
        out = Path(path) if path else (self._cache_dir or Path(".")) / "registry.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {name: asdict(meta) for name, meta in self._registry.items()}
        out.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved feature registry (%d features) to %s", len(data), out)
        return out

    def summary(self) -> str:
        """Return a human-readable summary of the feature store.

        Returns:
            Multi-line summary string.
        """
        groups = self.list_groups()
        lines = [f"FeatureStore: {self.feature_count} features in {len(groups)} groups"]
        for group, count in sorted(groups.items()):
            lines.append(f"  {group}: {count} features")
        return "\n".join(lines)
