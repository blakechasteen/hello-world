"""
Config Lens: YAML-Configured Domain Lenses

This module provides configuration-based semantic lenses that can be
defined in YAML files for rapid domain customization.

Key Capabilities:
- Load domain specifications from YAML
- Define custom semantic axes with exemplars
- Validate configurations for correctness
- Learn axis directions from exemplar sets
- Combine with learned axes for hybrid approaches

Design Philosophy:
- Configuration should be simple and human-readable
- Domain experts (not ML engineers) should be able to define axes
- Validation catches errors early
- Exemplar-based learning bridges config and data

Usage:
    # Load from YAML
    config = load_domain_config("medical.yaml")
    lens = ConfigLens(config)

    # Project activations
    projection = lens.project(activations)
    print(projection["severity"])  # Medical severity score

    # Create from code
    config = DomainConfig(
        name="medical",
        axes=[
            AxisSpec(name="severity", description="Medical severity level"),
            AxisSpec(name="urgency", description="Treatment urgency"),
        ]
    )
    lens = ConfigLens(config)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class ConfigValidationError(Exception):
    """Error in domain configuration."""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        super().__init__(f"{message}" + (f" (in {path})" if path else ""))


@dataclass
class ExemplarSet:
    """Set of exemplar texts/vectors for learning an axis."""

    positive: list[str | np.ndarray] = field(default_factory=list)
    negative: list[str | np.ndarray] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if exemplar set is empty."""
        return len(self.positive) == 0 and len(self.negative) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for YAML serialization)."""
        result = {}
        if self.positive:
            result["positive"] = [
                x if isinstance(x, str) else x.tolist()
                for x in self.positive
            ]
        if self.negative:
            result["negative"] = [
                x if isinstance(x, str) else x.tolist()
                for x in self.negative
            ]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExemplarSet":
        """Create from dictionary."""
        positive = []
        negative = []

        for item in data.get("positive", []):
            if isinstance(item, list):
                positive.append(np.array(item))
            else:
                positive.append(str(item))

        for item in data.get("negative", []):
            if isinstance(item, list):
                negative.append(np.array(item))
            else:
                negative.append(str(item))

        return cls(positive=positive, negative=negative)


@dataclass
class AxisSpec:
    """Specification for a semantic axis."""

    name: str
    description: str = ""
    exemplars: ExemplarSet | None = None

    # Optional: pre-defined direction vector
    direction: np.ndarray | None = None

    # Learning settings
    learn_from_exemplars: bool = True
    normalize: bool = True

    # Metadata
    category: str | None = None
    importance: float = 1.0
    safety_relevant: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.exemplars and not self.exemplars.is_empty():
            result["exemplars"] = self.exemplars.to_dict()
        if self.direction is not None:
            result["direction"] = self.direction.tolist()
        if self.category:
            result["category"] = self.category
        if self.importance != 1.0:
            result["importance"] = self.importance
        if self.safety_relevant:
            result["safety_relevant"] = True
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AxisSpec":
        """Create from dictionary."""
        exemplars = None
        if "exemplars" in data:
            exemplars = ExemplarSet.from_dict(data["exemplars"])

        direction = None
        if "direction" in data:
            direction = np.array(data["direction"])

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            exemplars=exemplars,
            direction=direction,
            learn_from_exemplars=data.get("learn_from_exemplars", True),
            normalize=data.get("normalize", True),
            category=data.get("category"),
            importance=data.get("importance", 1.0),
            safety_relevant=data.get("safety_relevant", False),
        )


@dataclass
class DomainConfig:
    """Complete domain configuration."""

    name: str
    description: str = ""
    version: str = "1.0"
    axes: list[AxisSpec] = field(default_factory=list)

    # Global settings
    normalize_projections: bool = True
    default_embedding_dim: int = 384

    # Metadata
    author: str | None = None
    created_at: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "axes": [ax.to_dict() for ax in self.axes],
            "settings": {
                "normalize_projections": self.normalize_projections,
                "default_embedding_dim": self.default_embedding_dim,
            },
            "metadata": {
                "author": self.author,
                "created_at": self.created_at,
                "tags": self.tags,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainConfig":
        """Create from dictionary."""
        axes = [AxisSpec.from_dict(ax) for ax in data.get("axes", [])]

        settings = data.get("settings", {})
        metadata = data.get("metadata", {})

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            axes=axes,
            normalize_projections=settings.get("normalize_projections", True),
            default_embedding_dim=settings.get("default_embedding_dim", 384),
            author=metadata.get("author"),
            created_at=metadata.get("created_at"),
            tags=metadata.get("tags", []),
        )

    def get_axis(self, name: str) -> AxisSpec | None:
        """Get axis by name."""
        for ax in self.axes:
            if ax.name == name:
                return ax
        return None

    def safety_relevant_axes(self) -> list[AxisSpec]:
        """Get all safety-relevant axes."""
        return [ax for ax in self.axes if ax.safety_relevant]


def validate_domain_config(config: DomainConfig) -> list[str]:
    """
    Validate a domain configuration.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Check name
    if not config.name:
        errors.append("Domain name is required")
    elif not config.name.replace("_", "").replace("-", "").isalnum():
        errors.append(f"Invalid domain name: {config.name}")

    # Check axes
    if not config.axes:
        errors.append("At least one axis is required")

    axis_names = set()
    for i, axis in enumerate(config.axes):
        # Check axis name
        if not axis.name:
            errors.append(f"Axis {i}: name is required")
        elif axis.name in axis_names:
            errors.append(f"Duplicate axis name: {axis.name}")
        else:
            axis_names.add(axis.name)

        # Check direction dimension
        if axis.direction is not None:
            if len(axis.direction) != config.default_embedding_dim:
                errors.append(
                    f"Axis '{axis.name}': direction dimension "
                    f"({len(axis.direction)}) doesn't match "
                    f"default_embedding_dim ({config.default_embedding_dim})"
                )

        # Check importance
        if axis.importance <= 0:
            errors.append(f"Axis '{axis.name}': importance must be positive")

    return errors


def load_domain_config(path: str | Path) -> DomainConfig:
    """
    Load domain configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        DomainConfig object

    Raises:
        ConfigValidationError: If config is invalid
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Try to import yaml
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for loading domain configs. "
            "Install with: pip install pyyaml"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = DomainConfig.from_dict(data)

    # Validate
    errors = validate_domain_config(config)
    if errors:
        raise ConfigValidationError(
            f"Invalid configuration: {'; '.join(errors)}",
            path=str(path)
        )

    return config


def save_domain_config(config: DomainConfig, path: str | Path) -> None:
    """
    Save domain configuration to YAML file.

    Args:
        config: DomainConfig to save
        path: Path to save to
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for saving domain configs. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


class ConfigLens:
    """
    YAML-configured semantic lens.

    Projects activations onto domain-specific semantic space
    defined by configuration.
    """

    def __init__(
        self,
        config: DomainConfig,
        embed_fn: callable | None = None,
    ):
        """
        Initialize config lens.

        Args:
            config: Domain configuration
            embed_fn: Optional embedding function for text exemplars
        """
        self.config = config
        self.embed_fn = embed_fn
        self._directions: dict[str, np.ndarray] = {}
        self._fitted = False

        # Initialize directions from config
        self._init_directions()

    def _init_directions(self) -> None:
        """Initialize axis directions from config."""
        for axis in self.config.axes:
            if axis.direction is not None:
                direction = axis.direction.copy()
                if axis.normalize:
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                self._directions[axis.name] = direction

    def fit(
        self,
        activations: np.ndarray | None = None,
        embed_fn: callable | None = None,
    ) -> None:
        """
        Fit lens by learning directions from exemplars.

        Args:
            activations: Optional background activations for regularization
            embed_fn: Embedding function for text exemplars
        """
        embed_fn = embed_fn or self.embed_fn

        for axis in self.config.axes:
            # Skip if already has direction or no exemplars
            if axis.name in self._directions:
                continue
            if not axis.learn_from_exemplars:
                continue
            if axis.exemplars is None or axis.exemplars.is_empty():
                continue

            direction = self._learn_direction(axis, embed_fn)
            if direction is not None:
                self._directions[axis.name] = direction

        self._fitted = True

    def _learn_direction(
        self,
        axis: AxisSpec,
        embed_fn: callable | None,
    ) -> np.ndarray | None:
        """Learn axis direction from exemplars."""
        if axis.exemplars is None:
            return None

        pos_vecs = []
        neg_vecs = []

        # Process positive exemplars
        for ex in axis.exemplars.positive:
            if isinstance(ex, np.ndarray):
                pos_vecs.append(ex)
            elif embed_fn is not None:
                vec = embed_fn(ex)
                if vec is not None:
                    pos_vecs.append(np.array(vec))

        # Process negative exemplars
        for ex in axis.exemplars.negative:
            if isinstance(ex, np.ndarray):
                neg_vecs.append(ex)
            elif embed_fn is not None:
                vec = embed_fn(ex)
                if vec is not None:
                    neg_vecs.append(np.array(vec))

        if not pos_vecs:
            return None

        # Compute direction
        pos_mean = np.mean(pos_vecs, axis=0)

        if neg_vecs:
            neg_mean = np.mean(neg_vecs, axis=0)
            direction = pos_mean - neg_mean
        else:
            direction = pos_mean

        # Normalize
        if axis.normalize:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

        return direction

    def project(self, activations: np.ndarray) -> dict[str, float]:
        """
        Project activations onto domain semantic space.

        Args:
            activations: Activation vector or matrix

        Returns:
            Dictionary mapping axis names to projection values
        """
        if activations.ndim > 1:
            activations = np.mean(activations, axis=0)

        projection = {}
        for axis in self.config.axes:
            if axis.name in self._directions:
                direction = self._directions[axis.name]
                value = float(np.dot(activations, direction))
                # Apply importance weighting
                value *= axis.importance
                projection[axis.name] = value
            else:
                projection[axis.name] = 0.0

        # Normalize if configured
        if self.config.normalize_projections:
            values = list(projection.values())
            if values:
                max_val = max(abs(v) for v in values) or 1.0
                projection = {k: v / max_val for k, v in projection.items()}

        return projection

    def transform(self, activations: np.ndarray) -> np.ndarray:
        """
        Transform activations to domain coordinates.

        Args:
            activations: (n_samples, n_features) or (n_features,)

        Returns:
            (n_samples, n_axes) or (n_axes,) coordinate array
        """
        axes = [ax for ax in self.config.axes if ax.name in self._directions]
        if not axes:
            return np.zeros((0,) if activations.ndim == 1 else (activations.shape[0], 0))

        directions = np.stack([self._directions[ax.name] for ax in axes])

        if activations.ndim == 1:
            return np.dot(directions, activations)
        return np.dot(activations, directions.T)

    def get_axis_direction(self, name: str) -> np.ndarray | None:
        """Get the direction vector for an axis."""
        return self._directions.get(name)

    def get_axis_names(self) -> list[str]:
        """Get names of all configured axes."""
        return [ax.name for ax in self.config.axes]

    def get_fitted_axes(self) -> list[str]:
        """Get names of axes with learned/set directions."""
        return list(self._directions.keys())

    def safety_projection(self, activations: np.ndarray) -> dict[str, float]:
        """
        Project onto safety-relevant axes only.

        Returns projection values for axes marked as safety_relevant.
        """
        projection = self.project(activations)
        safety_axes = self.config.safety_relevant_axes()
        return {ax.name: projection.get(ax.name, 0.0) for ax in safety_axes}

    def to_dict(self) -> dict[str, Any]:
        """Convert lens state to dictionary."""
        return {
            "config": self.config.to_dict(),
            "directions": {
                name: dir.tolist()
                for name, dir in self._directions.items()
            },
            "fitted": self._fitted,
        }


def create_lens_from_yaml(path: str | Path) -> ConfigLens:
    """
    Create a ConfigLens from a YAML file.

    Convenience function that loads config and creates lens.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configured ConfigLens
    """
    config = load_domain_config(path)
    return ConfigLens(config)


def create_minimal_config(
    name: str,
    axis_names: list[str],
) -> DomainConfig:
    """
    Create a minimal domain config with just axis names.

    Useful for quick prototyping.

    Args:
        name: Domain name
        axis_names: List of axis names

    Returns:
        DomainConfig with specified axes
    """
    return DomainConfig(
        name=name,
        description=f"Minimal {name} domain",
        axes=[AxisSpec(name=n) for n in axis_names],
    )
