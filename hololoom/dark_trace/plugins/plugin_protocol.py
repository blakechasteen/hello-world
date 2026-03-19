"""
Dark Trace Plugin Protocol

Defines the interface for Dark Trace plugins that extend interpretability
capabilities with custom lenses, validators, and analysis tools.

Author: HoloLoom Team
Created: December 2025
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class PluginCategory(Enum):
    """Categories of Dark Trace plugins."""
    LENS = "lens"                       # Custom feature extraction
    VALIDATOR = "validator"             # Causal validation methods
    ANALYZER = "analyzer"               # Analysis tools
    VISUALIZER = "visualizer"           # Visualization generators
    SAFETY = "safety"                   # Safety monitoring
    STEERING = "steering"               # Steering vector generators
    DATASET = "dataset"                 # Dataset loaders
    MODEL_ADAPTER = "model_adapter"     # Model adapters


class PluginStatus(Enum):
    """Status of a plugin in the registry."""
    PENDING = "pending"                 # Not yet loaded
    LOADING = "loading"                 # Currently loading
    ACTIVE = "active"                   # Loaded and active
    DISABLED = "disabled"               # Loaded but disabled
    ERROR = "error"                     # Load/runtime error
    UNVERIFIED = "unverified"           # Signature not verified


@dataclass
class PluginMetadata:
    """Metadata for a Dark Trace plugin."""
    name: str
    version: str
    author: str
    description: str
    category: PluginCategory

    # Dependencies
    requires: list[str] = field(default_factory=list)  # Other plugin dependencies
    python_requires: str = ">=3.9"

    # Capabilities
    provides: list[str] = field(default_factory=list)  # Features this plugin provides

    # Repository info
    homepage: str | None = None
    repository: str | None = None
    license: str = "MIT"

    # Tags for discovery
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category.value,
            "requires": self.requires,
            "python_requires": self.python_requires,
            "provides": self.provides,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMetadata":
        return cls(
            name=data["name"],
            version=data["version"],
            author=data["author"],
            description=data["description"],
            category=PluginCategory(data["category"]),
            requires=data.get("requires", []),
            python_requires=data.get("python_requires", ">=3.9"),
            provides=data.get("provides", []),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            license=data.get("license", "MIT"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class PluginSignature:
    """Cryptographic signature for plugin verification."""
    algorithm: str              # "RSA-SHA256" or "ECDSA-SHA256"
    signature: bytes            # The signature bytes
    public_key_id: str          # Identifier for the public key
    signed_at: datetime
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class PluginRating:
    """Rating for a plugin."""
    plugin_name: str
    plugin_version: str
    user_id: str
    rating: float               # 1-5 stars
    review: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    helpful_votes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "user_id": self.user_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
            "helpful_votes": self.helpful_votes,
        }


class DarkTracePlugin(ABC):
    """
    Base class for Dark Trace plugins.

    Plugins extend Dark Trace's interpretability capabilities with:
    - Custom lenses for feature extraction
    - Causal validators
    - Analysis tools
    - Visualization generators
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    def on_load(self) -> None:
        """Called when plugin is loaded. Override for initialization."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded. Override for cleanup."""
        pass

    def on_enable(self) -> None:
        """Called when plugin is enabled."""
        pass

    def on_disable(self) -> None:
        """Called when plugin is disabled."""
        pass

    def health_check(self) -> bool:
        """Return True if plugin is healthy."""
        return True

    def get_config_schema(self) -> dict[str, Any]:
        """Return JSON schema for plugin configuration."""
        return {}

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the plugin with given settings."""
        pass


class LensPlugin(DarkTracePlugin):
    """Plugin that provides custom feature extraction lenses."""

    @abstractmethod
    def extract_features(
        self,
        activations: np.ndarray,
        **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Extract features from activations.

        Args:
            activations: Model activations to analyze
            **kwargs: Additional parameters

        Returns:
            Dictionary of feature name -> feature values
        """
        pass

    def get_feature_names(self) -> list[str]:
        """Return names of features this lens extracts."""
        return []


class ValidatorPlugin(DarkTracePlugin):
    """Plugin that provides causal validation methods."""

    @abstractmethod
    def validate(
        self,
        hypothesis: str,
        activations: np.ndarray,
        intervention_fn: Callable,
        **kwargs
    ) -> dict[str, Any]:
        """
        Validate a causal hypothesis.

        Args:
            hypothesis: The hypothesis to validate
            activations: Model activations
            intervention_fn: Function to perform interventions
            **kwargs: Additional parameters

        Returns:
            Validation results with confidence scores
        """
        pass


class AnalyzerPlugin(DarkTracePlugin):
    """Plugin that provides analysis tools."""

    @abstractmethod
    def analyze(
        self,
        activations: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Analyze activations and return insights.

        Args:
            activations: Model activations
            feature_names: Optional feature names
            **kwargs: Additional parameters

        Returns:
            Analysis results
        """
        pass


class VisualizerPlugin(DarkTracePlugin):
    """Plugin that generates visualizations."""

    @abstractmethod
    def visualize(
        self,
        data: dict[str, Any],
        format: str = "html",
        **kwargs
    ) -> str:
        """
        Generate visualization from data.

        Args:
            data: Data to visualize
            format: Output format (html, svg, png, json)
            **kwargs: Additional parameters

        Returns:
            Visualization content as string
        """
        pass

    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        return ["html"]


class SafetyPlugin(DarkTracePlugin):
    """Plugin that provides safety monitoring."""

    @abstractmethod
    def check_safety(
        self,
        activations: np.ndarray,
        context: dict[str, Any] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Check activations for safety concerns.

        Args:
            activations: Model activations
            context: Optional context information
            **kwargs: Additional parameters

        Returns:
            Safety check results with concerns and severity
        """
        pass

    def get_safety_thresholds(self) -> dict[str, float]:
        """Return safety thresholds for different concern types."""
        return {}


class SteeringPlugin(DarkTracePlugin):
    """Plugin that provides steering vector generation."""

    @abstractmethod
    def generate_steering_vector(
        self,
        goal: str,
        reference_activations: np.ndarray | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a steering vector for a given goal.

        Args:
            goal: Description of steering goal
            reference_activations: Optional reference activations
            **kwargs: Additional parameters

        Returns:
            Steering vector as numpy array
        """
        pass

    def get_supported_goals(self) -> list[str]:
        """Return list of goals this plugin supports."""
        return []


class DatasetPlugin(DarkTracePlugin):
    """Plugin that provides dataset loading."""

    @abstractmethod
    def load_dataset(
        self,
        split: str = "train",
        **kwargs
    ) -> dict[str, Any]:
        """
        Load dataset for interpretability analysis.

        Args:
            split: Dataset split (train, val, test)
            **kwargs: Additional parameters

        Returns:
            Dataset dictionary with data and metadata
        """
        pass

    def get_available_splits(self) -> list[str]:
        """Return available dataset splits."""
        return ["train", "val", "test"]


class ModelAdapterPlugin(DarkTracePlugin):
    """Plugin that adapts models for interpretability analysis."""

    @abstractmethod
    def adapt(
        self,
        model: Any,
        **kwargs
    ) -> Any:
        """
        Adapt a model for interpretability analysis.

        Args:
            model: The model to adapt
            **kwargs: Additional parameters

        Returns:
            Adapted model
        """
        pass

    @abstractmethod
    def get_activations(
        self,
        adapted_model: Any,
        inputs: Any,
        layer: str | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Get activations from adapted model.

        Args:
            adapted_model: The adapted model
            inputs: Model inputs
            layer: Optional layer to extract from
            **kwargs: Additional parameters

        Returns:
            Activations as numpy array
        """
        pass


# Type for plugin factory functions
PluginFactory = Callable[[], DarkTracePlugin]


def create_plugin_metadata(
    name: str,
    version: str,
    author: str,
    description: str,
    category: PluginCategory,
    **kwargs
) -> PluginMetadata:
    """Factory function to create plugin metadata."""
    return PluginMetadata(
        name=name,
        version=version,
        author=author,
        description=description,
        category=category,
        **kwargs
    )
