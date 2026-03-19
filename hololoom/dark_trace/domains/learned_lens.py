"""
Learned Lens: Data-Driven Semantic Axis Discovery

This module provides methods for discovering interpretable semantic axes
from activation data, without requiring manual axis specification.

Key Capabilities:
- PCA-based axis discovery (captures variance directions)
- Contrastive axis discovery (learns from positive/negative pairs)
- Clustering-based discovery (finds natural groupings)
- Auto-labeling via LLM (describes what axes represent)

Research Basis:
- Concept bottleneck models (interpretable intermediate representations)
- Contrastive learning for semantic features
- Anthropic's methodology for feature labeling

Usage:
    # PCA-based discovery
    discovery = PCAAxisDiscovery(n_components=50)
    discovery.fit(activations)
    axes = discovery.get_axes()

    # Contrastive discovery
    pairs = [
        ContrastivePair(positive=pos_examples, negative=neg_examples, label="sentiment")
    ]
    discovery = ContrastiveAxisDiscovery(pairs)
    discovery.fit(activations)

    # Auto-labeling
    labeler = AutoLabeler(llm_client=client)
    for axis in axes:
        result = labeler.label(axis, top_examples)
        print(f"{axis.name}: {result.label} (confidence: {result.confidence})")
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DiscoveryMethod(Enum):
    """Method used for axis discovery."""
    PCA = "pca"
    CONTRASTIVE = "contrastive"
    CLUSTERING = "clustering"
    HYBRID = "hybrid"


@dataclass
class AxisDiscoveryConfig:
    """Configuration for axis discovery."""

    n_components: int = 50
    min_explained_variance: float = 0.01
    max_components: int | None = None
    normalize: bool = True
    center: bool = True
    random_state: int = 42

    # Contrastive learning settings
    margin: float = 1.0
    learning_rate: float = 0.01
    n_epochs: int = 100

    # Clustering settings
    n_clusters: int | None = None
    min_cluster_size: int = 5
    cluster_method: str = "kmeans"  # kmeans, hierarchical, dbscan

    @classmethod
    def default(cls) -> "AxisDiscoveryConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def high_resolution(cls) -> "AxisDiscoveryConfig":
        """High-resolution discovery (more axes, finer granularity)."""
        return cls(
            n_components=100,
            min_explained_variance=0.005,
            n_clusters=50,
        )

    @classmethod
    def fast(cls) -> "AxisDiscoveryConfig":
        """Fast discovery (fewer axes, quicker computation)."""
        return cls(
            n_components=20,
            min_explained_variance=0.05,
            n_epochs=50,
        )


@dataclass
class DiscoveredAxis:
    """A discovered semantic axis."""

    index: int
    direction: np.ndarray  # Unit vector in activation space
    explained_variance: float = 0.0
    method: DiscoveryMethod = DiscoveryMethod.PCA

    # Labeling (filled by AutoLabeler)
    label: str | None = None
    description: str | None = None
    confidence: float = 0.0

    # Examples that activate this axis
    top_positive_examples: list[Any] = field(default_factory=list)
    top_negative_examples: list[Any] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Human-readable name for the axis."""
        if self.label:
            return self.label
        return f"axis_{self.index}"

    def project(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto this axis."""
        if activations.ndim == 1:
            return float(np.dot(activations, self.direction))
        return np.dot(activations, self.direction)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "explained_variance": self.explained_variance,
            "method": self.method.value,
            "confidence": self.confidence,
            "direction_norm": float(np.linalg.norm(self.direction)),
            "metadata": self.metadata,
        }


@dataclass
class ContrastivePair:
    """A contrastive pair for axis discovery."""

    positive: list[np.ndarray]  # Examples that should activate positively
    negative: list[np.ndarray]  # Examples that should activate negatively
    label: str  # Label for this axis
    weight: float = 1.0  # Importance weight


class LearnedLens(ABC):
    """Abstract base class for learned semantic lenses."""

    @abstractmethod
    def fit(self, activations: np.ndarray, **kwargs) -> None:
        """Fit the lens to activation data."""
        pass

    @abstractmethod
    def get_axes(self) -> list[DiscoveredAxis]:
        """Get discovered axes."""
        pass

    @abstractmethod
    def project(self, activations: np.ndarray) -> dict[str, float]:
        """Project activations onto discovered semantic space."""
        pass

    def transform(self, activations: np.ndarray) -> np.ndarray:
        """Transform activations to semantic coordinates."""
        axes = self.get_axes()
        if not axes:
            return np.zeros((activations.shape[0] if activations.ndim > 1 else 1, 0))

        directions = np.stack([ax.direction for ax in axes], axis=0)
        if activations.ndim == 1:
            return np.dot(directions, activations)
        return np.dot(activations, directions.T)


class PCAAxisDiscovery(LearnedLens):
    """
    PCA-based semantic axis discovery.

    Finds principal directions of variance in activation space.
    These directions often correspond to interpretable semantic concepts.
    """

    def __init__(self, config: AxisDiscoveryConfig | None = None):
        self.config = config or AxisDiscoveryConfig.default()
        self._components: np.ndarray | None = None
        self._explained_variance: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._axes: list[DiscoveredAxis] = []
        self._fitted = False

    def fit(self, activations: np.ndarray, **kwargs) -> None:
        """
        Fit PCA to activation data.

        Args:
            activations: (n_samples, n_features) activation matrix
        """
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        n_samples, n_features = activations.shape

        # Center data
        if self.config.center:
            self._mean = np.mean(activations, axis=0)
            centered = activations - self._mean
        else:
            self._mean = np.zeros(n_features)
            centered = activations

        # Normalize
        if self.config.normalize:
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1
            centered = centered / norms

        # Compute covariance and eigendecomposition
        cov = np.cov(centered.T)
        if cov.ndim == 0:  # Single feature case
            cov = cov.reshape(1, 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvalues to explained variance ratio
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained_variance = eigenvalues / total_var
        else:
            explained_variance = np.zeros_like(eigenvalues)

        # Select components
        n_components = min(self.config.n_components, n_features, n_samples)
        if self.config.max_components:
            n_components = min(n_components, self.config.max_components)

        # Filter by minimum explained variance
        mask = explained_variance[:n_components] >= self.config.min_explained_variance
        n_keep = max(1, np.sum(mask))  # Keep at least 1

        self._components = eigenvectors[:, :n_keep].T
        self._explained_variance = explained_variance[:n_keep]

        # Create axis objects
        self._axes = []
        for i in range(n_keep):
            axis = DiscoveredAxis(
                index=i,
                direction=self._components[i],
                explained_variance=float(self._explained_variance[i]),
                method=DiscoveryMethod.PCA,
                metadata={
                    "eigenvalue": float(eigenvalues[i]),
                    "cumulative_variance": float(np.sum(explained_variance[:i+1])),
                }
            )
            self._axes.append(axis)

        self._fitted = True

    def get_axes(self) -> list[DiscoveredAxis]:
        """Get discovered axes."""
        if not self._fitted:
            return []
        return self._axes

    def project(self, activations: np.ndarray) -> dict[str, float]:
        """Project activations onto discovered semantic space."""
        if not self._fitted or not self._axes:
            return {}

        if self.config.center and self._mean is not None:
            centered = activations - self._mean
        else:
            centered = activations

        projection = {}
        for axis in self._axes:
            value = axis.project(centered)
            projection[axis.name] = float(value) if np.isscalar(value) else float(value.mean())

        return projection

    @property
    def n_components(self) -> int:
        """Number of discovered components."""
        return len(self._axes)

    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if self._explained_variance is None:
            return np.array([])
        return self._explained_variance


class ContrastiveAxisDiscovery(LearnedLens):
    """
    Contrastive axis discovery from positive/negative pairs.

    Learns directions that separate positive from negative examples,
    useful when you have labeled data or known contrasts.
    """

    def __init__(
        self,
        pairs: list[ContrastivePair] | None = None,
        config: AxisDiscoveryConfig | None = None,
    ):
        self.pairs = pairs or []
        self.config = config or AxisDiscoveryConfig.default()
        self._axes: list[DiscoveredAxis] = []
        self._fitted = False

    def add_pair(self, pair: ContrastivePair) -> None:
        """Add a contrastive pair."""
        self.pairs.append(pair)

    def fit(self, activations: np.ndarray = None, **kwargs) -> None:
        """
        Fit contrastive axes from pairs.

        Can optionally use background activations for regularization.
        """
        if not self.pairs:
            self._fitted = True
            return

        self._axes = []

        for idx, pair in enumerate(self.pairs):
            # Compute mean directions
            if len(pair.positive) == 0 or len(pair.negative) == 0:
                continue

            pos_mean = np.mean(pair.positive, axis=0)
            neg_mean = np.mean(pair.negative, axis=0)

            # Direction from negative to positive
            direction = pos_mean - neg_mean
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                continue

            # Compute separation quality
            pos_scores = [np.dot(p, direction) for p in pair.positive]
            neg_scores = [np.dot(n, direction) for n in pair.negative]

            mean_diff = np.mean(pos_scores) - np.mean(neg_scores)
            pooled_std = np.sqrt(
                (np.var(pos_scores) + np.var(neg_scores)) / 2 + 1e-8
            )
            separation = mean_diff / pooled_std  # Cohen's d

            axis = DiscoveredAxis(
                index=idx,
                direction=direction,
                explained_variance=min(1.0, abs(separation) / 3.0),  # Normalize
                method=DiscoveryMethod.CONTRASTIVE,
                label=pair.label,
                confidence=min(1.0, abs(separation) / 2.0),
                metadata={
                    "separation": float(separation),
                    "n_positive": len(pair.positive),
                    "n_negative": len(pair.negative),
                    "weight": pair.weight,
                }
            )
            self._axes.append(axis)

        self._fitted = True

    def get_axes(self) -> list[DiscoveredAxis]:
        """Get discovered axes."""
        if not self._fitted:
            return []
        return self._axes

    def project(self, activations: np.ndarray) -> dict[str, float]:
        """Project activations onto discovered semantic space."""
        if not self._fitted or not self._axes:
            return {}

        projection = {}
        for axis in self._axes:
            value = axis.project(activations)
            projection[axis.name] = float(value) if np.isscalar(value) else float(value.mean())

        return projection


class ClusteringAxisDiscovery(LearnedLens):
    """
    Clustering-based axis discovery.

    Finds natural groupings in activation space and creates axes
    based on cluster centroids.
    """

    def __init__(self, config: AxisDiscoveryConfig | None = None):
        self.config = config or AxisDiscoveryConfig.default()
        self._centroids: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._axes: list[DiscoveredAxis] = []
        self._fitted = False

    def fit(self, activations: np.ndarray, **kwargs) -> None:
        """
        Fit clustering to activation data.

        Args:
            activations: (n_samples, n_features) activation matrix
        """
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        n_samples, n_features = activations.shape

        # Determine number of clusters
        n_clusters = self.config.n_clusters
        if n_clusters is None:
            n_clusters = min(
                self.config.n_components,
                n_samples // self.config.min_cluster_size
            )
        n_clusters = max(2, min(n_clusters, n_samples))

        # K-means clustering
        if self.config.cluster_method == "kmeans":
            centroids, labels = self._kmeans(activations, n_clusters)
        else:
            # Default to k-means
            centroids, labels = self._kmeans(activations, n_clusters)

        self._centroids = centroids
        self._labels = labels

        # Create axes from centroids
        # Direction is from global mean to cluster centroid
        global_mean = np.mean(activations, axis=0)
        self._axes = []

        for i, centroid in enumerate(centroids):
            direction = centroid - global_mean
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.zeros_like(direction)
                direction[i % len(direction)] = 1.0

            # Cluster size as importance
            cluster_size = np.sum(labels == i)
            explained_variance = cluster_size / n_samples

            axis = DiscoveredAxis(
                index=i,
                direction=direction,
                explained_variance=explained_variance,
                method=DiscoveryMethod.CLUSTERING,
                metadata={
                    "cluster_size": int(cluster_size),
                    "centroid_norm": float(np.linalg.norm(centroid)),
                    "distance_from_mean": float(norm),
                }
            )
            self._axes.append(axis)

        self._fitted = True

    def _kmeans(
        self, data: np.ndarray, n_clusters: int, max_iter: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple k-means implementation."""
        n_samples = data.shape[0]

        # Random initialization
        np.random.seed(self.config.random_state)
        idx = np.random.choice(n_samples, size=n_clusters, replace=False)
        centroids = data[idx].copy()

        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                distances[:, k] = np.linalg.norm(data - centroids[k], axis=1)
            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = np.mean(data[mask], axis=0)

        return centroids, labels

    def get_axes(self) -> list[DiscoveredAxis]:
        """Get discovered axes."""
        if not self._fitted:
            return []
        return self._axes

    def project(self, activations: np.ndarray) -> dict[str, float]:
        """Project activations onto discovered semantic space."""
        if not self._fitted or not self._axes:
            return {}

        projection = {}
        for axis in self._axes:
            value = axis.project(activations)
            projection[axis.name] = float(value) if np.isscalar(value) else float(value.mean())

        return projection

    def get_cluster_assignments(self) -> np.ndarray | None:
        """Get cluster assignments from last fit."""
        return self._labels


@dataclass
class LabelingResult:
    """Result from auto-labeling an axis."""

    axis_index: int
    label: str
    description: str
    confidence: float
    supporting_examples: list[str] = field(default_factory=list)
    alternative_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "axis_index": self.axis_index,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "supporting_examples": self.supporting_examples,
            "alternative_labels": self.alternative_labels,
            "metadata": self.metadata,
        }


class AutoLabeler:
    """
    Automatic axis labeling using LLM.

    Given an axis and examples that activate it, generates
    human-readable labels and descriptions.
    """

    DEFAULT_PROMPT = """You are analyzing a semantic axis discovered in a neural network.
Given the following examples that strongly activate this axis (positive) and
examples that strongly deactivate it (negative), provide a concise label and description.

Positive examples (activate this axis):
{positive_examples}

Negative examples (deactivate this axis):
{negative_examples}

Provide:
1. A short label (1-3 words) for this axis
2. A one-sentence description of what this axis represents
3. Your confidence (0-1) in this interpretation
4. 2-3 alternative labels if the axis is ambiguous

Response format (JSON):
{{
    "label": "...",
    "description": "...",
    "confidence": 0.X,
    "alternatives": ["...", "..."]
}}
"""

    def __init__(
        self,
        llm_client: Callable | None = None,
        prompt_template: str | None = None,
        max_examples: int = 5,
    ):
        """
        Initialize auto-labeler.

        Args:
            llm_client: Callable that takes a prompt and returns response string
            prompt_template: Custom prompt template
            max_examples: Maximum examples to include in prompt
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.max_examples = max_examples
        self._cache: dict[int, LabelingResult] = {}

    def label(
        self,
        axis: DiscoveredAxis,
        positive_examples: list[str] | None = None,
        negative_examples: list[str] | None = None,
    ) -> LabelingResult:
        """
        Generate label for an axis.

        Args:
            axis: The axis to label
            positive_examples: Examples that activate the axis
            negative_examples: Examples that deactivate the axis

        Returns:
            LabelingResult with label, description, and confidence
        """
        # Check cache
        if axis.index in self._cache:
            return self._cache[axis.index]

        # Use provided examples or axis's stored examples
        pos = positive_examples or axis.top_positive_examples
        neg = negative_examples or axis.top_negative_examples

        # Format examples
        pos_str = self._format_examples(pos[:self.max_examples])
        neg_str = self._format_examples(neg[:self.max_examples])

        # Fallback if no LLM client
        if self.llm_client is None:
            result = self._heuristic_label(axis, pos, neg)
            self._cache[axis.index] = result
            return result

        # Build prompt
        prompt = self.prompt_template.format(
            positive_examples=pos_str,
            negative_examples=neg_str,
        )

        try:
            response = self.llm_client(prompt)
            result = self._parse_response(axis.index, response, pos)
        except Exception as e:
            # Fallback to heuristic
            result = self._heuristic_label(axis, pos, neg)
            result.metadata["llm_error"] = str(e)

        self._cache[axis.index] = result
        return result

    def _format_examples(self, examples: list[str]) -> str:
        """Format examples for prompt."""
        if not examples:
            return "(No examples provided)"
        return "\n".join(f"- {ex}" for ex in examples)

    def _parse_response(
        self, axis_index: int, response: str, examples: list[str]
    ) -> LabelingResult:
        """Parse LLM response."""
        import json

        try:
            # Try to extract JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return LabelingResult(
                    axis_index=axis_index,
                    label=data.get("label", f"axis_{axis_index}"),
                    description=data.get("description", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    supporting_examples=examples[:3],
                    alternative_labels=data.get("alternatives", []),
                )
        except json.JSONDecodeError:
            pass

        # Fallback: use response as label
        label = response.strip()[:50]
        return LabelingResult(
            axis_index=axis_index,
            label=label or f"axis_{axis_index}",
            description="Auto-generated from LLM response",
            confidence=0.3,
            supporting_examples=examples[:3],
        )

    def _heuristic_label(
        self,
        axis: DiscoveredAxis,
        positive: list[str],
        negative: list[str],
    ) -> LabelingResult:
        """Generate heuristic label when LLM not available."""
        # Simple heuristic: use method and index
        method_name = axis.method.value

        if positive and negative:
            # Try to find contrasting words
            pos_words = set(" ".join(positive).lower().split())
            neg_words = set(" ".join(negative).lower().split())
            unique_pos = pos_words - neg_words
            unique_neg = neg_words - pos_words

            if unique_pos:
                label = f"{method_name}_{list(unique_pos)[0]}"
            else:
                label = f"{method_name}_axis_{axis.index}"
        else:
            label = f"{method_name}_axis_{axis.index}"

        description = (
            f"Axis {axis.index} discovered via {method_name} "
            f"(variance: {axis.explained_variance:.3f})"
        )

        return LabelingResult(
            axis_index=axis.index,
            label=label,
            description=description,
            confidence=0.2,  # Low confidence for heuristic
            supporting_examples=positive[:3] if positive else [],
            metadata={"method": "heuristic"},
        )

    def label_batch(
        self,
        axes: list[DiscoveredAxis],
        examples_fn: Callable[[DiscoveredAxis], tuple[list[str], list[str]]] | None = None,
    ) -> list[LabelingResult]:
        """
        Label multiple axes.

        Args:
            axes: List of axes to label
            examples_fn: Function to get (positive, negative) examples for each axis

        Returns:
            List of LabelingResults
        """
        results = []
        for axis in axes:
            if examples_fn:
                pos, neg = examples_fn(axis)
            else:
                pos, neg = axis.top_positive_examples, axis.top_negative_examples
            result = self.label(axis, pos, neg)
            results.append(result)
        return results

    def update_axes(
        self,
        axes: list[DiscoveredAxis],
        examples_fn: Callable | None = None,
    ) -> None:
        """
        Update axes with generated labels.

        Modifies axes in place with label, description, and confidence.
        """
        results = self.label_batch(axes, examples_fn)
        for axis, result in zip(axes, results):
            axis.label = result.label
            axis.description = result.description
            axis.confidence = result.confidence

    def clear_cache(self) -> None:
        """Clear the labeling cache."""
        self._cache.clear()
