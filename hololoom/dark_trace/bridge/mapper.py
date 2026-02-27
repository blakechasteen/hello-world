"""
Dark Trace Bridge: Feature ↔ Semantic Bidirectional Mapping

Provides bidirectional mapping between learned SAE features and
interpretable semantic axes.

Key Features:
- SAE feature → top-k semantic meanings
- Semantic axis → top-k SAE features
- Unmapped feature discovery (novel semantic dimensions)
- Composite semantic signatures

Research Basis:
- SAE features often correlate with human-interpretable concepts
- Bridging enables using semantic steering via SAE features
- Unmapped features may represent novel dimensions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import numpy as np
import torch

from hololoom.dark_trace.protocol import (
    TraceLens,
    Feature,
    FeatureSource,
    LensType,
)


@dataclass
class SemanticMeaning:
    """
    Semantic meaning associated with a feature.

    Represents the interpretable semantic components of an SAE feature,
    such as "Warmth (0.85)" or "Formality (-0.62)".
    """

    # Semantic dimension name (e.g., "Warmth", "Formality")
    dimension: str

    # Correlation strength (-1 to +1)
    # Positive = feature correlates with positive pole
    # Negative = feature correlates with negative pole
    correlation: float

    # Statistical confidence in correlation (0 to 1)
    confidence: float = 0.0

    # Number of samples used to compute correlation
    sample_count: int = 0

    # Exemplar words for positive/negative poles
    positive_exemplars: List[str] = field(default_factory=list)
    negative_exemplars: List[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        """Get human-readable direction (positive or negative pole)."""
        if self.correlation > 0:
            return "positive"
        elif self.correlation < 0:
            return "negative"
        else:
            return "neutral"

    @property
    def pole(self) -> str:
        """Get the pole this feature aligns with."""
        if self.correlation > 0 and self.positive_exemplars:
            return self.positive_exemplars[0]
        elif self.correlation < 0 and self.negative_exemplars:
            return self.negative_exemplars[0]
        return "neutral"

    @property
    def strength(self) -> float:
        """Absolute correlation strength."""
        return abs(self.correlation)

    def describe(self) -> str:
        """Human-readable description."""
        sign = "+" if self.correlation > 0 else ""
        conf_str = f" [{self.confidence:.0%}]" if self.confidence > 0 else ""
        return f"{self.dimension}: {sign}{self.correlation:.3f}{conf_str} → '{self.pole}'"


@dataclass
class FeatureMapping:
    """
    Complete mapping for a single SAE feature.

    Contains all semantic meanings associated with a feature,
    along with metadata about the mapping quality.
    """

    # Feature identifier (e.g., "sae.42")
    feature_id: str

    # Feature index in SAE latent space
    feature_index: int

    # Top-k semantic meanings sorted by |correlation|
    semantic_meanings: List[SemanticMeaning] = field(default_factory=list)

    # Activation statistics
    mean_activation: float = 0.0
    max_activation: float = 0.0
    activation_frequency: float = 0.0  # Fraction of samples where active

    # Mapping quality metrics
    total_variance_explained: float = 0.0  # Sum of squared correlations
    is_well_mapped: bool = False  # Has at least one strong correlation

    # Human-assigned label (if available from labeler)
    human_label: Optional[str] = None

    @property
    def top_meaning(self) -> Optional[SemanticMeaning]:
        """Get the strongest semantic meaning."""
        if self.semantic_meanings:
            return max(self.semantic_meanings, key=lambda m: m.strength)
        return None

    @property
    def semantic_signature(self) -> str:
        """
        Compact signature of semantic components.

        Example: "Warmth(+0.85) + Formality(-0.62)"
        """
        if not self.semantic_meanings:
            return "unmapped"

        parts = []
        for meaning in self.semantic_meanings[:3]:  # Top 3
            sign = "+" if meaning.correlation > 0 else ""
            parts.append(f"{meaning.dimension}({sign}{meaning.correlation:.2f})")

        return " + ".join(parts)

    def describe(self, verbose: bool = False) -> str:
        """Human-readable description of feature mapping."""
        lines = [f"Feature {self.feature_id}:"]

        if self.human_label:
            lines.append(f"  Label: {self.human_label}")

        lines.append(f"  Semantic Signature: {self.semantic_signature}")

        if verbose and self.semantic_meanings:
            lines.append("  Meanings:")
            for meaning in self.semantic_meanings[:5]:
                lines.append(f"    • {meaning.describe()}")

        return "\n".join(lines)


class FeatureSemanticMapper:
    """
    Bidirectional mapper between SAE features and semantic axes.

    Enables:
    - Understanding what an SAE feature represents semantically
    - Finding which SAE features activate for a semantic goal
    - Discovering novel semantic dimensions (unmapped features)

    Usage:
        # Create mapper from correlator
        mapper = FeatureSemanticMapper(correlator)

        # Get semantic meaning of SAE feature
        mapping = mapper.get_feature_mapping("sae.42")
        print(mapping.semantic_signature)
        # Output: "Warmth(+0.85) + Formality(-0.62)"

        # Find features for semantic goal
        features = mapper.get_features_for_semantic("Warmth", k=5)
        # Returns: [("sae.42", 0.85), ("sae.156", 0.71), ...]

        # Discover unmapped features (potential novel dimensions)
        novel = mapper.get_unmapped_features(min_activation=0.1)
    """

    def __init__(
        self,
        correlator: Any,  # SAESemanticCorrelator
        min_correlation: float = 0.1,
        min_confidence: float = 0.5,
        well_mapped_threshold: float = 0.5,
    ):
        """
        Initialize mapper from correlation data.

        Args:
            correlator: SAESemanticCorrelator with computed correlations
            min_correlation: Minimum |correlation| to include in mapping
            min_confidence: Minimum confidence to trust correlation
            well_mapped_threshold: Minimum |correlation| for "well-mapped" status
        """
        self.correlator = correlator
        self.min_correlation = min_correlation
        self.min_confidence = min_confidence
        self.well_mapped_threshold = well_mapped_threshold

        # Cache for computed mappings
        self._feature_mappings: Dict[str, FeatureMapping] = {}
        self._semantic_to_features: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Track unmapped features
        self._unmapped_features: Set[str] = set()

        # Build mappings from correlator data
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build bidirectional mappings from correlator data."""
        # Get correlation results
        result = self.correlator.compute_correlations()

        # Build feature → semantic mappings
        for sae_id in result.correlations:
            if not sae_id.startswith("sae."):
                continue

            feature_idx = int(sae_id.replace("sae.", ""))
            semantic_meanings = []

            for sem_id, corr in result.correlations[sae_id].items():
                if not sem_id.startswith("semantic."):
                    continue

                if abs(corr) < self.min_correlation:
                    continue

                # Extract semantic dimension name
                dim_name = sem_id.replace("semantic.", "")

                # Get confidence from correlator if available
                confidence = 0.0
                if hasattr(self.correlator, '_correlation_matrix'):
                    corr_data = self.correlator._correlation_matrix._correlations.get(sae_id, {}).get(sem_id)
                    if corr_data:
                        confidence = corr_data.confidence

                meaning = SemanticMeaning(
                    dimension=dim_name,
                    correlation=corr,
                    confidence=confidence,
                    sample_count=result.sample_count,
                )
                semantic_meanings.append(meaning)

                # Build reverse mapping (semantic → SAE features)
                self._semantic_to_features[dim_name].append((sae_id, corr))

            # Sort by absolute correlation
            semantic_meanings.sort(key=lambda m: m.strength, reverse=True)

            # Create feature mapping
            total_variance = sum(m.correlation ** 2 for m in semantic_meanings)
            is_well_mapped = any(m.strength >= self.well_mapped_threshold for m in semantic_meanings)

            mapping = FeatureMapping(
                feature_id=sae_id,
                feature_index=feature_idx,
                semantic_meanings=semantic_meanings,
                total_variance_explained=total_variance,
                is_well_mapped=is_well_mapped,
            )

            self._feature_mappings[sae_id] = mapping

            # Track unmapped features
            if not is_well_mapped:
                self._unmapped_features.add(sae_id)

        # Sort semantic → features by correlation
        for dim_name in self._semantic_to_features:
            self._semantic_to_features[dim_name].sort(
                key=lambda x: abs(x[1]), reverse=True
            )

    def get_feature_mapping(self, feature_id: str) -> Optional[FeatureMapping]:
        """
        Get semantic mapping for an SAE feature.

        Args:
            feature_id: Feature ID (e.g., "sae.42")

        Returns:
            FeatureMapping with semantic meanings, or None if not found
        """
        return self._feature_mappings.get(feature_id)

    def get_semantic_meanings(
        self,
        feature_id: str,
        k: int = 5,
    ) -> List[SemanticMeaning]:
        """
        Get top-k semantic meanings for a feature.

        Args:
            feature_id: Feature ID
            k: Number of meanings to return

        Returns:
            List of SemanticMeaning sorted by |correlation|
        """
        mapping = self.get_feature_mapping(feature_id)
        if mapping is None:
            return []
        return mapping.semantic_meanings[:k]

    def get_features_for_semantic(
        self,
        semantic_dimension: str,
        k: int = 5,
        positive_only: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Get SAE features most correlated with a semantic dimension.

        Args:
            semantic_dimension: Dimension name (e.g., "Warmth")
            k: Number of features to return
            positive_only: If True, only return positively correlated features

        Returns:
            List of (feature_id, correlation) tuples
        """
        features = self._semantic_to_features.get(semantic_dimension, [])

        if positive_only:
            features = [(f, c) for f, c in features if c > 0]

        return features[:k]

    def get_unmapped_features(
        self,
        min_activation: float = 0.1,
    ) -> List[str]:
        """
        Get features with no strong semantic correlations.

        These may represent novel semantic dimensions not captured
        by the existing 244 axes.

        Args:
            min_activation: Minimum activation frequency to consider

        Returns:
            List of unmapped feature IDs
        """
        return list(self._unmapped_features)

    def get_composite_mapping(
        self,
        feature_ids: List[str],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Get composite semantic mapping for multiple features.

        Useful for understanding what a combination of features represents.

        Args:
            feature_ids: List of feature IDs
            weights: Optional weights (default: uniform)

        Returns:
            Dict mapping dimension name → weighted average correlation
        """
        if not feature_ids:
            return {}

        if weights is None:
            weights = [1.0] * len(feature_ids)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Accumulate weighted correlations
        composite: Dict[str, float] = defaultdict(float)
        contribution_count: Dict[str, int] = defaultdict(int)

        for feature_id, weight in zip(feature_ids, weights):
            mapping = self.get_feature_mapping(feature_id)
            if mapping is None:
                continue

            for meaning in mapping.semantic_meanings:
                composite[meaning.dimension] += meaning.correlation * weight
                contribution_count[meaning.dimension] += 1

        # Sort by absolute value
        result = dict(
            sorted(composite.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return result

    def find_similar_features(
        self,
        feature_id: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find features with similar semantic profiles.

        Uses cosine similarity over semantic correlation vectors.

        Args:
            feature_id: Reference feature ID
            k: Number of similar features to return

        Returns:
            List of (feature_id, similarity) tuples
        """
        ref_mapping = self.get_feature_mapping(feature_id)
        if ref_mapping is None:
            return []

        # Build reference vector
        ref_dims = {m.dimension: m.correlation for m in ref_mapping.semantic_meanings}
        if not ref_dims:
            return []

        similarities = []

        for other_id, other_mapping in self._feature_mappings.items():
            if other_id == feature_id:
                continue

            # Build comparison vector
            other_dims = {m.dimension: m.correlation for m in other_mapping.semantic_meanings}
            if not other_dims:
                continue

            # Compute cosine similarity
            common_dims = set(ref_dims.keys()) & set(other_dims.keys())
            if not common_dims:
                continue

            ref_vec = [ref_dims[d] for d in common_dims]
            other_vec = [other_dims[d] for d in common_dims]

            dot = sum(r * o for r, o in zip(ref_vec, other_vec))
            ref_norm = np.sqrt(sum(r ** 2 for r in ref_vec))
            other_norm = np.sqrt(sum(o ** 2 for o in other_vec))

            if ref_norm > 0 and other_norm > 0:
                similarity = dot / (ref_norm * other_norm)
                similarities.append((other_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def describe_feature(
        self,
        feature_id: str,
        verbose: bool = False,
    ) -> str:
        """
        Generate human-readable description of a feature.

        Args:
            feature_id: Feature ID
            verbose: If True, include more detail

        Returns:
            Human-readable description
        """
        mapping = self.get_feature_mapping(feature_id)
        if mapping is None:
            return f"Feature {feature_id}: No mapping available"

        return mapping.describe(verbose=verbose)

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the mapping quality.

        Returns:
            Dict with mapping statistics
        """
        n_features = len(self._feature_mappings)
        n_well_mapped = sum(1 for m in self._feature_mappings.values() if m.is_well_mapped)
        n_unmapped = len(self._unmapped_features)

        # Average meanings per feature
        meanings_counts = [
            len(m.semantic_meanings) for m in self._feature_mappings.values()
        ]
        avg_meanings = np.mean(meanings_counts) if meanings_counts else 0

        # Semantic coverage
        covered_dims = set()
        for mapping in self._feature_mappings.values():
            for meaning in mapping.semantic_meanings:
                covered_dims.add(meaning.dimension)

        return {
            "total_features": n_features,
            "well_mapped_features": n_well_mapped,
            "unmapped_features": n_unmapped,
            "mapping_rate": n_well_mapped / n_features if n_features > 0 else 0,
            "avg_meanings_per_feature": avg_meanings,
            "semantic_dimensions_covered": len(covered_dims),
            "unique_dimensions": list(covered_dims)[:20],
        }

    def save(self, path: str) -> None:
        """Save mappings to file."""
        import json

        data = {
            "feature_mappings": {
                fid: {
                    "feature_index": m.feature_index,
                    "semantic_signature": m.semantic_signature,
                    "is_well_mapped": m.is_well_mapped,
                    "total_variance_explained": m.total_variance_explained,
                    "human_label": m.human_label,
                    "meanings": [
                        {
                            "dimension": sm.dimension,
                            "correlation": sm.correlation,
                            "confidence": sm.confidence,
                        }
                        for sm in m.semantic_meanings[:10]
                    ],
                }
                for fid, m in self._feature_mappings.items()
            },
            "statistics": self.get_mapping_statistics(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load mappings from file."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        # Rebuild mappings from saved data
        self._feature_mappings = {}
        self._semantic_to_features = defaultdict(list)
        self._unmapped_features = set()

        for fid, m_data in data.get("feature_mappings", {}).items():
            meanings = [
                SemanticMeaning(
                    dimension=sm["dimension"],
                    correlation=sm["correlation"],
                    confidence=sm.get("confidence", 0.0),
                )
                for sm in m_data.get("meanings", [])
            ]

            mapping = FeatureMapping(
                feature_id=fid,
                feature_index=m_data.get("feature_index", 0),
                semantic_meanings=meanings,
                total_variance_explained=m_data.get("total_variance_explained", 0.0),
                is_well_mapped=m_data.get("is_well_mapped", False),
                human_label=m_data.get("human_label"),
            )

            self._feature_mappings[fid] = mapping

            if not mapping.is_well_mapped:
                self._unmapped_features.add(fid)

            # Rebuild reverse mapping
            for meaning in meanings:
                self._semantic_to_features[meaning.dimension].append(
                    (fid, meaning.correlation)
                )

        # Sort reverse mappings
        for dim in self._semantic_to_features:
            self._semantic_to_features[dim].sort(
                key=lambda x: abs(x[1]), reverse=True
            )
