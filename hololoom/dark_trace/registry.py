"""
Dark Trace Registry: Unified Feature Namespace

The FeatureRegistry provides a single namespace for all features across
interpretability lenses (SAE, Semantic, Domain). It enables:

- Unified feature lookup: "sae.42", "semantic.Warmth", "domain.medical.Severity"
- SAE ↔ Semantic correlation tracking
- Feature metadata management (labels, safety flags, statistics)
- Activation history for pattern analysis

Key Classes:
- FeatureRegistry: Central registry for all features
- CorrelationMatrix: SAE ↔ Semantic correlations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Iterator
from collections import defaultdict
from datetime import datetime
import json
import threading
import numpy as np

from hololoom.dark_trace.protocol import (
    Feature,
    FeatureSource,
    SafetyFlag,
    LensType,
)


# =============================================================================
# Correlation Matrix
# =============================================================================

@dataclass
class FeatureCorrelation:
    """Correlation between two features from different lenses."""
    source_id: str           # e.g., "sae.42"
    target_id: str           # e.g., "semantic.Warmth"
    correlation: float       # Pearson correlation coefficient
    sample_count: int        # Number of samples used
    confidence: float = 0.0  # Statistical confidence (1 - p_value)

    def is_significant(self, threshold: float = 0.95) -> bool:
        """Check if correlation is statistically significant."""
        return self.confidence >= threshold


class CorrelationMatrix:
    """
    Tracks correlations between SAE features and Semantic axes.

    This is a key component for bridging learned features (SAE)
    with interpretable dimensions (Semantic Axes).
    """

    def __init__(self):
        # {source_id: {target_id: FeatureCorrelation}}
        self._correlations: Dict[str, Dict[str, FeatureCorrelation]] = defaultdict(dict)

        # Running statistics for online correlation computation
        # {(source_id, target_id): {"n": int, "sum_x": float, "sum_y": float, ...}}
        self._running_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

        self._lock = threading.Lock()

    def update(
        self,
        source_activations: Dict[str, float],
        target_activations: Dict[str, float]
    ) -> None:
        """
        Update correlation estimates with new activation observations.

        Uses Welford's online algorithm for numerical stability.

        Args:
            source_activations: {source_feature_id: activation}
            target_activations: {target_feature_id: activation}
        """
        with self._lock:
            for src_id, src_val in source_activations.items():
                for tgt_id, tgt_val in target_activations.items():
                    key = (src_id, tgt_id)

                    if key not in self._running_stats:
                        self._running_stats[key] = {
                            "n": 0,
                            "mean_x": 0.0,
                            "mean_y": 0.0,
                            "m2_x": 0.0,
                            "m2_y": 0.0,
                            "m2_xy": 0.0,
                        }

                    stats = self._running_stats[key]
                    stats["n"] += 1
                    n = stats["n"]

                    # Update means (Welford)
                    delta_x = src_val - stats["mean_x"]
                    stats["mean_x"] += delta_x / n
                    delta_x2 = src_val - stats["mean_x"]

                    delta_y = tgt_val - stats["mean_y"]
                    stats["mean_y"] += delta_y / n
                    delta_y2 = tgt_val - stats["mean_y"]

                    # Update variances and covariance
                    stats["m2_x"] += delta_x * delta_x2
                    stats["m2_y"] += delta_y * delta_y2
                    stats["m2_xy"] += delta_x * delta_y2

    def compute_correlations(self, min_samples: int = 30) -> None:
        """
        Compute correlations from running statistics.

        Args:
            min_samples: Minimum samples required for correlation
        """
        with self._lock:
            for (src_id, tgt_id), stats in self._running_stats.items():
                n = stats["n"]
                if n < min_samples:
                    continue

                var_x = stats["m2_x"] / (n - 1) if n > 1 else 0
                var_y = stats["m2_y"] / (n - 1) if n > 1 else 0
                cov_xy = stats["m2_xy"] / (n - 1) if n > 1 else 0

                if var_x > 0 and var_y > 0:
                    correlation = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
                    correlation = max(-1.0, min(1.0, correlation))  # Clamp

                    # Approximate confidence (Fisher's z-transformation)
                    if abs(correlation) < 0.9999:
                        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
                        se = 1 / np.sqrt(n - 3) if n > 3 else 1.0
                        confidence = 1.0 - 2 * (1 - _normal_cdf(abs(z) / se))
                    else:
                        confidence = 1.0

                    self._correlations[src_id][tgt_id] = FeatureCorrelation(
                        source_id=src_id,
                        target_id=tgt_id,
                        correlation=correlation,
                        sample_count=n,
                        confidence=confidence,
                    )

    def get_correlation(self, source_id: str, target_id: str) -> Optional[FeatureCorrelation]:
        """Get correlation between two features."""
        return self._correlations.get(source_id, {}).get(target_id)

    def get_top_correlations(
        self,
        source_id: str,
        k: int = 5,
        min_confidence: float = 0.9
    ) -> List[FeatureCorrelation]:
        """
        Get top-k correlated features for a source feature.

        Args:
            source_id: Source feature ID
            k: Number of top correlations to return
            min_confidence: Minimum confidence threshold
        """
        if source_id not in self._correlations:
            return []

        correlations = [
            c for c in self._correlations[source_id].values()
            if c.confidence >= min_confidence
        ]

        # Sort by absolute correlation
        correlations.sort(key=lambda c: abs(c.correlation), reverse=True)
        return correlations[:k]

    def get_semantic_meanings(self, sae_feature_id: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get semantic meanings for an SAE feature.

        Returns list of (semantic_axis_name, correlation) tuples.
        """
        correlations = self.get_top_correlations(sae_feature_id, k=k)
        return [
            (c.target_id.replace("semantic.", ""), c.correlation)
            for c in correlations
            if c.target_id.startswith("semantic.")
        ]

    def get_sae_features_for_semantic(self, semantic_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get SAE features most correlated with a semantic axis.

        Returns list of (sae_feature_id, correlation) tuples.
        """
        results = []
        full_id = f"semantic.{semantic_id}" if not semantic_id.startswith("semantic.") else semantic_id

        for src_id, targets in self._correlations.items():
            if src_id.startswith("sae.") and full_id in targets:
                corr = targets[full_id]
                if corr.is_significant():
                    results.append((src_id, corr.correlation))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize correlations to dictionary."""
        return {
            src: {
                tgt: {"correlation": c.correlation, "samples": c.sample_count, "confidence": c.confidence}
                for tgt, c in targets.items()
            }
            for src, targets in self._correlations.items()
        }


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF for confidence calculation."""
    # Simple approximation using error function
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =============================================================================
# Feature Registry
# =============================================================================

class FeatureRegistry:
    """
    Central registry for all interpretability features.

    Provides:
    - Unified namespace: "sae.42", "semantic.Warmth", "domain.medical.Severity"
    - Feature metadata management
    - Activation history tracking
    - SAE ↔ Semantic correlation bridge
    """

    def __init__(self, enable_history: bool = True, history_size: int = 1000):
        """
        Initialize the feature registry.

        Args:
            enable_history: Track activation history
            history_size: Maximum history entries per feature
        """
        # Main storage: {feature_id: Feature}
        self._features: Dict[str, Feature] = {}

        # Index by source type
        self._by_source: Dict[FeatureSource, Set[str]] = defaultdict(set)

        # Index by safety flag
        self._by_safety: Dict[SafetyFlag, Set[str]] = defaultdict(set)

        # Activation history: {feature_id: [(timestamp, activation), ...]}
        self._enable_history = enable_history
        self._history_size = history_size
        self._activation_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Statistics
        self._activation_counts: Dict[str, int] = defaultdict(int)
        self._activation_sums: Dict[str, float] = defaultdict(float)
        self._activation_sq_sums: Dict[str, float] = defaultdict(float)

        # SAE ↔ Semantic correlations
        self.correlations = CorrelationMatrix()

        # Thread safety
        self._lock = threading.Lock()

    def register(self, feature: Feature) -> None:
        """
        Register a feature in the registry.

        Args:
            feature: Feature to register
        """
        with self._lock:
            self._features[feature.id] = feature
            self._by_source[feature.source].add(feature.id)
            self._by_safety[feature.safety_flag].add(feature.id)

    def register_batch(self, features: List[Feature]) -> None:
        """Register multiple features at once."""
        with self._lock:
            for feature in features:
                self._features[feature.id] = feature
                self._by_source[feature.source].add(feature.id)
                self._by_safety[feature.safety_flag].add(feature.id)

    def get(self, feature_id: str) -> Optional[Feature]:
        """Get a feature by ID."""
        return self._features.get(feature_id)

    def __getitem__(self, feature_id: str) -> Feature:
        """Get a feature by ID (raises KeyError if not found)."""
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")
        return self._features[feature_id]

    def __contains__(self, feature_id: str) -> bool:
        """Check if feature exists."""
        return feature_id in self._features

    def __len__(self) -> int:
        """Number of registered features."""
        return len(self._features)

    def __iter__(self) -> Iterator[Feature]:
        """Iterate over all features."""
        return iter(self._features.values())

    def list_by_source(self, source: FeatureSource) -> List[Feature]:
        """Get all features from a specific source."""
        return [self._features[fid] for fid in self._by_source.get(source, set())]

    def list_by_safety(self, safety_flag: SafetyFlag) -> List[Feature]:
        """Get all features with a specific safety flag."""
        return [self._features[fid] for fid in self._by_safety.get(safety_flag, set())]

    def get_dangerous_features(self) -> List[Feature]:
        """Get all features flagged as dangerous."""
        dangerous = set()
        for flag in [SafetyFlag.DANGEROUS, SafetyFlag.BLOCKED]:
            dangerous.update(self._by_safety.get(flag, set()))
        return [self._features[fid] for fid in dangerous]

    def update_safety_flag(self, feature_id: str, flag: SafetyFlag) -> None:
        """Update the safety flag for a feature."""
        if feature_id not in self._features:
            return

        with self._lock:
            feature = self._features[feature_id]
            old_flag = feature.safety_flag

            # Update indices
            self._by_safety[old_flag].discard(feature_id)
            self._by_safety[flag].add(feature_id)

            # Update feature (create new with updated flag due to dataclass)
            updated = Feature(
                id=feature.id,
                source=feature.source,
                index=feature.index,
                label=feature.label,
                description=feature.description,
                exemplars=feature.exemplars,
                activation_mean=feature.activation_mean,
                activation_std=feature.activation_std,
                sparsity=feature.sparsity,
                safety_flag=flag,
                metadata=feature.metadata,
            )
            self._features[feature_id] = updated

    def record_activation(self, feature_id: str, activation: float) -> None:
        """
        Record an activation for a feature.

        Updates statistics and optionally history.
        """
        with self._lock:
            # Update counts and sums for running stats
            self._activation_counts[feature_id] += 1
            self._activation_sums[feature_id] += activation
            self._activation_sq_sums[feature_id] += activation * activation

            # Update history if enabled
            if self._enable_history:
                history = self._activation_history[feature_id]
                history.append((datetime.now(), activation))

                # Trim history if too long
                if len(history) > self._history_size:
                    self._activation_history[feature_id] = history[-self._history_size:]

    def record_activations_batch(self, activations: Dict[str, float]) -> None:
        """Record multiple activations at once."""
        with self._lock:
            now = datetime.now()
            for feature_id, activation in activations.items():
                self._activation_counts[feature_id] += 1
                self._activation_sums[feature_id] += activation
                self._activation_sq_sums[feature_id] += activation * activation

                if self._enable_history:
                    history = self._activation_history[feature_id]
                    history.append((now, activation))
                    if len(history) > self._history_size:
                        self._activation_history[feature_id] = history[-self._history_size:]

    def get_activation_stats(self, feature_id: str) -> Dict[str, float]:
        """
        Get activation statistics for a feature.

        Returns:
            {"count": int, "mean": float, "std": float}
        """
        count = self._activation_counts.get(feature_id, 0)
        if count == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0}

        mean = self._activation_sums[feature_id] / count
        variance = (self._activation_sq_sums[feature_id] / count) - (mean * mean)
        std = np.sqrt(max(0, variance))

        return {"count": count, "mean": mean, "std": std}

    def get_history(
        self,
        feature_id: str,
        limit: Optional[int] = None
    ) -> List[Tuple[datetime, float]]:
        """Get activation history for a feature."""
        history = self._activation_history.get(feature_id, [])
        if limit:
            return history[-limit:]
        return history

    def update_correlations(
        self,
        sae_activations: Dict[str, float],
        semantic_activations: Dict[str, float]
    ) -> None:
        """
        Update SAE ↔ Semantic correlations with new observations.

        Call this after each analysis to build correlation matrix.
        """
        self.correlations.update(sae_activations, semantic_activations)

    def compute_correlations(self, min_samples: int = 30) -> None:
        """Compute correlation matrix from accumulated observations."""
        self.correlations.compute_correlations(min_samples)

    def explain_sae_feature(self, sae_feature_id: str, k: int = 3) -> str:
        """
        Generate semantic explanation for an SAE feature.

        Uses correlation matrix to find semantic meanings.
        """
        feature = self.get(sae_feature_id)
        if not feature:
            return f"Unknown feature: {sae_feature_id}"

        meanings = self.correlations.get_semantic_meanings(sae_feature_id, k=k)

        if not meanings:
            label = feature.label or sae_feature_id
            return f"{label}: No semantic correlations found yet"

        parts = [f"{name} ({corr:+.2f})" for name, corr in meanings]
        return f"SAE Feature {feature.index} ≈ {' + '.join(parts)}"

    def search(
        self,
        query: str,
        source: Optional[FeatureSource] = None,
        limit: int = 10
    ) -> List[Feature]:
        """
        Search features by label or description.

        Args:
            query: Search query (case-insensitive)
            source: Filter by source type
            limit: Maximum results
        """
        query_lower = query.lower()
        results = []

        candidates = self._features.values()
        if source:
            candidate_ids = self._by_source.get(source, set())
            candidates = [self._features[fid] for fid in candidate_ids]

        for feature in candidates:
            score = 0
            if feature.label and query_lower in feature.label.lower():
                score = 2
            elif feature.description and query_lower in feature.description.lower():
                score = 1

            if score > 0:
                results.append((score, feature))

        results.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in results[:limit]]

    def summary(self) -> str:
        """Generate registry summary."""
        lines = [
            "Dark Trace Feature Registry",
            f"Total Features: {len(self._features)}",
            "",
            "By Source:",
        ]

        for source in FeatureSource:
            count = len(self._by_source.get(source, set()))
            lines.append(f"  {source.value}: {count}")

        lines.append("\nBy Safety Flag:")
        for flag in SafetyFlag:
            count = len(self._by_safety.get(flag, set()))
            if count > 0:
                lines.append(f"  {flag.name}: {count}")

        # Top activated features
        if self._activation_counts:
            lines.append("\nMost Active Features:")
            sorted_counts = sorted(
                self._activation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for fid, count in sorted_counts:
                feature = self._features.get(fid)
                label = feature.label if feature else fid
                lines.append(f"  {label}: {count} activations")

        return "\n".join(lines)

    def update_feature_labels(
        self,
        labels: Dict[int, Any],
        source: Optional["FeatureSource"] = None,
        prefix: str = "sae",
    ) -> int:
        """
        Bulk update feature labels from autolabeling system.

        Integrates with FeatureLabeler to update descriptions and metadata
        for multiple features at once.

        Args:
            labels: Dict mapping feature index to FeatureLabel object or dict
                   (from labeler.py's FeatureLabel dataclass or similar)
            source: Feature source to update (default: SAE)
            prefix: Feature ID prefix (default: "sae")

        Returns:
            Number of features updated

        Example:
            # After running autolabeling
            from hololoom.dark_trace.sae.labeler import FeatureLabeler

            labeler = FeatureLabeler(llm_provider="anthropic")
            labels = await labeler.label_features(sae, [0, 1, 2, 3, 4])

            # Update registry with labels
            updated = registry.update_feature_labels(labels)
            print(f"Updated {updated} feature labels")
        """
        if source is None:
            source = FeatureSource.SAE

        updated = 0
        for idx, label in labels.items():
            feature_id = f"{prefix}.{idx}"

            # Get existing feature
            feature = self.get(feature_id)
            if not feature:
                continue

            # Extract label info (handle both dataclass and dict)
            if hasattr(label, "description"):
                # FeatureLabel dataclass
                description = label.description
                confidence = getattr(label, "confidence", 0.5)
                primary_axis = getattr(label, "primary_semantic_axis", None)
                alignments = getattr(label, "semantic_alignments", {})
                labeled_at = getattr(label, "labeled_at", None)
                labeler_model = getattr(label, "labeler_model", "unknown")
            elif isinstance(label, dict):
                # Dictionary format
                description = label.get("description", "")
                confidence = label.get("confidence", 0.5)
                primary_axis = label.get("primary_semantic_axis")
                alignments = label.get("semantic_alignments", {})
                labeled_at = label.get("labeled_at")
                labeler_model = label.get("labeler_model", "unknown")
            else:
                continue

            # Update feature
            feature.description = description
            feature.label = description[:50] + "..." if len(description) > 50 else description

            # Store autolabel metadata
            feature.metadata["autolabel"] = {
                "confidence": confidence,
                "primary_semantic_axis": primary_axis,
                "semantic_alignments": alignments,
                "labeled_at": labeled_at,
                "labeler_model": labeler_model,
            }

            updated += 1

        return updated

    # =============================================================================
    # Multi-Layer Correlation Integration
    # =============================================================================

    def register_layer_correlation(
        self,
        source_layer: int,
        target_layer: int,
        correlations: List[Tuple[int, int, float]],
        source_prefix: str = "sae",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Register cross-layer feature correlations from CorrelationTracker.

        Integrates multi-layer correlation data into the registry's correlation
        matrix, enabling cross-layer feature mapping and circuit discovery.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index
            correlations: List of (source_feature_idx, target_feature_idx, correlation)
                         tuples from CorrelationTracker
            source_prefix: Feature ID prefix (default: "sae")
            metadata: Optional metadata about correlation computation
                     (e.g., n_samples, algorithm, timestamp)

        Returns:
            Number of correlations registered

        Example:
            from hololoom.dark_trace.multilayer.correlation_tracker import (
                CorrelationTracker,
            )

            tracker = CorrelationTracker(n_layers=12)
            # ... process activations ...
            corr_data = tracker.get_layer_pair_correlations(0, 1, top_k=100)

            # Register in unified namespace
            registered = registry.register_layer_correlation(
                source_layer=0,
                target_layer=1,
                correlations=corr_data.top_correlations,
            )
        """
        registered = 0
        now = datetime.now()

        with self._lock:
            for src_idx, tgt_idx, correlation in correlations:
                src_id = f"{source_prefix}.L{source_layer}.{src_idx}"
                tgt_id = f"{source_prefix}.L{target_layer}.{tgt_idx}"

                # Create FeatureCorrelation entry
                corr_entry = FeatureCorrelation(
                    source_id=src_id,
                    target_id=tgt_id,
                    correlation=correlation,
                    sample_count=metadata.get("n_samples", 0) if metadata else 0,
                    confidence=0.95 if abs(correlation) > 0.3 else 0.8,
                )

                # Store in correlation matrix
                if src_id not in self.correlations._correlations:
                    self.correlations._correlations[src_id] = {}
                self.correlations._correlations[src_id][tgt_id] = corr_entry

                registered += 1

            # Store layer pair metadata
            pair_key = f"layer_pair.{source_layer}.{target_layer}"
            if not hasattr(self, "_layer_pair_metadata"):
                self._layer_pair_metadata: Dict[str, Dict[str, Any]] = {}

            self._layer_pair_metadata[pair_key] = {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "n_correlations": registered,
                "registered_at": now.isoformat(),
                **(metadata or {}),
            }

        return registered

    def get_layer_correlations(
        self,
        source_layer: int,
        target_layer: int,
        min_correlation: float = 0.3,
        source_prefix: str = "sae",
    ) -> List[FeatureCorrelation]:
        """
        Get all correlations between two layers.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index
            min_correlation: Minimum absolute correlation to include
            source_prefix: Feature ID prefix

        Returns:
            List of FeatureCorrelation objects
        """
        results = []
        src_pattern = f"{source_prefix}.L{source_layer}."
        tgt_pattern = f"{source_prefix}.L{target_layer}."

        for src_id, targets in self.correlations._correlations.items():
            if not src_id.startswith(src_pattern):
                continue

            for tgt_id, corr in targets.items():
                if tgt_id.startswith(tgt_pattern):
                    if abs(corr.correlation) >= min_correlation:
                        results.append(corr)

        # Sort by absolute correlation descending
        results.sort(key=lambda c: abs(c.correlation), reverse=True)
        return results

    def get_layer_pair_metadata(
        self,
        source_layer: int,
        target_layer: int,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a layer pair correlation registration."""
        if not hasattr(self, "_layer_pair_metadata"):
            return None
        pair_key = f"layer_pair.{source_layer}.{target_layer}"
        return self._layer_pair_metadata.get(pair_key)

    def get_feature_propagation_path(
        self,
        feature_idx: int,
        start_layer: int,
        end_layer: int,
        min_correlation: float = 0.3,
        source_prefix: str = "sae",
    ) -> List[Dict[str, Any]]:
        """
        Trace a feature's propagation path through layers.

        Uses registered correlations to find the most likely path
        a feature takes through the model's layers.

        Args:
            feature_idx: Starting feature index
            start_layer: Starting layer
            end_layer: Ending layer
            min_correlation: Minimum correlation threshold
            source_prefix: Feature ID prefix

        Returns:
            List of dicts with path information:
            [{"layer": int, "feature": int, "correlation": float}, ...]
        """
        path = [{"layer": start_layer, "feature": feature_idx, "correlation": 1.0}]
        current_feature = feature_idx
        current_layer = start_layer

        direction = 1 if end_layer > start_layer else -1
        layers_to_traverse = range(start_layer, end_layer, direction)

        for layer in layers_to_traverse:
            next_layer = layer + direction

            # Find best correlated feature in next layer
            correlations = self.get_layer_correlations(
                source_layer=layer,
                target_layer=next_layer,
                min_correlation=min_correlation,
                source_prefix=source_prefix,
            )

            # Find correlation from current feature
            src_id = f"{source_prefix}.L{layer}.{current_feature}"
            best_corr = None
            best_target_idx = None

            for corr in correlations:
                if corr.source_id == src_id:
                    if best_corr is None or abs(corr.correlation) > abs(best_corr):
                        best_corr = corr.correlation
                        # Extract target feature index
                        parts = corr.target_id.split(".")
                        if len(parts) >= 3:
                            try:
                                best_target_idx = int(parts[-1])
                            except ValueError:
                                pass

            if best_target_idx is not None:
                path.append({
                    "layer": next_layer,
                    "feature": best_target_idx,
                    "correlation": best_corr,
                })
                current_feature = best_target_idx
            else:
                # Path ends here
                break

        return path

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary."""
        return {
            "features": {
                fid: {
                    "source": f.source.value,
                    "index": f.index,
                    "label": f.label,
                    "description": f.description,
                    "safety_flag": f.safety_flag.name,
                    "sparsity": f.sparsity,
                }
                for fid, f in self._features.items()
            },
            "correlations": self.correlations.to_dict(),
            "activation_stats": {
                fid: self.get_activation_stats(fid)
                for fid in self._activation_counts
            },
        }

    def save(self, path: str) -> None:
        """Save registry to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureRegistry":
        """Load registry from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        registry = cls(enable_history=False)

        for fid, fdata in data.get("features", {}).items():
            feature = Feature(
                id=fid,
                source=FeatureSource(fdata["source"]),
                index=fdata["index"],
                label=fdata.get("label"),
                description=fdata.get("description"),
                safety_flag=SafetyFlag[fdata.get("safety_flag", "SAFE")],
                sparsity=fdata.get("sparsity", 0.0),
            )
            registry.register(feature)

        return registry


# =============================================================================
# Convenience Functions
# =============================================================================

def create_sae_features(n_features: int, prefix: str = "sae") -> List[Feature]:
    """
    Create SAE feature objects for registration.

    Args:
        n_features: Number of SAE features
        prefix: Feature ID prefix

    Returns:
        List of Feature objects ready for registration
    """
    return [
        Feature(
            id=f"{prefix}.{i}",
            source=FeatureSource.SAE,
            index=i,
            label=f"SAE Feature {i}",
        )
        for i in range(n_features)
    ]


def create_semantic_features(
    dimensions: List[Dict[str, Any]]
) -> List[Feature]:
    """
    Create Semantic feature objects from dimension definitions.

    Args:
        dimensions: List of {"name": str, "positive": [...], "negative": [...]}

    Returns:
        List of Feature objects ready for registration
    """
    return [
        Feature(
            id=f"semantic.{dim['name']}",
            source=FeatureSource.SEMANTIC,
            index=i,
            label=dim["name"],
            exemplars=dim.get("positive", []),
            metadata={
                "positive_exemplars": dim.get("positive", []),
                "negative_exemplars": dim.get("negative", []),
            },
        )
        for i, dim in enumerate(dimensions)
    ]
