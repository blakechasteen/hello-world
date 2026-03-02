"""
Dark Trace Orchestrator Integration

Integrates Dark Trace interpretability into HoloLoom's weaving orchestrator,
providing automatic feature analysis at each stage of the weaving cycle.

Key Features:
- Automatic feature extraction at weaving stages
- Real-time safety feature detection
- Complete feature provenance for decisions
- Steering integration for controlled generation

Integration Points:
1. Feature Extraction (Step 4): Extract SAE/Semantic features from DotPlasma
2. Decision Making (Step 7): Trace features to tool selection
3. Output (Step 9): Attach feature analysis to Spacetime

Usage:
    from hololoom.dark_trace.integration.orchestrator import (
        DarkTraceOrchestrator,
        create_traced_orchestrator,
    )

    # Create orchestrator with interpretability
    orchestrator = create_traced_orchestrator(config, shards)

    # Weave with automatic feature tracing
    spacetime = await orchestrator.weave(query)

    # Access interpretability analysis
    trace = spacetime.metadata['dark_trace']
    print(f"Active features: {len(trace.active_features)}")
    print(f"Safety-relevant: {trace.safety_features}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum
import numpy as np
import time

if TYPE_CHECKING:
    from hololoom.config import Config
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.result import TraceResult


class TracePoint(Enum):
    """Points in the weaving cycle where tracing occurs."""
    FEATURE_EXTRACTION = "feature_extraction"  # Step 4: DotPlasma creation
    MEMORY_RETRIEVAL = "memory_retrieval"      # Step 6: Context retrieval
    DECISION = "decision"                       # Step 7: Tool selection
    OUTPUT = "output"                          # Step 9: Spacetime creation


@dataclass
class FeatureTrace:
    """Trace of a single feature activation."""
    feature_id: str
    activation: float
    layer: Optional[str] = None
    label: Optional[str] = None
    is_safety_relevant: bool = False
    semantic_dimensions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """Trace connecting features to a decision."""
    tool_selected: str
    confidence: float
    top_features: List[FeatureTrace] = field(default_factory=list)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    safety_score: Optional[float] = None
    explanation: Optional[str] = None


@dataclass
class TracedSpacetime:
    """Extended Spacetime with full interpretability trace."""

    # Original spacetime data
    response: str
    confidence: float
    tool_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Interpretability traces
    feature_traces: List[FeatureTrace] = field(default_factory=list)
    decision_trace: Optional[DecisionTrace] = None
    safety_features: List[FeatureTrace] = field(default_factory=list)

    # Timing
    trace_time_ms: float = 0.0

    @property
    def active_features(self) -> List[str]:
        """Get list of active feature IDs."""
        return [ft.feature_id for ft in self.feature_traces]

    @property
    def has_safety_concerns(self) -> bool:
        """Check if any safety-relevant features are highly active."""
        return any(ft.activation > 0.5 for ft in self.safety_features)

    def get_feature_explanation(self) -> str:
        """Generate explanation of active features."""
        if not self.feature_traces:
            return "No feature traces available."

        lines = [f"Active Features ({len(self.feature_traces)} total):"]

        # Top 5 by activation
        sorted_features = sorted(
            self.feature_traces,
            key=lambda f: f.activation,
            reverse=True
        )[:5]

        for ft in sorted_features:
            label = ft.label or ft.feature_id
            safety_marker = " [SAFETY]" if ft.is_safety_relevant else ""
            lines.append(f"  - {label}: {ft.activation:.3f}{safety_marker}")

        if self.safety_features:
            lines.append(f"\nSafety-Relevant Features ({len(self.safety_features)}):")
            for ft in self.safety_features[:3]:
                lines.append(f"  - {ft.label or ft.feature_id}: {ft.activation:.3f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "response": self.response,
            "confidence": self.confidence,
            "tool_used": self.tool_used,
            "metadata": self.metadata,
            "feature_traces": [
                {
                    "feature_id": ft.feature_id,
                    "activation": ft.activation,
                    "label": ft.label,
                    "is_safety_relevant": ft.is_safety_relevant,
                }
                for ft in self.feature_traces
            ],
            "safety_features": [
                {
                    "feature_id": ft.feature_id,
                    "activation": ft.activation,
                    "label": ft.label,
                }
                for ft in self.safety_features
            ],
            "trace_time_ms": self.trace_time_ms,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for Dark Trace orchestrator integration."""

    # Tracing settings
    enable_feature_tracing: bool = True
    enable_safety_detection: bool = True
    enable_decision_tracing: bool = True

    # Which trace points to enable
    trace_points: Set[TracePoint] = field(
        default_factory=lambda: {
            TracePoint.FEATURE_EXTRACTION,
            TracePoint.DECISION,
            TracePoint.OUTPUT,
        }
    )

    # Feature selection
    max_features_to_trace: int = 50
    min_activation_threshold: float = 0.1
    safety_activation_threshold: float = 0.3

    # Performance
    async_tracing: bool = False  # Run tracing in background
    cache_traces: bool = True


class DarkTraceOrchestrator:
    """
    Wraps a HoloLoom orchestrator with Dark Trace interpretability.

    Provides automatic feature analysis at each stage of the weaving cycle,
    creating full provenance from input to decision to output.
    """

    def __init__(
        self,
        orchestrator: Any,
        engine: Optional["DarkTraceEngine"] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize traced orchestrator.

        Args:
            orchestrator: Base HoloLoom WeavingOrchestrator
            engine: DarkTraceEngine for feature analysis
            config: Orchestrator configuration
        """
        self.orchestrator = orchestrator
        self.engine = engine
        self.config = config or OrchestratorConfig()

        # Trace cache
        self._trace_cache: Dict[str, TracedSpacetime] = {}

    async def weave(
        self,
        query: Any,
        **kwargs,
    ) -> TracedSpacetime:
        """
        Weave with full interpretability tracing.

        Args:
            query: Query to process
            **kwargs: Additional arguments for orchestrator

        Returns:
            TracedSpacetime with full feature analysis
        """
        start_time = time.time()

        # Run base weaving
        spacetime = await self.orchestrator.weave(query, **kwargs)

        # Extract traces if engine available
        feature_traces = []
        decision_trace = None
        safety_features = []

        if self.engine and self.config.enable_feature_tracing:
            # Get activations from spacetime or orchestrator
            activations = self._extract_activations(spacetime)

            if activations is not None:
                # Analyze with Dark Trace engine
                trace_result = self.engine.analyze(activations)

                # Convert to feature traces
                feature_traces = self._create_feature_traces(trace_result)

                # Extract safety-relevant features
                if self.config.enable_safety_detection:
                    safety_features = self._detect_safety_features(
                        feature_traces,
                        trace_result
                    )

                # Create decision trace
                if self.config.enable_decision_tracing:
                    decision_trace = self._create_decision_trace(
                        spacetime,
                        feature_traces
                    )

        trace_time = (time.time() - start_time) * 1000

        # Create traced spacetime
        traced = TracedSpacetime(
            response=getattr(spacetime, 'response', str(spacetime)),
            confidence=getattr(spacetime, 'confidence', 0.5),
            tool_used=getattr(spacetime, 'tool_used', 'unknown'),
            metadata=getattr(spacetime, 'metadata', {}),
            feature_traces=feature_traces,
            decision_trace=decision_trace,
            safety_features=safety_features,
            trace_time_ms=trace_time,
        )

        # Attach to metadata
        traced.metadata['dark_trace'] = traced.to_dict()

        # Cache if enabled
        if self.config.cache_traces:
            cache_key = str(query)[:100]
            self._trace_cache[cache_key] = traced

        return traced

    def _extract_activations(self, spacetime: Any) -> Optional[np.ndarray]:
        """Extract activations from spacetime for analysis."""
        # Try different sources
        if hasattr(spacetime, 'activations'):
            return spacetime.activations

        if hasattr(spacetime, 'features'):
            features = spacetime.features
            if hasattr(features, 'embedding'):
                return np.array(features.embedding)
            if isinstance(features, np.ndarray):
                return features

        if hasattr(spacetime, 'metadata'):
            if 'activations' in spacetime.metadata:
                return np.array(spacetime.metadata['activations'])
            if 'features' in spacetime.metadata:
                return np.array(spacetime.metadata['features'])

        return None

    def _create_feature_traces(
        self,
        trace_result: "TraceResult",
    ) -> List[FeatureTrace]:
        """Create feature traces from trace result."""
        traces = []

        # Get active features from each lens
        for lens_name, lens_result in trace_result.lens_results.items():
            if lens_result is None:
                continue

            active = getattr(lens_result, 'active_features', [])
            for feature in active[:self.config.max_features_to_trace]:
                feature_id = getattr(feature, 'id', str(feature))
                activation = getattr(feature, 'activation', 0.5)

                if activation < self.config.min_activation_threshold:
                    continue

                trace = FeatureTrace(
                    feature_id=f"{lens_name}.{feature_id}",
                    activation=float(activation),
                    label=getattr(feature, 'label', None),
                    is_safety_relevant=getattr(feature, 'safety_relevant', False),
                    metadata={"lens": lens_name},
                )
                traces.append(trace)

        return traces

    def _detect_safety_features(
        self,
        feature_traces: List[FeatureTrace],
        trace_result: "TraceResult",
    ) -> List[FeatureTrace]:
        """Detect safety-relevant features."""
        safety_features = []

        # Get features already marked as safety-relevant
        for ft in feature_traces:
            if ft.is_safety_relevant and ft.activation >= self.config.safety_activation_threshold:
                safety_features.append(ft)

        # Check semantic dimensions if available
        if hasattr(trace_result, 'semantic_analysis'):
            semantic = trace_result.semantic_analysis
            if semantic:
                # Look for high deception, harm, power-seeking
                safety_dims = ['deception', 'harm_potential', 'power_seeking', 'manipulation']
                for dim in safety_dims:
                    if dim in semantic and semantic[dim] > self.config.safety_activation_threshold:
                        safety_features.append(FeatureTrace(
                            feature_id=f"semantic.{dim}",
                            activation=float(semantic[dim]),
                            label=dim.replace('_', ' ').title(),
                            is_safety_relevant=True,
                            semantic_dimensions={dim: semantic[dim]},
                        ))

        return safety_features

    def _create_decision_trace(
        self,
        spacetime: Any,
        feature_traces: List[FeatureTrace],
    ) -> DecisionTrace:
        """Create decision trace linking features to tool selection."""
        tool_used = getattr(spacetime, 'tool_used', 'unknown')
        confidence = getattr(spacetime, 'confidence', 0.5)

        # Get top features
        top_features = sorted(
            feature_traces,
            key=lambda f: f.activation,
            reverse=True
        )[:10]

        # Calculate feature contributions (simplified)
        contributions = {}
        total_activation = sum(f.activation for f in top_features) or 1.0
        for ft in top_features:
            contributions[ft.feature_id] = ft.activation / total_activation

        # Compute safety score from safety features
        safety_score = None
        if feature_traces:
            safety_activations = [
                ft.activation for ft in feature_traces
                if ft.is_safety_relevant
            ]
            if safety_activations:
                # Higher score = more safety concerns
                safety_score = float(np.mean(safety_activations))

        return DecisionTrace(
            tool_selected=tool_used,
            confidence=confidence,
            top_features=top_features,
            feature_contributions=contributions,
            safety_score=safety_score,
            explanation=self._generate_explanation(top_features, tool_used),
        )

    def _generate_explanation(
        self,
        top_features: List[FeatureTrace],
        tool_used: str,
    ) -> str:
        """Generate natural language explanation of decision."""
        if not top_features:
            return f"Selected {tool_used} with no strong feature activations."

        feature_labels = []
        for ft in top_features[:3]:
            label = ft.label or ft.feature_id.split('.')[-1]
            feature_labels.append(label)

        features_str = ", ".join(feature_labels)
        return f"Selected {tool_used} based on features: {features_str}"

    def get_cached_trace(self, query: str) -> Optional[TracedSpacetime]:
        """Get cached trace for a query."""
        cache_key = query[:100]
        return self._trace_cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear trace cache."""
        self._trace_cache.clear()

    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.orchestrator, 'close'):
            await self.orchestrator.close()


def create_traced_orchestrator(
    config: "Config",
    shards: Any = None,
    dark_trace_engine: Optional["DarkTraceEngine"] = None,
    orchestrator_config: Optional[OrchestratorConfig] = None,
) -> DarkTraceOrchestrator:
    """
    Create a DarkTrace-enabled orchestrator.

    Args:
        config: HoloLoom configuration
        shards: Memory shards (optional)
        dark_trace_engine: Existing engine (optional, will create if needed)
        orchestrator_config: Orchestrator configuration

    Returns:
        Configured DarkTraceOrchestrator
    """
    # Import here to avoid circular imports
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    # Create base orchestrator
    base_orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

    # Create or use engine
    engine = dark_trace_engine
    if engine is None:
        try:
            from hololoom.dark_trace.engine import DarkTraceEngine
            from hololoom.dark_trace.trace_config import TraceConfig
            engine = DarkTraceEngine(TraceConfig.standard())
        except ImportError:
            engine = None

    return DarkTraceOrchestrator(
        orchestrator=base_orchestrator,
        engine=engine,
        config=orchestrator_config or OrchestratorConfig(),
    )
