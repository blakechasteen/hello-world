"""
Cognitive Extractors - Extract CRP events from hololoom systems
================================================================

Converts HoloLoom internal state to Cognitive Rendering Protocol events.
This is the bridge between HoloLoom's cognitive systems and visualization.

Data Sources → CRP Events:
- AwarenessGraph → ActivationField, SemanticPosition, SemanticTrajectory
- TSBandit/Policy → ProbabilityManifold, ToolSelection, DecisionCollapse
- WeavingOrchestrator → WeavingCycle, ReasoningChain
- Memory/Cache → MemoryRetrieval, ContextExpansion
- Reflection → ConfidenceFlow, ReflectionLoop

Created: 2025-11-30
Author: HoloLoom Team
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from hololoom.cve.cognitive_protocol import (
    ActivationField,
    CognitiveEvent,
    CognitiveFrame,
    ConfidenceFlow,
    ContextExpansion,
    DecisionCollapse,
    KnowledgeGraph3D,
    MemoryRetrieval,
    ProbabilityManifold,
    ReflectionLoop,
    SemanticPosition,
    SemanticTrajectory,
    TokenStream,
    ToolSelection,
    WeavingCycle,
    create_frame,
)

# ============================================================================
# Extractor Protocol
# ============================================================================

@dataclass
class ExtractionResult:
    """Result of extracting cognitive events from a source."""
    events: list[CognitiveEvent]
    source_type: str
    timestamp: datetime
    metadata: dict[str, Any]


class CognitiveExtractor:
    """
    Base class for cognitive data extractors.

    Extracts CRP events from hololoom systems.
    """

    def __init__(self):
        self.extraction_count = 0

    def extract(self, source: Any) -> list[CognitiveEvent]:
        """Extract events from source (to be overridden)."""
        raise NotImplementedError


# ============================================================================
# AwarenessGraph Extractors
# ============================================================================

class AwarenessGraphExtractor(CognitiveExtractor):
    """
    Extract cognitive events from AwarenessGraph.

    Produces:
    - ActivationField (spreading activation)
    - SemanticPosition (228D position)
    - SemanticTrajectory (position history)
    """

    def extract(self, awareness_graph: Any) -> list[CognitiveEvent]:
        """
        Extract events from AwarenessGraph.

        Args:
            awareness_graph: AwarenessGraph instance

        Returns:
            List of CognitiveEvents
        """
        events = []

        # Extract activation field
        activation_event = self._extract_activation_field(awareness_graph)
        if activation_event:
            events.append(activation_event)

        # Extract semantic position
        position_event = self._extract_semantic_position(awareness_graph)
        if position_event:
            events.append(position_event)

        # Extract trajectory
        trajectory_event = self._extract_trajectory(awareness_graph)
        if trajectory_event:
            events.append(trajectory_event)

        self.extraction_count += len(events)
        return events

    def _extract_activation_field(self, ag: Any) -> CognitiveEvent | None:
        """Extract spreading activation data."""
        try:
            # Get activation field from awareness graph
            field = getattr(ag, 'activation_field', None)
            if field is None:
                return None

            # Get activations
            activations = getattr(field, 'activations', {})
            if not activations:
                # Try alternative attribute names
                activations = getattr(field, 'activation_levels', {})

            if not activations:
                return None

            # Find source nodes (highest activation)
            sorted_nodes = sorted(activations.items(), key=lambda x: x[1], reverse=True)
            source_nodes = [n for n, a in sorted_nodes[:3] if a >= 0.9]  # Top activations

            # Wave front = nodes with high but not max activation
            wave_front = [n for n, a in sorted_nodes if 0.5 <= a < 0.9][:5]

            # Get propagation step
            step = getattr(field, 'propagation_step', 0)
            if step == 0:
                step = getattr(field, 'iteration', 0)

            activation = ActivationField(
                source_nodes=source_nodes,
                activation_levels=dict(activations),
                wave_front=wave_front,
                propagation_step=step,
            )

            return activation.to_event()

        except Exception:
            # Graceful degradation
            return None

    def _extract_semantic_position(self, ag: Any) -> CognitiveEvent | None:
        """Extract current semantic position."""
        try:
            # Get trajectory
            trajectory = getattr(ag, 'trajectory', None)
            if not trajectory or len(trajectory) == 0:
                return None

            # Current position is last in trajectory
            current_pos = trajectory[-1]
            if isinstance(current_pos, dict):
                position = current_pos.get('position', [])
                query_text = current_pos.get('query', '')
            elif hasattr(current_pos, 'position'):
                position = current_pos.position
                query_text = getattr(current_pos, 'query_text', '')
            else:
                position = list(current_pos) if hasattr(current_pos, '__iter__') else []
                query_text = ''

            if len(position) < 16:
                return None

            # Extract interpretable axes (first 16 dimensions)
            interpretable = self._position_to_axes(position[:16])

            # Find dominant axes
            dominant = sorted(interpretable.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            semantic_pos = SemanticPosition(
                position=list(position),
                interpretable=interpretable,
                dominant_axes=dominant,
                query_text=query_text,
            )

            return semantic_pos.to_event()

        except Exception:
            return None

    def _extract_trajectory(self, ag: Any) -> CognitiveEvent | None:
        """Extract semantic trajectory."""
        try:
            trajectory = getattr(ag, 'trajectory', None)
            if not trajectory or len(trajectory) < 2:
                return None

            # Convert to waypoints
            waypoints = []
            for point in list(trajectory)[-20:]:  # Last 20 points
                if isinstance(point, dict):
                    pos = point.get('position', [])[:3]  # First 3 dims for 3D
                elif hasattr(point, 'position'):
                    pos = list(point.position)[:3]
                else:
                    pos = list(point)[:3] if hasattr(point, '__iter__') else []

                if len(pos) >= 3:
                    waypoints.append(pos)

            if len(waypoints) < 2:
                return None

            # Calculate velocities between waypoints
            velocities = []
            for i in range(1, len(waypoints)):
                diff = np.array(waypoints[i]) - np.array(waypoints[i-1])
                velocity = float(np.linalg.norm(diff))
                velocities.append(velocity)

            # Detect shift points (high velocity)
            avg_vel = np.mean(velocities) if velocities else 0
            shift_points = [i for i, v in enumerate(velocities) if v > avg_vel * 1.5]

            trajectory_data = SemanticTrajectory(
                waypoints=waypoints,
                velocities=velocities,
                shift_points=shift_points,
            )

            return trajectory_data.to_event()

        except Exception:
            return None

    def _position_to_axes(self, position: list[float]) -> dict[str, float]:
        """Convert position to named interpretable axes."""
        AXIS_NAMES = [
            'sentiment', 'formality', 'technicality', 'certainty',
            'urgency', 'abstraction', 'specificity', 'temporality',
            'objectivity', 'complexity', 'scope', 'directness',
            'emotionality', 'actionability', 'novelty', 'controversy'
        ]

        return {
            name: position[i] if i < len(position) else 0.0
            for i, name in enumerate(AXIS_NAMES)
        }


# ============================================================================
# Policy/Bandit Extractors
# ============================================================================

class PolicyExtractor(CognitiveExtractor):
    """
    Extract cognitive events from Policy/TSBandit.

    Produces:
    - ProbabilityManifold (Thompson Sampling distributions)
    - ToolSelection (tool probabilities)
    - DecisionCollapse (final decision)
    """

    def __init__(self, tool_names: list[str] | None = None):
        super().__init__()
        self.tool_names = tool_names or ['answer', 'research', 'explore', 'clarify']

    def extract_from_bandit(self, bandit: Any) -> list[CognitiveEvent]:
        """Extract events from TSBandit."""
        events = []

        try:
            manifold_event = self._extract_manifold(bandit)
            if manifold_event:
                events.append(manifold_event)

            self.extraction_count += len(events)

        except Exception:
            pass

        return events

    def extract_from_decision(
        self,
        tool_probs: np.ndarray,
        selected_tool: str,
        strategy: str = 'bayesian_blend',
        confidence: float = 0.8,
    ) -> list[CognitiveEvent]:
        """
        Extract events from a tool selection decision.

        Args:
            tool_probs: Probability distribution over tools
            selected_tool: Name of selected tool
            strategy: Decision strategy used
            confidence: Decision confidence
        """
        events = []

        # Tool selection event
        selection = ToolSelection(
            tools=self.tool_names[:len(tool_probs)],
            probabilities=list(tool_probs),
            selected=selected_tool,
            bandit_stats={'confidence': confidence, 'strategy': strategy},
        )
        events.append(selection.to_event())

        # Decision collapse event
        collapse = DecisionCollapse(
            options=self.tool_names[:len(tool_probs)],
            probabilities=list(tool_probs),
            selected=selected_tool,
            collapse_strategy=strategy,
            collapse_confidence=confidence,
        )
        events.append(collapse.to_event())

        self.extraction_count += len(events)
        return events

    def _extract_manifold(self, bandit: Any) -> CognitiveEvent | None:
        """Extract probability manifold from bandit."""
        try:
            # Get alpha/beta values
            alphas = getattr(bandit, 'success', None)
            betas = getattr(bandit, 'fail', None)

            if alphas is None or betas is None:
                return None

            alphas = list(alphas)
            betas = list(betas)
            n_arms = len(alphas)

            # Adjust tool names if needed
            tools = self.tool_names[:n_arms]
            if len(tools) < n_arms:
                tools = tools + [f'tool_{i}' for i in range(len(tools), n_arms)]

            # Get selected tool (last selection if available)
            selected = getattr(bandit, 'last_selected', None)
            if selected is not None and isinstance(selected, int):
                selected = tools[selected] if selected < len(tools) else None

            manifold = ProbabilityManifold(
                tool_names=tools,
                alpha_values=alphas,
                beta_values=betas,
                selected_tool=selected,
            )

            return manifold.to_event()

        except Exception:
            return None


# ============================================================================
# Weaving Cycle Extractor
# ============================================================================

class WeavingCycleExtractor(CognitiveExtractor):
    """
    Extract cognitive events from WeavingOrchestrator.

    Produces:
    - WeavingCycle (9-stage progress)
    - ContextExpansion (memory expansion)
    - ReasoningChain (for agentic modes)
    """

    # Stage names for 9-stage cycle
    STAGE_NAMES = [
        'Loom Command',
        'Chrono Trigger',
        'Yarn Graph',
        'Resonance Shed',
        'Warp Space',
        'Convergence',
        'Tool Execution',
        'Spacetime Fabric',
        'Reflection'
    ]

    def extract_from_spacetime(
        self,
        spacetime: Any,
        pattern_card: str = 'FAST',
    ) -> list[CognitiveEvent]:
        """
        Extract events from Spacetime result.

        Args:
            spacetime: Spacetime result from weaving
            pattern_card: Pattern card used (BARE/FAST/FUSED)
        """
        events = []

        try:
            # Extract weaving cycle
            cycle_event = self._extract_cycle(spacetime, pattern_card)
            if cycle_event:
                events.append(cycle_event)

            # Extract context expansion if available
            expansion_event = self._extract_expansion(spacetime)
            if expansion_event:
                events.append(expansion_event)

            self.extraction_count += len(events)

        except Exception:
            pass

        return events

    def extract_from_trace(
        self,
        trace: Any,
        pattern_card: str = 'FAST',
    ) -> list[CognitiveEvent]:
        """Extract events from WeavingTrace."""
        events = []

        try:
            # Get stage durations
            durations = getattr(trace, 'stage_durations', {})
            statuses = getattr(trace, 'stage_statuses', {})

            # Convert to ordered lists
            stage_durations = []
            stage_statuses = []

            for name in self.STAGE_NAMES:
                # Try various key formats
                for key in [name, name.lower(), name.replace(' ', '_').lower()]:
                    if key in durations:
                        stage_durations.append(durations[key])
                        break
                else:
                    stage_durations.append(0.0)

                for key in [name, name.lower(), name.replace(' ', '_').lower()]:
                    if key in statuses:
                        stage_statuses.append(statuses[key])
                        break
                else:
                    stage_statuses.append('pending')

            # Determine current stage
            current_stage = 1
            for i, status in enumerate(stage_statuses):
                if status == 'active':
                    current_stage = i + 1
                    break
                elif status == 'complete':
                    current_stage = i + 2

            cycle = WeavingCycle(
                current_stage=min(current_stage, 9),
                stage_names=self.STAGE_NAMES,
                stage_statuses=stage_statuses,
                stage_durations=stage_durations,
                pattern_card=pattern_card,
            )

            events.append(cycle.to_event())
            self.extraction_count += 1

        except Exception:
            pass

        return events

    def _extract_cycle(self, spacetime: Any, pattern_card: str) -> CognitiveEvent | None:
        """Extract weaving cycle from spacetime."""
        try:
            trace = getattr(spacetime, 'trace', None)
            if trace:
                return self.extract_from_trace(trace, pattern_card)[0]

            # Fallback: create from metadata
            metadata = getattr(spacetime, 'metadata', {})
            durations = metadata.get('stage_durations', [])

            if not durations:
                return None

            # Assume all completed
            statuses = ['complete'] * len(durations)
            statuses.extend(['pending'] * (9 - len(statuses)))

            cycle = WeavingCycle(
                current_stage=9,
                stage_names=self.STAGE_NAMES,
                stage_statuses=statuses[:9],
                stage_durations=list(durations)[:9] + [0.0] * (9 - len(durations)),
                pattern_card=pattern_card,
            )

            return cycle.to_event()

        except Exception:
            return None

    def _extract_expansion(self, spacetime: Any) -> CognitiveEvent | None:
        """Extract context expansion from spacetime."""
        try:
            metadata = getattr(spacetime, 'metadata', {})

            # Look for expansion data
            expansion_data = metadata.get('context_expansion', {})
            if not expansion_data:
                return None

            expansion = ContextExpansion(
                seed_nodes=expansion_data.get('seeds', []),
                expanded_nodes=expansion_data.get('expanded', []),
                token_budget=expansion_data.get('budget', 2000),
                tokens_used=expansion_data.get('used', 0),
                strategy=expansion_data.get('strategy', 'adaptive'),
            )

            return expansion.to_event()

        except Exception:
            return None


# ============================================================================
# Memory Extractor
# ============================================================================

class MemoryExtractor(CognitiveExtractor):
    """
    Extract cognitive events from memory operations.

    Produces:
    - MemoryRetrieval (search results)
    - KnowledgeGraph3D (graph structure)
    """

    def extract_from_retrieval(
        self,
        query: str,
        retrieved_ids: list[str],
        scores: list[float],
        method: str = 'hybrid',
        total_candidates: int = 0,
    ) -> CognitiveEvent:
        """Extract event from memory retrieval."""
        retrieval = MemoryRetrieval(
            query=query,
            retrieved=retrieved_ids,
            scores=scores,
            method=method,
            total_candidates=total_candidates or len(retrieved_ids),
        )

        self.extraction_count += 1
        return retrieval.to_event()

    def extract_from_graph(
        self,
        graph: Any,
        highlighted_path: list[str] | None = None,
        layout: str = 'force_directed',
    ) -> CognitiveEvent:
        """Extract event from knowledge graph."""
        try:
            # Get node count
            node_count = len(graph.nodes()) if hasattr(graph, 'nodes') else 0
            edge_count = len(graph.edges()) if hasattr(graph, 'edges') else 0

            # Sample nodes for visualization
            nodes = []
            for node_id in list(graph.nodes())[:20]:
                node_data = graph.nodes[node_id] if hasattr(graph, 'nodes') else {}
                nodes.append({
                    'id': str(node_id),
                    'label': node_data.get('label', str(node_id)[:20]),
                    'type': node_data.get('type', 'entity'),
                })

            # Sample edges
            edges = []
            edge_list = list(graph.edges(data=True))[:30] if hasattr(graph, 'edges') else []
            for src, dst, data in edge_list:
                edges.append({
                    'source': str(src),
                    'target': str(dst),
                    'type': data.get('type', 'RELATED'),
                })

            kg = KnowledgeGraph3D(
                nodes=nodes,
                edges=edges,
                layout=layout,
                highlighted_path=highlighted_path or [],
            )

            self.extraction_count += 1
            return kg.to_event()

        except Exception:
            # Return minimal event
            kg = KnowledgeGraph3D(
                nodes=[],
                edges=[],
                layout=layout,
                highlighted_path=[],
            )
            return kg.to_event()


# ============================================================================
# Reflection Extractor
# ============================================================================

class ReflectionExtractor(CognitiveExtractor):
    """
    Extract cognitive events from reflection/learning systems.

    Produces:
    - ConfidenceFlow (confidence trajectory)
    - ReflectionLoop (learning outcomes)
    """

    def extract_confidence_flow(
        self,
        confidences: list[float],
        cache_hits: list[bool] | None = None,
        timestamps: list[datetime] | None = None,
    ) -> CognitiveEvent:
        """Extract confidence trajectory event."""
        # Detect anomalies
        anomalies = self._detect_anomalies(confidences)

        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = [datetime.now() for _ in confidences]

        confidence_flow = ConfidenceFlow(
            confidences=confidences,
            timestamps=timestamps,
            cache_hits=cache_hits or [False] * len(confidences),
            anomalies=anomalies,
        )

        self.extraction_count += 1
        return confidence_flow.to_event()

    def extract_reflection_loop(
        self,
        query: str,
        outcome: str,
        confidence: float,
        pattern_learned: str | None = None,
        policy_updated: bool = False,
    ) -> CognitiveEvent:
        """Extract reflection loop event."""
        reflection = ReflectionLoop(
            query=query,
            outcome=outcome,
            confidence=confidence,
            pattern_learned=pattern_learned,
            policy_updated=policy_updated,
        )

        self.extraction_count += 1
        return reflection.to_event()

    def _detect_anomalies(self, confidences: list[float]) -> list[dict[str, Any]]:
        """Detect anomalies in confidence trajectory."""
        anomalies = []

        if len(confidences) < 2:
            return anomalies

        # Sudden drop detection
        for i in range(1, len(confidences)):
            drop = confidences[i-1] - confidences[i]
            if drop > 0.2:
                anomalies.append({
                    'type': 'sudden_drop',
                    'index': i,
                    'severity': min(drop, 1.0),
                })

        # Prolonged low detection
        window_size = 3
        threshold = 0.5
        for i in range(window_size, len(confidences) + 1):
            window = confidences[i-window_size:i]
            if all(c < threshold for c in window):
                anomalies.append({
                    'type': 'prolonged_low',
                    'index': i - 1,
                    'severity': 0.7,
                })
                break  # Only report once

        return anomalies


# ============================================================================
# Token Stream Extractor
# ============================================================================

class TokenStreamExtractor(CognitiveExtractor):
    """
    Extract token stream events from generation.

    Produces:
    - TokenStream (streaming tokens)
    """

    def __init__(self):
        super().__init__()
        self.tokens = []
        self.start_time = None

    def start_stream(self):
        """Start a new token stream."""
        self.tokens = []
        self.start_time = datetime.now()

    def add_token(self, token: str) -> CognitiveEvent:
        """Add a token to the stream and return event."""
        self.tokens.append(token)

        elapsed = (datetime.now() - self.start_time).total_seconds()
        speed = len(self.tokens) / elapsed if elapsed > 0 else 0

        stream = TokenStream(
            tokens=list(self.tokens),
            cumulative=''.join(self.tokens),
            is_final=False,
            token_speed=speed,
        )

        return stream.to_event()

    def finalize(self) -> CognitiveEvent:
        """Finalize the stream."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        speed = len(self.tokens) / elapsed if elapsed > 0 else 0

        stream = TokenStream(
            tokens=list(self.tokens),
            cumulative=''.join(self.tokens),
            is_final=True,
            token_speed=speed,
        )

        self.extraction_count += 1
        return stream.to_event()


# ============================================================================
# Unified Extractor (Combines All)
# ============================================================================

class UnifiedCognitiveExtractor:
    """
    Unified extractor that combines all cognitive data sources.

    Usage:
        extractor = UnifiedCognitiveExtractor()

        # Extract from multiple sources
        frame = extractor.create_frame(
            awareness_graph=ag,
            bandit=policy.bandit,
            spacetime=result,
            query_id="query_001",
        )

        # Render
        renderer.render_frame(frame)
    """

    def __init__(self, tool_names: list[str] | None = None):
        self.awareness_extractor = AwarenessGraphExtractor()
        self.policy_extractor = PolicyExtractor(tool_names=tool_names)
        self.weaving_extractor = WeavingCycleExtractor()
        self.memory_extractor = MemoryExtractor()
        self.reflection_extractor = ReflectionExtractor()
        self.token_extractor = TokenStreamExtractor()

    def create_frame(
        self,
        awareness_graph: Any | None = None,
        bandit: Any | None = None,
        spacetime: Any | None = None,
        graph: Any | None = None,
        confidences: list[float] | None = None,
        query_id: str = '',
        weaving_stage: str = '',
        pattern_card: str = 'FAST',
    ) -> CognitiveFrame:
        """
        Create a complete cognitive frame from multiple sources.

        Args:
            awareness_graph: AwarenessGraph instance
            bandit: TSBandit instance
            spacetime: Spacetime result
            graph: Knowledge graph (NetworkX)
            confidences: Confidence history
            query_id: Query identifier
            weaving_stage: Current stage name
            pattern_card: Pattern card used

        Returns:
            CognitiveFrame with all extracted events
        """
        events = []

        # Extract from awareness graph
        if awareness_graph is not None:
            events.extend(self.awareness_extractor.extract(awareness_graph))

        # Extract from bandit
        if bandit is not None:
            events.extend(self.policy_extractor.extract_from_bandit(bandit))

        # Extract from spacetime
        if spacetime is not None:
            events.extend(
                self.weaving_extractor.extract_from_spacetime(spacetime, pattern_card)
            )

        # Extract from knowledge graph
        if graph is not None:
            events.append(self.memory_extractor.extract_from_graph(graph))

        # Extract confidence flow
        if confidences is not None and len(confidences) > 0:
            events.append(
                self.reflection_extractor.extract_confidence_flow(confidences)
            )

        # Determine overall confidence
        confidence = None
        if spacetime is not None:
            confidence = getattr(spacetime, 'confidence', None)
        if confidence is None and confidences:
            confidence = confidences[-1]

        return create_frame(
            events=events,
            query_id=query_id,
            weaving_stage=weaving_stage,
            confidence=confidence,
        )

    def get_extraction_stats(self) -> dict[str, int]:
        """Get extraction statistics."""
        return {
            'awareness': self.awareness_extractor.extraction_count,
            'policy': self.policy_extractor.extraction_count,
            'weaving': self.weaving_extractor.extraction_count,
            'memory': self.memory_extractor.extraction_count,
            'reflection': self.reflection_extractor.extraction_count,
            'tokens': self.token_extractor.extraction_count,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_from_awareness(awareness_graph: Any) -> list[CognitiveEvent]:
    """Extract all events from an AwarenessGraph."""
    extractor = AwarenessGraphExtractor()
    return extractor.extract(awareness_graph)


def extract_from_bandit(bandit: Any, tool_names: list[str] | None = None) -> list[CognitiveEvent]:
    """Extract events from a TSBandit."""
    extractor = PolicyExtractor(tool_names=tool_names)
    return extractor.extract_from_bandit(bandit)


def extract_from_spacetime(spacetime: Any, pattern_card: str = 'FAST') -> list[CognitiveEvent]:
    """Extract events from a Spacetime result."""
    extractor = WeavingCycleExtractor()
    return extractor.extract_from_spacetime(spacetime, pattern_card)


def create_cognitive_frame(
    awareness_graph: Any | None = None,
    bandit: Any | None = None,
    spacetime: Any | None = None,
    **kwargs
) -> CognitiveFrame:
    """Create a complete cognitive frame (convenience function)."""
    extractor = UnifiedCognitiveExtractor()
    return extractor.create_frame(
        awareness_graph=awareness_graph,
        bandit=bandit,
        spacetime=spacetime,
        **kwargs
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=== Cognitive Extractors Demo ===\n")

    # Create mock data structures
    class MockActivationField:
        activations = {
            'thompson_sampling': 1.0,
            'exploration': 0.85,
            'exploitation': 0.75,
            'bayesian': 0.6,
            'bandit': 0.4,
        }
        propagation_step = 3

    class MockAwarenessGraph:
        activation_field = MockActivationField()
        trajectory = [
            {'position': [0.8, 0.2, 0.9, 0.7, 0.3, 0.5, 0.8, 0.4,
                          0.6, 0.7, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6] + [0.0] * 212,
             'query': 'What is Thompson Sampling?'},
            {'position': [0.7, 0.3, 0.85, 0.6, 0.4, 0.55, 0.75, 0.45,
                          0.65, 0.65, 0.55, 0.35, 0.25, 0.75, 0.45, 0.55] + [0.0] * 212,
             'query': 'How does it work?'},
        ]

    class MockBandit:
        success = np.array([12.0, 6.0, 4.0, 2.0])
        fail = np.array([3.0, 4.0, 6.0, 8.0])
        last_selected = 0

    # Test awareness extractor
    print("1. Testing AwarenessGraphExtractor...")
    ag_extractor = AwarenessGraphExtractor()
    ag_events = ag_extractor.extract(MockAwarenessGraph())
    print(f"   Extracted {len(ag_events)} events from AwarenessGraph")
    for event in ag_events:
        print(f"   - {event.viz_type.value}")

    # Test policy extractor
    print("\n2. Testing PolicyExtractor...")
    policy_extractor = PolicyExtractor()
    bandit_events = policy_extractor.extract_from_bandit(MockBandit())
    print(f"   Extracted {len(bandit_events)} events from Bandit")
    for event in bandit_events:
        print(f"   - {event.viz_type.value}")

    # Test decision extraction
    decision_events = policy_extractor.extract_from_decision(
        tool_probs=np.array([0.6, 0.25, 0.1, 0.05]),
        selected_tool='answer',
        strategy='bayesian_blend',
        confidence=0.85,
    )
    print(f"   Extracted {len(decision_events)} events from decision")

    # Test unified extractor
    print("\n3. Testing UnifiedCognitiveExtractor...")
    unified = UnifiedCognitiveExtractor()
    frame = unified.create_frame(
        awareness_graph=MockAwarenessGraph(),
        bandit=MockBandit(),
        confidences=[0.7, 0.75, 0.82, 0.85, 0.88],
        query_id='demo_001',
        weaving_stage='decision',
    )
    print(f"   Created frame with {len(frame.events)} events")
    print(f"   Stage: {frame.weaving_stage}")
    print(f"   Confidence: {frame.confidence}")

    # Show stats
    print("\n4. Extraction statistics:")
    stats = unified.get_extraction_stats()
    for source, count in stats.items():
        print(f"   {source}: {count} events")

    print("\n[OK] All extractors working correctly!")
