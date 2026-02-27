#!/usr/bin/env python3
"""
🌊📐 MULTI-SCALE STREAMING SEMANTIC CALCULUS
============================================
Real-time word-by-word semantic analysis with nested temporal scales.

This module combines:
- Word-level semantic trajectories (micro-movements)
- Phrase-level semantic gestures (local patterns)
- Sentence-level semantic thoughts (complete ideas)
- Paragraph-level semantic arcs (narrative beats)
- Chunk-level semantic acts (structural evolution)

The key innovation: RESONANCE BETWEEN SCALES
- When all scales align → Narrative momentum, flow
- When scales diverge → Complexity, irony, sophistication
- When scales couple → Fractal structure, self-similarity

Mathematical Framework:
- Position: q(t) in 244D semantic space per scale
- Velocity: dq/dt = rate of semantic change
- Acceleration: d²q/dt² = semantic forces
- Cross-scale coupling: correlation between scales
- Resonance: phase coherence across scales

Applications:
- Real-time writing feedback
- Pacing analysis (tension curves)
- Style fingerprinting (fractal signatures)
- Character voice consistency checking
- Genre classification via multi-scale dynamics
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, AsyncIterator, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

from .dimensions import SemanticSpectrum, EXTENDED_244_DIMENSIONS
from .integrator import GeometricIntegrator, SemanticState


class TemporalScale(Enum):
    """Nested temporal scales for analysis."""
    WORD = "word"              # 1-5 tokens
    PHRASE = "phrase"          # 5-15 tokens
    SENTENCE = "sentence"      # 15-50 tokens
    PARAGRAPH = "paragraph"    # 50-200 tokens
    CHUNK = "chunk"            # 200-1000 tokens


@dataclass
class ScaleWindow:
    """Sliding window for a specific temporal scale."""
    scale: TemporalScale
    buffer: deque = field(default_factory=deque)
    max_size: int = 100  # tokens

    # Trajectory tracking
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    velocities: deque = field(default_factory=lambda: deque(maxlen=50))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=50))

    # Statistics
    total_tokens: int = 0
    analysis_count: int = 0
    last_analysis_time: float = 0.0


@dataclass
class ScaleResonance:
    """Measures coupling between temporal scales."""
    scale_pair: Tuple[TemporalScale, TemporalScale]
    position_correlation: float
    velocity_correlation: float
    phase_coherence: float  # 0-1, measures alignment
    coupling_strength: float  # How tightly scales move together

    @property
    def resonance_score(self) -> float:
        """Overall resonance metric."""
        return (self.position_correlation + self.velocity_correlation +
                self.phase_coherence + self.coupling_strength) / 4.0


@dataclass
class MultiScaleSnapshot:
    """Snapshot of semantic state across all scales."""
    timestamp: float
    word_count: int

    # Per-scale states
    states_by_scale: Dict[TemporalScale, Dict[str, np.ndarray]]

    # Cross-scale metrics
    resonances: List[ScaleResonance]
    dominant_dimensions: List[str]  # Top changing dimensions
    narrative_momentum: float  # How aligned are all scales?
    complexity_index: float  # How divergent are scales?

    # Interpretable summary
    semantic_summary: str


class StreamingSemanticCalculus:
    """
    Real-time multi-scale semantic calculus analyzer.

    Processes text word-by-word, tracking semantic trajectories at
    multiple nested temporal scales and measuring their interactions.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        scales: List[TemporalScale] = None,
        enable_visualization: bool = True,
        snapshot_interval: float = 2.0
    ):
        """
        Initialize streaming semantic calculus.

        Args:
            embed_fn: Function that embeds text → 384D vector
            scales: Temporal scales to analyze (default: all)
            enable_visualization: Track data for visualization
            snapshot_interval: Seconds between snapshots
        """
        self.embed_fn = embed_fn
        self.scales = scales or list(TemporalScale)
        self.enable_visualization = enable_visualization
        self.snapshot_interval = snapshot_interval

        # Initialize semantic spectrum (244D projection)
        self.spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
        print("🔧 Learning 244-dimensional semantic axes...")
        self.spectrum.learn_axes(embed_fn)
        print("✅ Semantic axes learned!")

        # Create windows for each scale
        self.windows = {
            TemporalScale.WORD: ScaleWindow(TemporalScale.WORD, max_size=5),
            TemporalScale.PHRASE: ScaleWindow(TemporalScale.PHRASE, max_size=15),
            TemporalScale.SENTENCE: ScaleWindow(TemporalScale.SENTENCE, max_size=50),
            TemporalScale.PARAGRAPH: ScaleWindow(TemporalScale.PARAGRAPH, max_size=200),
            TemporalScale.CHUNK: ScaleWindow(TemporalScale.CHUNK, max_size=1000),
        }

        # Snapshot history
        self.snapshots: List[MultiScaleSnapshot] = []
        self.event_callbacks: List[Callable] = []

        # Global tracking
        self.total_words_processed = 0
        self.start_time = time.time()

    def on_snapshot(self, callback: Callable[[MultiScaleSnapshot], None]):
        """Register callback for snapshot events."""
        self.event_callbacks.append(callback)

    async def stream_analyze(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[MultiScaleSnapshot]:
        """
        Analyze text stream word-by-word with multi-scale calculus.

        Args:
            text_stream: Async iterator yielding words or small text chunks

        Yields:
            MultiScaleSnapshot objects at regular intervals
        """
        last_snapshot_time = time.time()

        async for token in text_stream:
            # Add token to all scale windows
            for window in self.windows.values():
                window.buffer.append(token)
                window.total_tokens += 1

            self.total_words_processed += 1

            # Analyze each scale window
            await self._analyze_all_scales()

            # Check if time for snapshot
            now = time.time()
            if now - last_snapshot_time >= self.snapshot_interval:
                snapshot = await self._create_snapshot()

                # Emit to callbacks
                for callback in self.event_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(snapshot)
                    else:
                        callback(snapshot)

                yield snapshot
                last_snapshot_time = now

        # Final snapshot
        final_snapshot = await self._create_snapshot()
        for callback in self.event_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(final_snapshot)
            else:
                callback(final_snapshot)
        yield final_snapshot

    async def _analyze_all_scales(self):
        """Analyze current state of all scale windows."""
        for scale, window in self.windows.items():
            if len(window.buffer) == 0:
                continue

            # Get text for this scale
            text = " ".join(window.buffer)

            # Embed and project to 244D
            embedding = self.embed_fn(text)
            semantic_position = self.spectrum.project_vector(embedding)
            position_vector = np.array([semantic_position[dim.name]
                                       for dim in EXTENDED_244_DIMENSIONS])

            # Store position
            window.positions.append(position_vector)

            # Compute velocity (if we have history)
            if len(window.positions) >= 2:
                velocity = window.positions[-1] - window.positions[-2]
                window.velocities.append(velocity)

            # Compute acceleration (if we have velocity history)
            if len(window.velocities) >= 2:
                acceleration = window.velocities[-1] - window.velocities[-2]
                window.accelerations.append(acceleration)

            window.analysis_count += 1
            window.last_analysis_time = time.time()

    async def _create_snapshot(self) -> MultiScaleSnapshot:
        """Create multi-scale snapshot of current semantic state."""
        timestamp = time.time()

        # Gather states by scale
        states_by_scale = {}
        for scale, window in self.windows.items():
            if len(window.positions) > 0:
                states_by_scale[scale] = {
                    'position': window.positions[-1] if len(window.positions) > 0 else None,
                    'velocity': window.velocities[-1] if len(window.velocities) > 0 else None,
                    'acceleration': window.accelerations[-1] if len(window.accelerations) > 0 else None,
                }

        # Compute cross-scale resonances
        resonances = self._compute_resonances(states_by_scale)

        # Find dominant dimensions (highest velocity magnitude)
        dominant_dims = self._find_dominant_dimensions(states_by_scale)

        # Compute narrative momentum (how aligned are scales?)
        momentum = self._compute_narrative_momentum(resonances)

        # Compute complexity index (how divergent are scales?)
        complexity = self._compute_complexity_index(states_by_scale)

        # Generate interpretable summary
        summary = self._generate_summary(dominant_dims, momentum, complexity)

        snapshot = MultiScaleSnapshot(
            timestamp=timestamp,
            word_count=self.total_words_processed,
            states_by_scale=states_by_scale,
            resonances=resonances,
            dominant_dimensions=dominant_dims,
            narrative_momentum=momentum,
            complexity_index=complexity,
            semantic_summary=summary
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _compute_resonances(
        self,
        states_by_scale: Dict[TemporalScale, Dict[str, np.ndarray]]
    ) -> List[ScaleResonance]:
        """Compute coupling between all scale pairs."""
        resonances = []
        scales = list(states_by_scale.keys())

        for i, scale1 in enumerate(scales):
            for scale2 in scales[i+1:]:
                state1 = states_by_scale[scale1]
                state2 = states_by_scale[scale2]

                # Skip if either scale lacks data
                if state1['position'] is None or state2['position'] is None:
                    continue

                # Position correlation
                pos_corr = np.corrcoef(state1['position'], state2['position'])[0, 1]

                # Velocity correlation
                vel_corr = 0.0
                if state1['velocity'] is not None and state2['velocity'] is not None:
                    vel_corr = np.corrcoef(state1['velocity'], state2['velocity'])[0, 1]

                # Phase coherence (normalized dot product)
                phase_coherence = 0.0
                if state1['position'] is not None and state2['position'] is not None:
                    p1_norm = state1['position'] / (np.linalg.norm(state1['position']) + 1e-10)
                    p2_norm = state2['position'] / (np.linalg.norm(state2['position']) + 1e-10)
                    phase_coherence = abs(np.dot(p1_norm, p2_norm))

                # Coupling strength (inverse distance in semantic space)
                distance = np.linalg.norm(state1['position'] - state2['position'])
                coupling = 1.0 / (1.0 + distance)

                resonance = ScaleResonance(
                    scale_pair=(scale1, scale2),
                    position_correlation=float(pos_corr) if not np.isnan(pos_corr) else 0.0,
                    velocity_correlation=float(vel_corr) if not np.isnan(vel_corr) else 0.0,
                    phase_coherence=float(phase_coherence),
                    coupling_strength=float(coupling)
                )

                resonances.append(resonance)

        return resonances

    def _find_dominant_dimensions(
        self,
        states_by_scale: Dict[TemporalScale, Dict[str, np.ndarray]],
        top_k: int = 8
    ) -> List[str]:
        """Find dimensions with highest velocity magnitude across all scales."""
        # Aggregate velocities across scales
        all_velocities = []
        for state in states_by_scale.values():
            if state['velocity'] is not None:
                all_velocities.append(state['velocity'])

        if not all_velocities:
            return []

        # Average velocity magnitude per dimension
        avg_velocity = np.mean(np.abs(np.array(all_velocities)), axis=0)

        # Get top K dimensions
        top_indices = np.argsort(avg_velocity)[-top_k:][::-1]

        dimension_names = [dim.name for dim in EXTENDED_244_DIMENSIONS]
        return [dimension_names[i] for i in top_indices]

    def _compute_narrative_momentum(self, resonances: List[ScaleResonance]) -> float:
        """Compute how aligned all scales are (0-1)."""
        if not resonances:
            return 0.0

        # Average resonance score across all scale pairs
        avg_resonance = np.mean([r.resonance_score for r in resonances])
        return float(np.clip(avg_resonance, 0.0, 1.0))

    def _compute_complexity_index(
        self,
        states_by_scale: Dict[TemporalScale, Dict[str, np.ndarray]]
    ) -> float:
        """Compute how divergent scales are (0-1)."""
        positions = [s['position'] for s in states_by_scale.values()
                    if s['position'] is not None]

        if len(positions) < 2:
            return 0.0

        # Compute pairwise distances
        distances = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                dist = np.linalg.norm(p1 - p2)
                distances.append(dist)

        # Normalize by dimensionality (244D)
        avg_distance = np.mean(distances) / np.sqrt(244)
        return float(np.clip(avg_distance, 0.0, 1.0))

    def _generate_summary(
        self,
        dominant_dims: List[str],
        momentum: float,
        complexity: float
    ) -> str:
        """Generate human-readable semantic summary."""
        # Interpret momentum
        if momentum > 0.7:
            momentum_desc = "Strong narrative flow - all scales aligned"
        elif momentum > 0.4:
            momentum_desc = "Moderate flow - scales somewhat aligned"
        else:
            momentum_desc = "Divergent scales - complex/sophisticated"

        # Interpret complexity
        if complexity > 0.7:
            complexity_desc = "High semantic divergence across scales"
        elif complexity > 0.4:
            complexity_desc = "Moderate complexity"
        else:
            complexity_desc = "Simple, coherent structure"

        # Dominant dimensions
        dims_str = ", ".join(dominant_dims[:3]) if dominant_dims else "None"

        return f"{momentum_desc}. {complexity_desc}. Active: {dims_str}."


async def demonstrate_word_by_word():
    """Demonstrate word-by-word multi-scale analysis."""
    print("🌊📐 MULTI-SCALE STREAMING SEMANTIC CALCULUS")
    print("=" * 80)
    print()

    # Simple embedding function for demo (you'd use sentence-transformers in production)
    def simple_embed(text: str) -> np.ndarray:
        """Simple hash-based embedding for demo."""
        # In production, use: model.encode(text)
        np.random.seed(hash(text.lower()) % (2**32))
        return np.random.randn(384)

    # Sample text with clear narrative arc
    text = """
    The startup was dying. Sarah knew it but couldn't admit it. Three years of work,
    evaporating. Her co-founder suggested the impossible: pivot. Start over. The mentor
    called with one question that changed everything. What problem are you really solving?
    That night, Sarah found clarity in the darkness. The rebuild was painful but necessary.
    Six months later, everything clicked. The product found its market. Growth exploded.
    But the real transformation was Sarah herself, from employee to entrepreneur.
    """

    # Create analyzer
    print("Initializing multi-scale analyzer...")
    analyzer = StreamingSemanticCalculus(
        embed_fn=simple_embed,
        snapshot_interval=1.0
    )

    # Setup callback
    def print_snapshot(snapshot: MultiScaleSnapshot):
        print(f"\n📊 SNAPSHOT @ {snapshot.word_count} words")
        print(f"   Momentum: {snapshot.narrative_momentum:.3f} | Complexity: {snapshot.complexity_index:.3f}")
        print(f"   Active: {', '.join(snapshot.dominant_dimensions[:3])}")
        print(f"   {snapshot.semantic_summary}")

        # Show strongest resonance
        if snapshot.resonances:
            top_resonance = max(snapshot.resonances, key=lambda r: r.resonance_score)
            s1, s2 = top_resonance.scale_pair
            print(f"   🔗 Strongest coupling: {s1.value} ↔ {s2.value} ({top_resonance.resonance_score:.3f})")

    analyzer.on_snapshot(print_snapshot)

    # Simulate word-by-word stream
    async def word_stream():
        words = text.split()
        for word in words:
            yield word
            await asyncio.sleep(0.1)  # Simulate typing speed

    print("\n🚀 Starting word-by-word analysis...\n")

    async for snapshot in analyzer.stream_analyze(word_stream()):
        pass  # Snapshots printed via callback

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"   Total words: {analyzer.total_words_processed}")
    print(f"   Total snapshots: {len(analyzer.snapshots)}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_word_by_word())
