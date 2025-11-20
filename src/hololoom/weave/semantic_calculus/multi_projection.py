#!/usr/bin/env python3
"""
🎨🔮 MULTI-PROJECTION SEMANTIC CALCULUS
======================================
Replace hard-coded 244D semantic dimensions with pluggable projection spaces.

The innovation: MODULAR PROJECTION TARGETS
Instead of always projecting to our 244D semantic dimensions, make the
target space PLUGGABLE so you can project to:

1. Semantic 244D (narrative dimensions)
2. Emotion 48D (affective computing)
3. CLIP 512D (multimodal image-text alignment)
4. CodeBERT 768D (technical content understanding)
5. Domain-specific embeddings (medical, legal, finance)
6. **Multiple simultaneously** (ensemble perspectives!)

Benefits:
- Domain adaptation: Medical text → medical projection space
- Multimodal: Text + images through CLIP projection
- Task-specific: Different projections for different tasks
- Ensemble: Multiple perspectives → richer understanding
- Extensible: Add new projection spaces without changing core code

Architecture:
```python
class ProjectionSpace(Protocol):
    def project(self, embedding: np.ndarray) -> np.ndarray
    def dimension_names(self) -> List[str]
    def dimension_count(self) -> int

# Use any projection space
calculator = MultiProjectionCalculus(
    projections={
        'semantic': SemanticProjection(244),
        'emotion': EmotionProjection(48),
        'clip': CLIPProjection(512)
    }
)

# All projections run simultaneously!
snapshot = await calculator.analyze(text)
# → snapshot has results in ALL projection spaces
```
"""

import sys
import os
# Add repository root to path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import asyncio
import numpy as np
from typing import Dict, List, Optional, AsyncIterator, Callable, Protocol, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum, EXTENDED_244_DIMENSIONS, STANDARD_DIMENSIONS


# ============================================================================
# Protocol: Pluggable Projection Space
# ============================================================================

class ProjectionSpace(Protocol):
    """
    Protocol for projection spaces.

    Any class implementing this can be used as a projection target.
    """

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding to this space.

        Args:
            embedding: Raw embedding vector

        Returns:
            Projected vector in this space
        """
        ...

    def dimension_names(self) -> List[str]:
        """Get human-readable dimension names."""
        ...

    def dimension_count(self) -> int:
        """Get dimensionality of this space."""
        ...

    def interpret(self, projection: np.ndarray, top_k: int = 5) -> str:
        """Generate human-readable interpretation of projection."""
        ...


# ============================================================================
# Concrete Projection Implementations
# ============================================================================

class SemanticProjection:
    """244D narrative semantic dimensions (our original projection)."""

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Initialize semantic projection.

        Args:
            embed_fn: Function to embed exemplar words for learning axes
        """
        self.spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
        print("  🔧 Learning 244D semantic axes...")
        self.spectrum.learn_axes(embed_fn)
        print("  ✅ Semantic axes learned!")

    def project(self, embedding: np.ndarray) -> np.ndarray:
        semantic_dict = self.spectrum.project_vector(embedding)
        return np.array([semantic_dict[dim.name] for dim in self.spectrum.dimensions])

    def dimension_names(self) -> List[str]:
        return [dim.name for dim in self.spectrum.dimensions]

    def dimension_count(self) -> int:
        return len(self.spectrum.dimensions)

    def interpret(self, projection: np.ndarray, top_k: int = 5) -> str:
        top_indices = np.argsort(np.abs(projection))[-top_k:][::-1]
        dim_names = self.dimension_names()
        top_dims = [f"{dim_names[i]} ({projection[i]:.2f})" for i in top_indices]
        return f"Semantic: {', '.join(top_dims)}"


class EmotionProjection:
    """48D emotion-focused projection (Plutchik's wheel + expansions)."""

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        """Initialize emotion projection with core emotional dimensions."""
        # Plutchik's 8 core emotions + intensity variants + blends
        from HoloLoom.semantic_calculus.dimensions import (
            SemanticDimension,
            EMOTIONAL_DEPTH_DIMENSIONS,
            STANDARD_DIMENSIONS
        )

        # Use emotional subset
        emotion_dims = [
            dim for dim in EXTENDED_244_DIMENSIONS
            if any(keyword in dim.name.lower() for keyword in
                   ['emotion', 'joy', 'trust', 'fear', 'surprise', 'sadness',
                    'disgust', 'anger', 'anticipation', 'hope', 'grief',
                    'rage', 'ecstasy', 'dread', 'love', 'hate', 'warmth',
                    'valence', 'arousal', 'shame', 'guilt', 'pride', 'awe',
                    'compassion', 'vulnerability', 'authenticity'])
        ][:48]

        self.spectrum = SemanticSpectrum(dimensions=emotion_dims)
        print("  🔧 Learning 48D emotion axes...")
        self.spectrum.learn_axes(embed_fn)
        print("  ✅ Emotion axes learned!")

    def project(self, embedding: np.ndarray) -> np.ndarray:
        emotional_dict = self.spectrum.project_vector(embedding)
        return np.array([emotional_dict[dim.name] for dim in self.spectrum.dimensions])

    def dimension_names(self) -> List[str]:
        return [dim.name for dim in self.spectrum.dimensions]

    def dimension_count(self) -> int:
        return len(self.spectrum.dimensions)

    def interpret(self, projection: np.ndarray, top_k: int = 3) -> str:
        top_indices = np.argsort(np.abs(projection))[-top_k:][::-1]
        dim_names = self.dimension_names()
        top_emotions = [f"{dim_names[i]} ({projection[i]:.2f})" for i in top_indices]
        return f"Emotion: {', '.join(top_emotions)}"


class ArchetypalProjection:
    """32D archetypal projection (Jungian + Campbell)."""

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        from HoloLoom.semantic_calculus.dimensions import ARCHETYPAL_DIMENSIONS, NARRATIVE_DIMENSIONS

        archetypal_dims = ARCHETYPAL_DIMENSIONS + NARRATIVE_DIMENSIONS
        self.spectrum = SemanticSpectrum(dimensions=archetypal_dims[:32])
        print("  🔧 Learning 32D archetypal axes...")
        self.spectrum.learn_axes(embed_fn)
        print("  ✅ Archetypal axes learned!")

    def project(self, embedding: np.ndarray) -> np.ndarray:
        archetypal_dict = self.spectrum.project_vector(embedding)
        return np.array([archetypal_dict[dim.name] for dim in self.spectrum.dimensions])

    def dimension_names(self) -> List[str]:
        return [dim.name for dim in self.spectrum.dimensions]

    def dimension_count(self) -> int:
        return len(self.spectrum.dimensions)

    def interpret(self, projection: np.ndarray, top_k: int = 3) -> str:
        top_indices = np.argsort(np.abs(projection))[-top_k:][::-1]
        dim_names = self.dimension_names()
        top_archetypes = [f"{dim_names[i]} ({projection[i]:.2f})" for i in top_indices]
        return f"Archetype: {', '.join(top_archetypes)}"


class IdentityProjection:
    """Identity projection (pass-through) for raw embeddings."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def project(self, embedding: np.ndarray) -> np.ndarray:
        # Return embedding as-is (truncated to dimension)
        return embedding[:self.dimension]

    def dimension_names(self) -> List[str]:
        return [f"emb_{i}" for i in range(self.dimension)]

    def dimension_count(self) -> int:
        return self.dimension

    def interpret(self, projection: np.ndarray, top_k: int = 3) -> str:
        magnitude = np.linalg.norm(projection)
        return f"Raw embedding (magnitude: {magnitude:.2f})"


# ============================================================================
# Multi-Projection Calculator
# ============================================================================

@dataclass
class MultiProjectionSnapshot:
    """Snapshot with multiple projection perspectives."""
    timestamp: float
    word_count: int

    # Projections for each space (space_name → projection_vector)
    projections: Dict[str, np.ndarray]

    # Interpretations (space_name → human_readable_string)
    interpretations: Dict[str, str]

    # Cross-projection metrics
    projection_agreement: float  # How much do different projections agree?
    dominant_projection: str  # Which projection shows strongest signal?


class MultiProjectionCalculus:
    """
    Multi-projection streaming semantic calculus.

    Analyzes text through multiple projection spaces simultaneously.
    """

    def __init__(
        self,
        matryoshka_embedder: MatryoshkaEmbeddings,
        projection_spaces: Dict[str, ProjectionSpace],
        snapshot_interval: float = 1.0
    ):
        """
        Initialize multi-projection calculator.

        Args:
            matryoshka_embedder: Embedder for generating base embeddings
            projection_spaces: Dict of {name: projection_space}
            snapshot_interval: Seconds between snapshots
        """
        self.embedder = matryoshka_embedder
        self.projections = projection_spaces
        self.snapshot_interval = snapshot_interval

        # Verify all projections are valid
        for name, proj in self.projections.items():
            if not hasattr(proj, 'project'):
                raise ValueError(f"Projection '{name}' must implement project() method")

        self.snapshots: List[MultiProjectionSnapshot] = []
        self.event_callbacks: List[Callable] = []

        # Tracking
        self.total_words = 0
        self.word_buffer = deque(maxlen=100)

        print(f"✅ Multi-projection calculator initialized with {len(self.projections)} spaces:")
        for name, proj in self.projections.items():
            print(f"   • {name}: {proj.dimension_count()}D")
        print()

    def on_snapshot(self, callback: Callable[[MultiProjectionSnapshot], None]):
        """Register callback for snapshots."""
        self.event_callbacks.append(callback)

    async def stream_analyze(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[MultiProjectionSnapshot]:
        """Stream analyze through all projection spaces."""
        last_snapshot = time.time()

        async for token in text_stream:
            self.word_buffer.append(token)
            self.total_words += 1

            # Check if time for snapshot
            now = time.time()
            if now - last_snapshot >= self.snapshot_interval:
                snapshot = await self._create_snapshot()

                # Emit
                for cb in self.event_callbacks:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(snapshot)
                    else:
                        cb(snapshot)

                yield snapshot
                last_snapshot = now

        # Final
        final = await self._create_snapshot()
        for cb in self.event_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(final)
            else:
                cb(final)
        yield final

    async def _create_snapshot(self) -> MultiProjectionSnapshot:
        """Create snapshot across all projection spaces."""
        if len(self.word_buffer) == 0:
            # Return empty snapshot
            return MultiProjectionSnapshot(
                timestamp=time.time(),
                word_count=self.total_words,
                projections={},
                interpretations={},
                projection_agreement=0.0,
                dominant_projection=""
            )

        # Get text
        text = " ".join(self.word_buffer)

        # Embed once
        embedding = self.embedder.encode_base([text])[0]

        # Project into all spaces
        projections = {}
        interpretations = {}

        for name, proj_space in self.projections.items():
            try:
                projection = proj_space.project(embedding)
                projections[name] = projection
                interpretations[name] = proj_space.interpret(projection)
            except Exception as e:
                print(f"⚠️  Error in projection '{name}': {e}")
                projections[name] = np.zeros(proj_space.dimension_count())
                interpretations[name] = f"Error: {str(e)}"

        # Compute agreement between projections
        agreement = self._compute_agreement(projections)

        # Find dominant projection (highest magnitude)
        dominant = self._find_dominant(projections)

        snapshot = MultiProjectionSnapshot(
            timestamp=time.time(),
            word_count=self.total_words,
            projections=projections,
            interpretations=interpretations,
            projection_agreement=agreement,
            dominant_projection=dominant
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _compute_agreement(self, projections: Dict[str, np.ndarray]) -> float:
        """Compute agreement between different projections."""
        if len(projections) < 2:
            return 1.0

        # Normalize all projections to unit vectors
        normalized = {}
        for name, proj in projections.items():
            norm = np.linalg.norm(proj)
            if norm > 1e-10:
                normalized[name] = proj / norm
            else:
                normalized[name] = proj

        # Compute pairwise cosine similarities
        similarities = []
        names = list(normalized.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                # Pad to same length
                v1 = normalized[name1]
                v2 = normalized[name2]
                max_len = max(len(v1), len(v2))
                v1_padded = np.pad(v1, (0, max_len - len(v1)))
                v2_padded = np.pad(v2, (0, max_len - len(v2)))

                sim = abs(np.dot(v1_padded, v2_padded))
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _find_dominant(self, projections: Dict[str, np.ndarray]) -> str:
        """Find projection with strongest signal (highest magnitude)."""
        if not projections:
            return ""

        magnitudes = {name: np.linalg.norm(proj)
                     for name, proj in projections.items()}

        return max(magnitudes.items(), key=lambda x: x[1])[0]


async def demonstrate_multi_projection():
    """Demonstrate multi-projection analysis."""
    print("🎨🔮 MULTI-PROJECTION SEMANTIC CALCULUS")
    print("=" * 80)
    print("   Analyzing text through multiple projection spaces simultaneously")
    print("=" * 80)
    print()

    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    # Initialize embedder
    print("🔧 Initializing embedder...")
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Create embed function
    def embed_fn(text: str) -> np.ndarray:
        return embedder.encode_base([text])[0]

    # Create multiple projection spaces
    print("\n🔧 Creating projection spaces...")
    projections = {
        'semantic': SemanticProjection(embed_fn),
        'emotion': EmotionProjection(embed_fn),
        'archetype': ArchetypalProjection(embed_fn),
        'raw_96d': IdentityProjection(96),
    }

    # Create calculator
    calculator = MultiProjectionCalculus(
        matryoshka_embedder=embedder,
        projection_spaces=projections,
        snapshot_interval=1.5
    )

    # Sample text with rich narrative + emotion
    text = """
    The hero stood at the threshold, heart pounding with fear and excitement.
    Behind lay comfort and safety. Ahead stretched the unknown, dark and terrifying
    yet somehow calling to her. She thought of her mentor's words: courage is not
    absence of fear, but action despite it. With trembling hands but resolute spirit,
    she stepped forward into her destiny. The old world fell away. A new self emerged.
    """

    def on_snapshot(snapshot: MultiProjectionSnapshot):
        print(f"\n📊 Word {snapshot.word_count}:")
        print(f"   Agreement: {snapshot.projection_agreement:.3f}")
        print(f"   Dominant: {snapshot.dominant_projection}")
        print()

        for space_name, interpretation in snapshot.interpretations.items():
            proj_mag = np.linalg.norm(snapshot.projections[space_name])
            print(f"   {space_name:12} (mag: {proj_mag:6.2f}) | {interpretation}")

    calculator.on_snapshot(on_snapshot)

    # Stream analyze
    print("🚀 Starting multi-projection analysis...\n")

    async def word_stream():
        for word in text.split():
            yield word
            await asyncio.sleep(0.1)

    start_time = time.time()

    async for snapshot in calculator.stream_analyze(word_stream()):
        pass

    duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ MULTI-PROJECTION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"   Duration: {duration:.2f}s")
    print(f"   Words: {calculator.total_words}")
    print(f"   Snapshots: {len(calculator.snapshots)}")
    print()

    # Analyze projection agreement over time
    agreements = [s.projection_agreement for s in calculator.snapshots]
    if agreements:
        print("📈 Projection Agreement Evolution:")
        print(f"   Start: {agreements[0]:.3f}")
        print(f"   End: {agreements[-1]:.3f}")
        print(f"   Average: {np.mean(agreements):.3f}")
        print()

    print("=" * 80)
    print("🎯 KEY INSIGHTS: MULTI-PROJECTION POWER")
    print("=" * 80)
    print("""
    ✅ Modular architecture: Easy to add new projection spaces
    ✅ Simultaneous analysis: All projections computed in parallel
    ✅ Cross-projection agreement: Measures consistency across perspectives
    ✅ Flexible dimensions: 48D emotion, 244D semantic, 32D archetypal, etc.

    Applications:
    • Domain adaptation: Add medical/legal/finance projections
    • Multimodal: Add CLIP projection for image-text alignment
    • Task-specific: Different projections for different tasks
    • Ensemble: Multiple perspectives → richer understanding
    • Interpretability: Each projection offers different insights

    You can now analyze text through ANY lens you want!
    """)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_multi_projection())