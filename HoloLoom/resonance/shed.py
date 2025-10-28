"""
Resonance Shed - Feature Interference Zone
===========================================
The active feature extraction zone where patterns interfere and combine.

Philosophy:
In weaving, the "shed" is the opening created when warp threads are lifted,
allowing the shuttle to pass through. The Resonance Shed is the computational
analogue - where multiple feature extraction processes (motifs, embeddings,
spectral) interfere and resonate to create rich, multi-modal representations.

The shed creates "feature interference patterns" - like light or sound waves
combining to produce constructive/destructive interference. This produces the
DotPlasma - the flowing feature fluid that feeds into decision-making.

Lifecycle:
1. lift() - Activate feature extraction threads (motif, embedding, spectral)
2. interfere() - Combine features through attention/fusion
3. lower() - Collapse to final DotPlasma representation
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Threads
# ============================================================================

@dataclass
class FeatureThread:
    """
    A single thread of feature extraction.

    Represents one modality of feature extraction (e.g., motifs, embeddings).
    """
    name: str  # "motif", "embedding", "spectral"
    features: Any  # Extracted features
    weight: float = 1.0  # Contribution weight (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Resonance Shed
# ============================================================================

class ResonanceShed:
    """
    Feature interference zone for multi-modal extraction.

    The Resonance Shed manages multiple parallel feature extraction processes,
    combining them through interference patterns to create rich representations.

    It lifts threads (activates extractors), lets them interfere (combine features),
    and lowers to produce DotPlasma (final feature representation).

    Components:
    - Motif detection: Pattern recognition (symbolic)
    - Embeddings: Dense semantic vectors (continuous)
    - Spectral: Graph structure features (topological)
    - Semantic flow: Trajectory analysis (velocity, acceleration, curvature)

    Usage:
        shed = ResonanceShed(motif_detector, embedder, spectral_fusion, semantic_calculus)
        dot_plasma = await shed.weave(query_text, context_graph)
    """

    def __init__(
        self,
        motif_detector=None,
        embedder=None,
        spectral_fusion=None,
        semantic_calculus=None,
        interference_mode: str = "weighted_sum",
        max_feature_density: float = 1.0
    ):
        """
        Initialize Resonance Shed.

        Args:
            motif_detector: Optional motif detection module
            embedder: Optional embedding module
            spectral_fusion: Optional spectral feature module
            semantic_calculus: Optional semantic flow calculus module
            interference_mode: How to combine features ("weighted_sum", "attention", "concat")
            max_feature_density: Maximum feature density before pressure relief (0-1)
                                0.85 = shed features when > 85% of extractors active
        """
        self.motif_detector = motif_detector
        self.embedder = embedder
        self.spectral_fusion = spectral_fusion
        self.semantic_calculus = semantic_calculus
        self.interference_mode = interference_mode
        self.max_feature_density = max_feature_density

        # Active threads
        self.threads: List[FeatureThread] = []
        self.is_lifted = False

        # Pressure tracking
        self.current_density = 0.0
        self.pressure_relief_count = 0

        logger.info(f"ResonanceShed initialized (mode={interference_mode}, max_density={max_feature_density:.2f}, semantic_flow={semantic_calculus is not None})")

    async def weave(
        self,
        text: str,
        context_graph=None,
        thread_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Complete weaving cycle: lift → interfere → lower.

        Main API for feature extraction. Performs full shed cycle to produce
        DotPlasma features.

        Args:
            text: Input text to extract features from
            context_graph: Optional context knowledge graph
            thread_weights: Optional weights for each feature thread

        Returns:
            DotPlasma dict with combined features
        """
        logger.info(f"Weaving features for text: '{text[:50]}...'")

        # Lift threads (activate extractors)
        await self.lift(text, context_graph, thread_weights)

        # Interfere (combine features)
        plasma = self.interfere()

        # Lower (finalize)
        self.lower()

        logger.info(f"Woven DotPlasma with {len(self.threads)} feature threads")
        return plasma

    async def lift(
        self,
        text: str,
        context_graph=None,
        thread_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Lift feature extraction threads.

        Activates all available feature extractors in parallel.

        Args:
            text: Input text
            context_graph: Optional context graph for spectral features
            thread_weights: Optional custom weights per thread
        """
        if self.is_lifted:
            logger.warning("Shed already lifted, lowering first")
            self.lower()

        logger.debug("Lifting feature threads...")

        self.threads = []
        default_weights = thread_weights or {}

        # Thread 1: Motif detection (symbolic patterns)
        if self.motif_detector:
            try:
                motifs = await self.motif_detector.detect(text)
                motif_patterns = [m.pattern for m in motifs] if motifs else []

                self.threads.append(FeatureThread(
                    name="motif",
                    features=motif_patterns,
                    weight=default_weights.get("motif", 1.0),
                    metadata={"count": len(motif_patterns)}
                ))
                logger.debug(f"  Motif thread lifted: {len(motif_patterns)} patterns")
            except Exception as e:
                logger.warning(f"Motif extraction failed: {e}")

        # Thread 2: Embeddings (continuous semantic)
        if self.embedder:
            try:
                # embedder.encode() is synchronous, not async
                embeddings = self.embedder.encode([text])
                embedding = embeddings[0] if len(embeddings) > 0 else []

                self.threads.append(FeatureThread(
                    name="embedding",
                    features=embedding,
                    weight=default_weights.get("embedding", 1.0),
                    metadata={"dim": len(embedding) if hasattr(embedding, '__len__') else 0}
                ))
                logger.debug(f"  Embedding thread lifted: dim={len(embedding)}")
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")

        # Thread 3: Spectral features (graph topology)
        if self.spectral_fusion and context_graph is not None:
            try:
                spectral, metrics = await self.spectral_fusion.features(
                    context_graph,
                    [text],
                    self.embedder
                )

                self.threads.append(FeatureThread(
                    name="spectral",
                    features=spectral,
                    weight=default_weights.get("spectral", 1.0),
                    metadata={"metrics": metrics}
                ))
                logger.debug(f"  Spectral thread lifted: {metrics}")
            except Exception as e:
                logger.warning(f"Spectral extraction failed: {e}")

        # Thread 4: Semantic flow (trajectory analysis)
        if self.semantic_calculus:
            try:
                # Check if semantic_calculus is a SemanticAnalyzer (new organized structure)
                # or legacy SemanticFlowCalculus (backward compatible)
                from HoloLoom.semantic_calculus.analyzer import SemanticAnalyzer

                if isinstance(self.semantic_calculus, SemanticAnalyzer):
                    # New integration layer - clean interface
                    semantic_features = self.semantic_calculus.extract_features(text)
                else:
                    # Legacy SemanticFlowCalculus - direct usage (backward compatible)
                    words = text.split()
                    trajectory = self.semantic_calculus.compute_trajectory(words)
                    semantic_features = {
                        "trajectory": trajectory,
                        "avg_velocity": float(np.mean([s.speed for s in trajectory.states])),
                        "avg_acceleration": float(np.mean([s.acceleration_magnitude for s in trajectory.states])),
                        "curvature": [trajectory.curvature(i) for i in range(len(trajectory.states))],
                        "total_distance": float(trajectory.total_distance()),
                        "n_states": len(trajectory.states)
                    }

                self.threads.append(FeatureThread(
                    name="semantic_flow",
                    features=semantic_features,
                    weight=default_weights.get("semantic_flow", 1.0),
                    metadata={
                        "n_words": semantic_features.get('n_states', 0),
                        "avg_speed": semantic_features.get("avg_velocity", 0.0),
                        "avg_acceleration": semantic_features.get("avg_acceleration", 0.0)
                    }
                ))
                logger.debug(f"  Semantic flow thread lifted: {semantic_features.get('n_states', 0)} words")
            except Exception as e:
                logger.warning(f"Semantic flow extraction failed: {e}")

        self.is_lifted = True

        # Calculate feature density
        max_threads = sum([
            self.motif_detector is not None,
            self.embedder is not None,
            self.spectral_fusion is not None,
            self.semantic_calculus is not None
        ])
        self.current_density = len(self.threads) / max_threads if max_threads > 0 else 0.0

        # Apply pressure relief if needed
        if self.current_density > self.max_feature_density:
            self._apply_pressure_relief()

        logger.info(f"Lifted {len(self.threads)} feature threads (density={self.current_density:.2f})")

    def interfere(self) -> Dict[str, Any]:
        """
        Create feature interference patterns.

        Combines multiple feature threads through the configured interference mode.

        Returns:
            DotPlasma dict with combined features
        """
        if not self.is_lifted:
            logger.warning("Cannot interfere: shed not lifted")
            return {}

        logger.debug(f"Creating feature interference ({self.interference_mode})...")

        # Extract features by type
        motif_thread = next((t for t in self.threads if t.name == "motif"), None)
        embedding_thread = next((t for t in self.threads if t.name == "embedding"), None)
        spectral_thread = next((t for t in self.threads if t.name == "spectral"), None)
        semantic_flow_thread = next((t for t in self.threads if t.name == "semantic_flow"), None)

        # Build DotPlasma
        plasma = {
            "motifs": motif_thread.features if motif_thread else [],
            "psi": embedding_thread.features if embedding_thread else [],
            "spectral": spectral_thread.features if spectral_thread else None,
            "semantic_flow": semantic_flow_thread.features if semantic_flow_thread else None,
            "metadata": {
                "thread_count": len(self.threads),
                "interference_mode": self.interference_mode,
                "thread_weights": {t.name: t.weight for t in self.threads}
            }
        }

        # Apply interference (fusion)
        if self.interference_mode == "weighted_sum" and len(self.threads) > 1:
            # Combine embeddings with weights (if multiple embedding-like features exist)
            # For now, just use primary embedding
            pass

        elif self.interference_mode == "attention":
            # Cross-attention between features (future enhancement)
            plasma["metadata"]["attention_applied"] = True

        elif self.interference_mode == "concat":
            # Concatenate all feature vectors (future enhancement)
            plasma["metadata"]["concatenated"] = True

        # Add thread metadata
        plasma["threads"] = [
            {
                "name": t.name,
                "weight": t.weight,
                "metadata": t.metadata
            }
            for t in self.threads
        ]

        return plasma

    def _apply_pressure_relief(self) -> None:
        """
        Apply pressure relief by shedding lowest-weight threads.

        When feature density exceeds max_feature_density, the system is overloaded.
        This method "exhales" excess features by dropping the weakest threads,
        allowing the system to breathe.

        Keeps only enough threads to meet max_feature_density threshold.
        """
        if not self.threads:
            return

        target_count = max(1, int(len(self.threads) * self.max_feature_density))

        if target_count >= len(self.threads):
            return  # No relief needed

        # Sort threads by weight (descending)
        self.threads.sort(key=lambda t: t.weight, reverse=True)

        # Keep only top threads
        dropped = self.threads[target_count:]
        self.threads = self.threads[:target_count]

        self.pressure_relief_count += 1

        logger.warning(
            f"PRESSURE RELIEF: Shed {len(dropped)} threads "
            f"(density {self.current_density:.2f} > {self.max_feature_density:.2f}). "
            f"Dropped: {[t.name for t in dropped]}"
        )

        # Update density
        max_threads = 4  # motif, embedding, spectral, semantic_flow
        self.current_density = len(self.threads) / max_threads

    def lower(self) -> None:
        """
        Lower the shed (deactivate).

        Clears active threads and resets state.
        """
        if not self.is_lifted:
            return

        logger.debug(f"Lowering shed ({len(self.threads)} threads)")

        self.threads = []
        self.is_lifted = False
        self.current_density = 0.0

    def get_trace(self) -> Dict[str, Any]:
        """
        Get extraction trace without lowering.

        Returns:
            Dict with current shed state
        """
        return {
            "is_lifted": self.is_lifted,
            "thread_count": len(self.threads),
            "threads": [
                {
                    "name": t.name,
                    "weight": t.weight,
                    "has_features": t.features is not None,
                    "metadata": t.metadata
                }
                for t in self.threads
            ],
            "interference_mode": self.interference_mode
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_resonance_shed(
    motif_detector=None,
    embedder=None,
    spectral_fusion=None,
    semantic_calculus=None,
    mode: str = "weighted_sum"
) -> ResonanceShed:
    """
    Create Resonance Shed with specified extractors.

    Args:
        motif_detector: Motif detection module
        embedder: Embedding module
        spectral_fusion: Spectral feature module
        semantic_calculus: Semantic flow calculus module
        mode: Interference mode

    Returns:
        Configured ResonanceShed
    """
    return ResonanceShed(
        motif_detector=motif_detector,
        embedder=embedder,
        spectral_fusion=spectral_fusion,
        semantic_calculus=semantic_calculus,
        interference_mode=mode
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Resonance Shed Demo")
        print("="*80 + "\n")

        # Mock extractors
        class MockMotifDetector:
            async def detect(self, text):
                from holoLoom.documentation.types import Motif
                return [
                    Motif(pattern="ALGORITHM", span=(0, 10), score=0.9),
                    Motif(pattern="OPTIMIZATION", span=(10, 20), score=0.8)
                ]

        class MockEmbedder:
            async def encode(self, texts):
                return [np.random.randn(384).tolist() for _ in texts]

        # Create shed with mock extractors
        shed = ResonanceShed(
            motif_detector=MockMotifDetector(),
            embedder=MockEmbedder(),
            interference_mode="weighted_sum"
        )

        # Weave features
        text = "Thompson Sampling balances exploration and exploitation"
        plasma = await shed.weave(
            text,
            thread_weights={"motif": 0.8, "embedding": 1.0}
        )

        # Display results
        print("DotPlasma Features:")
        print(f"  Motifs: {plasma['motifs']}")
        print(f"  Embedding dim: {len(plasma['psi'])}")
        print(f"  Threads: {plasma['metadata']['thread_count']}")
        print(f"  Interference mode: {plasma['metadata']['interference_mode']}")
        print()

        print("Thread Details:")
        for thread in plasma['threads']:
            print(f"  {thread['name']}: weight={thread['weight']}, {thread['metadata']}")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
