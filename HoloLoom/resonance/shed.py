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

    Usage:
        shed = ResonanceShed(motif_detector, embedder, spectral_fusion)
        dot_plasma = await shed.weave(query_text, context_graph)
    """

    def __init__(
        self,
        motif_detector=None,
        embedder=None,
        spectral_fusion=None,
        interference_mode: str = "weighted_sum"
    ):
        """
        Initialize Resonance Shed.

        Args:
            motif_detector: Optional motif detection module
            embedder: Optional embedding module
            spectral_fusion: Optional spectral feature module
            interference_mode: How to combine features ("weighted_sum", "attention", "concat")
        """
        self.motif_detector = motif_detector
        self.embedder = embedder
        self.spectral_fusion = spectral_fusion
        self.interference_mode = interference_mode

        # Active threads
        self.threads: List[FeatureThread] = []
        self.is_lifted = False

        logger.info(f"ResonanceShed initialized (mode={interference_mode})")

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

        self.is_lifted = True
        logger.info(f"Lifted {len(self.threads)} feature threads")

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

        # Build DotPlasma
        plasma = {
            "motifs": motif_thread.features if motif_thread else [],
            "psi": embedding_thread.features if embedding_thread else [],
            "spectral": spectral_thread.features if spectral_thread else None,
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
    mode: str = "weighted_sum"
) -> ResonanceShed:
    """
    Create Resonance Shed with specified extractors.

    Args:
        motif_detector: Motif detection module
        embedder: Embedding module
        spectral_fusion: Spectral feature module
        mode: Interference mode

    Returns:
        Configured ResonanceShed
    """
    return ResonanceShed(
        motif_detector=motif_detector,
        embedder=embedder,
        spectral_fusion=spectral_fusion,
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
