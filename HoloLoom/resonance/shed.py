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

from HoloLoom.alignment.safety_guardrails import (
    ActionCategory,
    ActionRequest,
    SafetyDecision,
    SafetyGuardrails,
    create_guardrails,
)

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
        max_feature_density: float = 1.0,
        target_scale: Optional[int] = None,
        guardrails: Optional[SafetyGuardrails] = None,
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
            target_scale: Target embedding scale (for matryoshka embeddings)
        """
        self.motif_detector = motif_detector
        self.embedder = embedder
        self.spectral_fusion = spectral_fusion
        self.semantic_calculus = semantic_calculus
        self.interference_mode = interference_mode
        self.max_feature_density = max_feature_density
        self.target_scale = target_scale
        self.guardrails = guardrails or create_guardrails()
        self.guardrails_enabled = self.guardrails is not None

        # Active threads
        self.threads: List[FeatureThread] = []
        self.is_lifted = False

        # Pressure tracking
        self.current_density = 0.0
        self.pressure_relief_count = 0

        # Guardrail tracking
        self._guardrail_decisions: Dict[str, Optional[SafetyDecision]] = {
            "weave": None,
            "lift": None,
            "interfere": None,
        }
        self._last_guardrail_snapshot: Dict[str, Optional[SafetyDecision]] = {}
        self._last_text_input: str = ""

        logger.info(f"ResonanceShed initialized (mode={interference_mode}, max_density={max_feature_density:.2f}, semantic_flow={semantic_calculus is not None})")

    @staticmethod
    def _decision_to_dict(decision: Optional[SafetyDecision]) -> Optional[Dict[str, Any]]:
        return decision.to_dict() if decision else None

    def _evaluate_guardrails(
        self,
        stage: str,
        *,
        action: str,
        category: ActionCategory,
        context: Dict[str, Any],
        text_input: str = "",
    ) -> Optional[SafetyDecision]:
        if not self.guardrails_enabled:
            return None

        request = ActionRequest(
            action=action,
            category=category,
            context=context,
        )

        decision = self.guardrails.evaluate(request, text_input=text_input or "")

        if not decision.allowed:
            logger.warning("Resonance guardrails blocked action '%s': %s", action, decision.reason)
            raise PermissionError(decision.reason)

        if decision.requires_approval:
            logger.warning(
                "Resonance guardrails require approval for action '%s': %s",
                action,
                decision.reason,
            )
            raise PermissionError(f"Feature extraction requires approval: {decision.reason}")

        self._guardrail_decisions[stage] = decision
        return decision

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

        self._last_text_input = text
        guardrail_context = {
            "interference_mode": self.interference_mode,
            "has_context_graph": context_graph is not None,
            "thread_overrides": list((thread_weights or {}).keys()),
        }
        self._evaluate_guardrails(
            stage="weave",
            action="feature_weave",
            category=ActionCategory.ANALYSIS,
            context=guardrail_context,
            text_input=text,
        )

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

        self._evaluate_guardrails(
            stage="lift",
            action="feature_lift",
            category=ActionCategory.ANALYSIS,
            context={
                "has_motif_detector": bool(self.motif_detector),
                "has_embedder": bool(self.embedder),
                "has_spectral": bool(self.spectral_fusion and context_graph is not None),
                "has_semantic_flow": bool(self.semantic_calculus),
            },
            text_input=text,
        )

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
                # Phase 5: Use matryoshka scale-aware encoding for dimensional alignment
                # This ensures DotPlasma embeddings match policy's expected dimension
                # (e.g., 96d for BARE, 192d for FAST, 384d for FUSED)
                if self.target_scale and hasattr(self.embedder, 'encode_scales'):
                    embeddings = self.embedder.encode_scales([text], size=self.target_scale)
                    logger.debug(f"  Using encode_scales with size={self.target_scale}")
                else:
                    # Fallback: Use standard encode() (returns full dimension)
                    embeddings = self.embedder.encode([text])
                    logger.debug(f"  Using standard encode() (full dimension)")

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

        self._evaluate_guardrails(
            stage="interfere",
            action="feature_interfere",
            category=ActionCategory.ANALYSIS,
            context={
                "thread_count": len(self.threads),
                "interference_mode": self.interference_mode,
            },
            text_input=self._last_text_input,
        )

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
            # Weighted fusion: combine embeddings with thread weights
            plasma = self._fuse_weighted_sum(plasma, motif_thread, embedding_thread, spectral_thread, semantic_flow_thread)

        elif self.interference_mode == "attention":
            # Attention-based fusion: cross-attention between modalities
            plasma = self._fuse_attention(plasma, motif_thread, embedding_thread, spectral_thread, semantic_flow_thread)
            plasma["metadata"]["attention_applied"] = True

        elif self.interference_mode == "concat":
            # Concatenation fusion: stack all feature vectors
            plasma = self._fuse_concat(plasma, motif_thread, embedding_thread, spectral_thread, semantic_flow_thread)
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

        guardrail_metadata = {
            stage: self._decision_to_dict(decision)
            for stage, decision in self._guardrail_decisions.items()
            if decision is not None
        }
        if guardrail_metadata:
            plasma["metadata"]["guardrails"] = guardrail_metadata

        self._last_guardrail_snapshot = self._guardrail_decisions.copy()

        return plasma

    def _fuse_weighted_sum(
        self,
        plasma: Dict[str, Any],
        motif_thread: Optional[FeatureThread],
        embedding_thread: Optional[FeatureThread],
        spectral_thread: Optional[FeatureThread],
        semantic_flow_thread: Optional[FeatureThread]
    ) -> Dict[str, Any]:
        """
        Weighted sum fusion: Combine features using thread weights.

        Strategy:
        - Primary embedding is kept as-is (psi)
        - If spectral features exist, blend with embedding using weights
        - Semantic flow features are kept separate (trajectory data)
        - Motifs remain symbolic (not fused into psi)

        Args:
            plasma: Base DotPlasma dict
            motif_thread: Motif feature thread
            embedding_thread: Embedding feature thread
            spectral_thread: Spectral feature thread
            semantic_flow_thread: Semantic flow feature thread

        Returns:
            Enhanced plasma with weighted fusion applied
        """
        # If we have both embedding and spectral, blend them
        if embedding_thread and spectral_thread:
            emb_features = np.array(embedding_thread.features)
            spec_features = np.array(spectral_thread.features)

            # Normalize weights
            emb_weight = embedding_thread.weight
            spec_weight = spectral_thread.weight
            total_weight = emb_weight + spec_weight
            emb_weight /= total_weight
            spec_weight /= total_weight

            # Pad spectral to match embedding dimension
            if len(spec_features) < len(emb_features):
                spec_features = np.pad(
                    spec_features,
                    (0, len(emb_features) - len(spec_features))
                )
            elif len(spec_features) > len(emb_features):
                # Truncate spectral
                spec_features = spec_features[:len(emb_features)]

            # Weighted blend
            fused_psi = emb_weight * emb_features + spec_weight * spec_features

            plasma["psi"] = fused_psi.tolist()
            plasma["metadata"]["fusion_weights"] = {
                "embedding": float(emb_weight),
                "spectral": float(spec_weight)
            }

        return plasma

    def _fuse_attention(
        self,
        plasma: Dict[str, Any],
        motif_thread: Optional[FeatureThread],
        embedding_thread: Optional[FeatureThread],
        spectral_thread: Optional[FeatureThread],
        semantic_flow_thread: Optional[FeatureThread]
    ) -> Dict[str, Any]:
        """
        Attention-based fusion: Cross-attention between modalities.

        Strategy:
        - Compute attention weights between embedding and other modalities
        - Use softmax to normalize attention scores
        - Combine features using attention-weighted sum
        - Results in richer, context-aware representations

        Formula:
            score(query, key) = query · key / sqrt(dim)
            attention = softmax(scores)
            output = sum(attention[i] * value[i])

        Args:
            plasma: Base DotPlasma dict
            motif_thread: Motif feature thread
            embedding_thread: Embedding feature thread
            spectral_thread: Spectral feature thread
            semantic_flow_thread: Semantic flow feature thread

        Returns:
            Enhanced plasma with attention fusion applied
        """
        if not embedding_thread:
            # Can't do attention without query (embedding)
            return plasma

        query = np.array(embedding_thread.features)
        keys = []
        values = []
        modality_names = []

        # Collect keys/values from available modalities
        if spectral_thread:
            spec_features = np.array(spectral_thread.features)
            # Pad/truncate to match query dimension
            if len(spec_features) < len(query):
                spec_features = np.pad(
                    spec_features,
                    (0, len(query) - len(spec_features))
                )
            elif len(spec_features) > len(query):
                spec_features = spec_features[:len(query)]

            keys.append(spec_features)
            values.append(spec_features)
            modality_names.append("spectral")

        if semantic_flow_thread:
            # Extract numeric features from semantic flow
            sem_features = semantic_flow_thread.features
            if isinstance(sem_features, dict):
                # Convert dict to vector
                sem_vec = np.array([
                    sem_features.get("avg_velocity", 0.0),
                    sem_features.get("avg_acceleration", 0.0),
                    sem_features.get("total_distance", 0.0),
                    float(sem_features.get("n_states", 0))
                ])

                # Pad to match query dimension
                if len(sem_vec) < len(query):
                    sem_vec = np.pad(
                        sem_vec,
                        (0, len(query) - len(sem_vec))
                    )
                elif len(sem_vec) > len(query):
                    sem_vec = sem_vec[:len(query)]

                keys.append(sem_vec)
                values.append(sem_vec)
                modality_names.append("semantic_flow")

        # Add embedding itself as a key/value
        keys.append(query)
        values.append(query)
        modality_names.append("embedding")

        if len(keys) <= 1:
            # Not enough modalities for attention
            return plasma

        # Compute attention scores
        keys_matrix = np.stack(keys)  # (n_modalities, dim)

        # Scaled dot-product attention
        scores = np.dot(keys_matrix, query) / np.sqrt(len(query))

        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)

        # Weighted combination
        values_matrix = np.stack(values)
        fused_psi = np.sum(
            attention_weights[:, np.newaxis] * values_matrix,
            axis=0
        )

        plasma["psi"] = fused_psi.tolist()
        plasma["metadata"]["attention_weights"] = {
            name: float(weight)
            for name, weight in zip(modality_names, attention_weights)
        }

        return plasma

    def _fuse_concat(
        self,
        plasma: Dict[str, Any],
        motif_thread: Optional[FeatureThread],
        embedding_thread: Optional[FeatureThread],
        spectral_thread: Optional[FeatureThread],
        semantic_flow_thread: Optional[FeatureThread]
    ) -> Dict[str, Any]:
        """
        Concatenation fusion: Stack all feature vectors.

        Strategy:
        - Concatenate all numeric feature vectors
        - Results in high-dimensional representation
        - Preserves all information from each modality
        - Higher memory/compute cost

        Order: [embedding, spectral, semantic_flow_numeric]

        Args:
            plasma: Base DotPlasma dict
            motif_thread: Motif feature thread
            embedding_thread: Embedding feature thread
            spectral_thread: Spectral feature thread
            semantic_flow_thread: Semantic flow feature thread

        Returns:
            Enhanced plasma with concatenation fusion applied
        """
        feature_parts = []
        dimension_map = {}
        current_offset = 0

        # Part 1: Embedding features
        if embedding_thread:
            emb_features = np.array(embedding_thread.features)
            feature_parts.append(emb_features)
            dimension_map["embedding"] = (current_offset, current_offset + len(emb_features))
            current_offset += len(emb_features)

        # Part 2: Spectral features
        if spectral_thread:
            spec_features = np.array(spectral_thread.features)
            feature_parts.append(spec_features)
            dimension_map["spectral"] = (current_offset, current_offset + len(spec_features))
            current_offset += len(spec_features)

        # Part 3: Semantic flow features (numeric only)
        if semantic_flow_thread:
            sem_features = semantic_flow_thread.features
            if isinstance(sem_features, dict):
                # Convert dict to vector
                sem_vec = np.array([
                    sem_features.get("avg_velocity", 0.0),
                    sem_features.get("avg_acceleration", 0.0),
                    sem_features.get("total_distance", 0.0),
                    float(sem_features.get("n_states", 0))
                ])
                feature_parts.append(sem_vec)
                dimension_map["semantic_flow"] = (current_offset, current_offset + len(sem_vec))
                current_offset += len(sem_vec)

        # Concatenate all parts
        if feature_parts:
            fused_psi = np.concatenate(feature_parts)
            plasma["psi"] = fused_psi.tolist()
            plasma["metadata"]["concat_dimensions"] = dimension_map
            plasma["metadata"]["total_dim"] = len(fused_psi)

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
        self._guardrail_decisions = {stage: None for stage in self._guardrail_decisions}

    def get_trace(self) -> Dict[str, Any]:
        """
        Get extraction trace without lowering.

        Returns:
            Dict with current shed state
        """
        guardrail_source = self._guardrail_decisions if self.is_lifted else self._last_guardrail_snapshot
        guardrail_metadata = {
            stage: self._decision_to_dict(decision)
            for stage, decision in (guardrail_source or {}).items()
            if decision is not None
        }

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
            "interference_mode": self.interference_mode,
            "guardrails": guardrail_metadata or None,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_resonance_shed(
    motif_detector=None,
    embedder=None,
    spectral_fusion=None,
    semantic_calculus=None,
    mode: str = "weighted_sum",
    guardrails: Optional[SafetyGuardrails] = None,
) -> ResonanceShed:
    """
    Create Resonance Shed with specified extractors.

    Args:
        motif_detector: Motif detection module
        embedder: Embedding module
        spectral_fusion: Spectral feature module
        semantic_calculus: Semantic flow calculus module
        mode: Interference mode
        guardrails: Shared safety guardrails instance

    Returns:
        Configured ResonanceShed
    """
    return ResonanceShed(
        motif_detector=motif_detector,
        embedder=embedder,
        spectral_fusion=spectral_fusion,
        semantic_calculus=semantic_calculus,
        interference_mode=mode,
        guardrails=guardrails,
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
                from HoloLoom.documentation.types import Motif
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
