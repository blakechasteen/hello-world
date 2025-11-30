"""
Feature Extraction Stage
========================
Step 4 of the weaving cycle: Resonance Shed lifts feature threads and creates DotPlasma.

This stage extracts multi-modal features including:
- Motifs (symbolic patterns)
- Embeddings (multi-scale vectors)
- Spectral features (graph topology)
- Semantic calculus (if enabled)
"""

import logging
import time
from typing import Dict, Any, Optional, List

from HoloLoom.protocols.types import Query, Features
from HoloLoom.resonance.shed import ResonanceShed
from HoloLoom.motif.base import create_motif_detector
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
from HoloLoom.loom.command import PatternSpec
from HoloLoom.weaving.protocols import StageResult


class FeatureExtractionStage:
    """
    Feature Extraction Stage (Step 4: Resonance Shed).

    Extracts and fuses multi-modal features to create DotPlasma,
    the flowing feature representation used for decision making.
    """

    def __init__(
        self,
        config: Any,
        linguistic_gate: Optional[Any] = None,
        guardrails: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feature extraction stage.

        Args:
            config: Configuration object
            linguistic_gate: Optional linguistic processing gate
            guardrails: Optional safety guardrails
            logger: Logger instance
        """
        self.config = config
        self.linguistic_gate = linguistic_gate
        self.guardrails = guardrails
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        query: Query,
        context: Dict[str, Any],
        pattern_spec: PatternSpec,
    ) -> StageResult:
        """
        Execute feature extraction through Resonance Shed.

        Args:
            query: The user query
            context: Shared context
            pattern_spec: Pattern specification for feature extraction

        Returns:
            StageResult with DotPlasma features
        """
        start_time = time.time()

        try:
            # Create components based on pattern spec
            motif_detector = create_motif_detector(mode=pattern_spec.motif_mode)
            spectral_fusion = SpectralFusion() if pattern_spec.enable_spectral else None

            # Create embedder with pattern-specific scales
            pattern_embedder = self._create_embedder(pattern_spec)

            # Create semantic analyzer if enabled
            semantic_calculus = await self._create_semantic_analyzer(pattern_spec, pattern_embedder)

            # Create Resonance Shed
            resonance_shed = ResonanceShed(
                motif_detector=motif_detector,
                embedder=pattern_embedder,
                spectral_fusion=spectral_fusion,
                semantic_calculus=semantic_calculus,
                interference_mode="weighted_sum",
                target_scale=max(pattern_spec.scales),
                guardrails=self.guardrails,
            )

            # Extract features to create DotPlasma
            dot_plasma = await resonance_shed.weave(
                text=query.text,
                context_graph=context.get("knowledge_graph")
            )

            duration_ms = (time.time() - start_time) * 1000

            thread_count = len(dot_plasma.get('threads', []))
            self.logger.info(f"  [4] DotPlasma created with {thread_count} feature threads")

            return StageResult(
                stage_name="feature_extraction",
                duration_ms=duration_ms,
                data={
                    "dot_plasma": dot_plasma,
                    "features": self._extract_features_from_plasma(dot_plasma),
                    "thread_count": thread_count,
                },
                metadata={
                    "motif_mode": pattern_spec.motif_mode,
                    "spectral_enabled": pattern_spec.enable_spectral,
                    "semantic_enabled": pattern_spec.enable_semantic_flow,
                    "linguistic_enabled": self.linguistic_gate is not None,
                }
            )

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return StageResult(
                stage_name="feature_extraction",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                data={}
            )

    def _create_embedder(self, pattern_spec: PatternSpec) -> Any:
        """
        Create the appropriate embedder based on configuration.

        Args:
            pattern_spec: Pattern specification

        Returns:
            Embedder instance
        """
        # Use linguistic gate if available and enabled
        if self.linguistic_gate and self.config.enable_linguistic_gate:
            self.logger.info(
                f"  [4a] Phase 5 Linguistic Gate enabled "
                f"(mode={self.config.linguistic_mode}, cache={self.config.use_compositional_cache})"
            )
            return self.linguistic_gate

        # Use zero-copy embeddings if enabled
        elif self.config.enable_zero_copy_embeddings:
            from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings
            self.logger.info(
                f"  [4a] Zero-copy embeddings enabled "
                f"(cache={self.config.zero_copy_cache_path})"
            )
            return ZeroCopyMatryoshkaEmbeddings(
                sizes=pattern_spec.scales,
                base_model_name=self.config.base_model_name,
                store_path=self.config.zero_copy_cache_path,
                max_cache_size=self.config.zero_copy_cache_size
            )

        # Standard matryoshka embeddings
        else:
            return MatryoshkaEmbeddings(
                sizes=pattern_spec.scales,
                base_model_name=self.config.base_model_name
            )

    async def _create_semantic_analyzer(self, pattern_spec: PatternSpec, embedder: Any) -> Optional[Any]:
        """
        Create semantic analyzer if enabled.

        Args:
            pattern_spec: Pattern specification
            embedder: Embedder instance

        Returns:
            Semantic analyzer or None
        """
        if not pattern_spec.enable_semantic_flow:
            return None

        try:
            from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
            from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

            sem_config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)
            embed_fn = lambda words: embedder.encode(words)

            semantic_analyzer = create_semantic_analyzer(embed_fn, config=sem_config)

            self.logger.info(
                f"  [4a] Semantic analyzer enabled ({sem_config.dimensions}D, "
                f"cache={sem_config.enable_cache})"
            )
            return semantic_analyzer

        except ImportError:
            self.logger.warning("Semantic calculus not available")
            return None

    def _extract_features_from_plasma(self, dot_plasma: Dict[str, Any]) -> Features:
        """
        Extract Features object from DotPlasma.

        Args:
            dot_plasma: The DotPlasma dictionary

        Returns:
            Features object
        """
        # DotPlasma structure should contain features
        # This is a simplified extraction - adapt based on actual structure
        return Features(
            motifs=dot_plasma.get("motifs", []),
            embeddings=dot_plasma.get("embeddings", {}),
            spectral=dot_plasma.get("spectral", {}),
        )

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "feature_extraction"

    def get_required_context_keys(self) -> List[str]:
        """Get required context keys."""
        return []  # Optional: knowledge_graph

    def get_output_context_keys(self) -> List[str]:
        """Get output context keys."""
        return ["dot_plasma", "features"]