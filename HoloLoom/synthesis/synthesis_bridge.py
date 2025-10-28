#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesis Bridge - Connects synthesis modules to weaving cycle
================================================================
Bridges the synthesis pipeline into the weaving orchestrator.

Provides a clean interface for:
- Memory enrichment during weaving
- Pattern extraction from features
- Synthesis insights for decision making

Integration Point: Between ResonanceShed and WarpSpace
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from HoloLoom.synthesis.enriched_memory import EnrichedMemory, MemoryEnricher, ReasoningType
    from HoloLoom.synthesis.pattern_extractor import PatternExtractor, Pattern, PatternType
    from HoloLoom.synthesis.data_synthesizer import DataSynthesizer, SynthesisConfig, TrainingExample
except ImportError as e:
    print(f"Synthesis modules not available: {e}")
    EnrichedMemory = None
    MemoryEnricher = None
    PatternExtractor = None
    DataSynthesizer = None

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """
    Result of synthesis processing during weaving.

    Contains enriched data that can inform decision making.
    """
    # Enrichment
    enriched_query: Optional[EnrichedMemory] = None
    enriched_context: List[EnrichedMemory] = field(default_factory=list)

    # Pattern extraction
    patterns: List[Pattern] = field(default_factory=list)
    pattern_count_by_type: Dict[str, int] = field(default_factory=dict)

    # Synthesis insights
    key_entities: List[str] = field(default_factory=list)
    relationships: List[tuple] = field(default_factory=list)  # (subject, predicate, object)
    topics: List[str] = field(default_factory=list)
    reasoning_type: Optional[str] = None

    # Metadata
    synthesis_duration_ms: float = 0.0
    confidence: float = 0.0

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to dict for Spacetime trace."""
        return {
            'pattern_count': len(self.patterns),
            'pattern_types': self.pattern_count_by_type,
            'entities': self.key_entities[:10],  # Top 10
            'relationships': self.relationships[:5],  # Top 5
            'topics': self.topics[:5],
            'reasoning_type': self.reasoning_type,
            'synthesis_duration_ms': self.synthesis_duration_ms,
            'confidence': self.confidence
        }


class SynthesisBridge:
    """
    Bridge between weaving cycle and synthesis modules.

    Provides a clean interface for synthesis during weaving without
    tightly coupling the orchestrator to synthesis implementation details.

    Usage in WeavingOrchestrator:
        bridge = SynthesisBridge()
        synthesis_result = await bridge.synthesize(
            query_text=query,
            dot_plasma=dot_plasma,
            context_shards=context_shards
        )
    """

    def __init__(
        self,
        enable_enrichment: bool = True,
        enable_pattern_extraction: bool = True,
        min_pattern_confidence: float = 0.4,
        domain_terms: Optional[List[str]] = None
    ):
        """
        Initialize synthesis bridge.

        Args:
            enable_enrichment: Enable memory enrichment
            enable_pattern_extraction: Enable pattern extraction
            min_pattern_confidence: Minimum confidence for patterns
            domain_terms: Optional domain-specific terms for enrichment
        """
        self.enable_enrichment = enable_enrichment
        self.enable_pattern_extraction = enable_pattern_extraction

        # Initialize synthesis modules if available
        if MemoryEnricher:
            self.enricher = MemoryEnricher(domain_terms=domain_terms)
        else:
            self.enricher = None
            logger.warning("MemoryEnricher not available")

        if PatternExtractor:
            self.extractor = PatternExtractor(min_confidence=min_pattern_confidence)
        else:
            self.extractor = None
            logger.warning("PatternExtractor not available")

        if DataSynthesizer:
            synthesis_config = SynthesisConfig(
                min_confidence=min_pattern_confidence,
                include_reasoning=True,
                include_context=True
            )
            self.synthesizer = DataSynthesizer(synthesis_config)
        else:
            self.synthesizer = None
            logger.warning("DataSynthesizer not available")

        logger.info(f"SynthesisBridge initialized (enrichment={enable_enrichment}, patterns={enable_pattern_extraction})")

    async def synthesize(
        self,
        query_text: str,
        dot_plasma: Dict[str, Any],
        context_shards: List[Any],
        pattern_spec: Optional[Any] = None
    ) -> SynthesisResult:
        """
        Main synthesis operation during weaving.

        Takes query + context + features and enriches them with
        synthesis insights.

        Args:
            query_text: User query
            dot_plasma: Features from ResonanceShed
            context_shards: Retrieved context memories
            pattern_spec: Optional pattern card configuration

        Returns:
            SynthesisResult with enriched data
        """
        start_time = datetime.now()
        result = SynthesisResult()

        if not self.enricher:
            logger.warning("Synthesis modules not available, skipping")
            return result

        logger.info(f"Synthesizing insights for query: '{query_text[:50]}...'")

        # ====================================================================
        # STEP 1: Enrich Query
        # ====================================================================
        if self.enable_enrichment:
            try:
                # Create memory object from query
                query_memory = {
                    'id': f'query_{datetime.now().timestamp()}',
                    'text': query_text,
                    'timestamp': datetime.now().isoformat(),
                    'importance': 1.0,  # Query is always important
                    'metadata': {}
                }

                enriched_query = self.enricher.enrich(query_memory)
                result.enriched_query = enriched_query
                result.key_entities = enriched_query.entities
                result.relationships = enriched_query.relationships
                result.topics = enriched_query.topics
                result.reasoning_type = enriched_query.reasoning_type.value

                logger.info(f"  Enriched query: {len(enriched_query.entities)} entities, "
                          f"{len(enriched_query.topics)} topics, "
                          f"type={enriched_query.reasoning_type.value}")

            except Exception as e:
                logger.warning(f"Query enrichment failed: {e}")

        # ====================================================================
        # STEP 2: Enrich Context
        # ====================================================================
        if self.enable_enrichment and context_shards:
            try:
                for idx, shard in enumerate(context_shards[:5]):  # Limit for performance
                    # Convert shard to memory format
                    context_memory = {
                        'id': getattr(shard, 'id', f'context_{idx}'),
                        'text': getattr(shard, 'text', str(shard)),
                        'timestamp': getattr(shard, 'timestamp', datetime.now()).isoformat(),
                        'importance': getattr(shard, 'importance', 0.5),
                        'metadata': getattr(shard, 'metadata', {})
                    }

                    enriched_context = self.enricher.enrich(context_memory)
                    result.enriched_context.append(enriched_context)

                logger.info(f"  Enriched {len(result.enriched_context)} context shards")

            except Exception as e:
                logger.warning(f"Context enrichment failed: {e}")

        # ====================================================================
        # STEP 3: Extract Patterns
        # ====================================================================
        if self.enable_pattern_extraction and self.extractor:
            try:
                # Extract from all enriched memories
                all_enriched = [result.enriched_query] if result.enriched_query else []
                all_enriched.extend(result.enriched_context)

                if all_enriched:
                    patterns = self.extractor.extract_patterns(all_enriched)
                    result.patterns = patterns

                    # Count by type
                    for pattern in patterns:
                        pattern_type = pattern.pattern_type.value
                        result.pattern_count_by_type[pattern_type] = \
                            result.pattern_count_by_type.get(pattern_type, 0) + 1

                    # Calculate average confidence
                    if patterns:
                        result.confidence = sum(p.confidence for p in patterns) / len(patterns)

                    logger.info(f"  Extracted {len(patterns)} patterns: {result.pattern_count_by_type}")

            except Exception as e:
                logger.warning(f"Pattern extraction failed: {e}")

        # ====================================================================
        # Finalize
        # ====================================================================
        end_time = datetime.now()
        result.synthesis_duration_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(f"Synthesis complete: {result.synthesis_duration_ms:.1f}ms, "
                   f"{len(result.patterns)} patterns, "
                   f"confidence={result.confidence:.2f}")

        return result

    def enrich_decision_context(self, synthesis_result: SynthesisResult) -> Dict[str, Any]:
        """
        Create decision context from synthesis results.

        Formats synthesis insights for use in decision making.

        Args:
            synthesis_result: Result from synthesize()

        Returns:
            Dict with decision-relevant synthesis insights
        """
        return {
            'entities': synthesis_result.key_entities,
            'topics': synthesis_result.topics,
            'reasoning_type': synthesis_result.reasoning_type,
            'pattern_count': len(synthesis_result.patterns),
            'pattern_types': list(synthesis_result.pattern_count_by_type.keys()),
            'confidence': synthesis_result.confidence,
            'has_qa_patterns': 'qa_pair' in synthesis_result.pattern_count_by_type,
            'has_reasoning_chains': 'reasoning_chain' in synthesis_result.pattern_count_by_type
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_synthesis_bridge(
    enable_enrichment: bool = True,
    enable_patterns: bool = True,
    min_confidence: float = 0.4,
    domain_terms: Optional[List[str]] = None
) -> SynthesisBridge:
    """
    Create synthesis bridge with configuration.

    Args:
        enable_enrichment: Enable memory enrichment
        enable_patterns: Enable pattern extraction
        min_confidence: Minimum pattern confidence
        domain_terms: Domain-specific terms

    Returns:
        Configured SynthesisBridge
    """
    return SynthesisBridge(
        enable_enrichment=enable_enrichment,
        enable_pattern_extraction=enable_patterns,
        min_pattern_confidence=min_confidence,
        domain_terms=domain_terms
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("SYNTHESIS BRIDGE DEMO")
        print("="*80)

        # Create bridge
        bridge = SynthesisBridge(
            enable_enrichment=True,
            enable_pattern_extraction=True,
            min_pattern_confidence=0.4
        )

        # Mock inputs
        query = "What is Thompson Sampling and how does it balance exploration vs exploitation?"
        dot_plasma = {
            'motifs': ['THOMPSON', 'SAMPLING', 'EXPLORATION'],
            'psi': [0.1] * 384,
            'metadata': {}
        }
        context_shards = []  # Empty for demo

        # Run synthesis
        result = await bridge.synthesize(
            query_text=query,
            dot_plasma=dot_plasma,
            context_shards=context_shards
        )

        # Show results
        print(f"\nSynthesis Results:")
        print(f"  Duration: {result.synthesis_duration_ms:.1f}ms")
        print(f"  Entities: {result.key_entities}")
        print(f"  Topics: {result.topics}")
        print(f"  Reasoning type: {result.reasoning_type}")
        print(f"  Patterns: {len(result.patterns)}")
        print(f"  Pattern types: {result.pattern_count_by_type}")
        print(f"  Confidence: {result.confidence:.2f}")

        # Decision context
        decision_ctx = bridge.enrich_decision_context(result)
        print(f"\nDecision Context:")
        for key, value in decision_ctx.items():
            print(f"  {key}: {value}")

        print("\nDemo complete!")

    asyncio.run(demo())
