"""
Research Synthesis Engine
=========================

Applies DreamEngine synthesis to research papers for cross-paper insights.

**Philosophy**: Papers "talk to each other" - automatic discovery of patterns,
contradictions, and gaps across multiple research papers.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from HoloLoom.synthesis.background_scheduler import (
    BackgroundScheduler,
    SchedulerConfig,
    TriggerStrategy
)
from HoloLoom.synthesis.pattern_synthesis import PatternSynthesizer
from HoloLoom.synthesis.contradiction_detection import ContradictionDetector
from HoloLoom.synthesis.gap_identification import GapIdentifier
from HoloLoom.synthesis.types import (
    SynthesisResults,
    SyntheticMemory,
    Contradiction,
    KnowledgeGap
)


# ============================================================================
# Research Synthesis Engine
# ============================================================================

class ResearchSynthesisEngine:
    """
    Applies DreamEngine synthesis to research papers.

    **Capabilities**:
    - Pattern synthesis: Common themes across papers
    - Contradiction detection: Conflicting claims between papers
    - Gap identification: Missing knowledge in paper collection

    Example:
        >>> engine = ResearchSynthesisEngine()
        >>> await engine.start()
        >>>
        >>> # After papers ingested
        >>> results = await engine.synthesize(knowledge_graph)
        >>>
        >>> # View insights
        >>> patterns = engine.get_patterns()
        >>> contradictions = engine.get_contradictions()
        >>> gaps = engine.get_gaps()
    """

    def __init__(self, auto_synthesis: bool = False):
        """
        Initialize research synthesis engine.

        Args:
            auto_synthesis: Whether to run synthesis automatically after paper ingestion
        """
        # Create synthesis components
        self.pattern_synthesizer = PatternSynthesizer()
        self.contradiction_detector = ContradictionDetector()
        self.gap_identifier = GapIdentifier()

        # Create scheduler (manual trigger by default for research assistant)
        config = SchedulerConfig(
            trigger_strategy=TriggerStrategy.MANUAL,
            enabled=auto_synthesis
        )

        self.scheduler = BackgroundScheduler(
            pattern_synthesizer=self.pattern_synthesizer,
            contradiction_detector=self.contradiction_detector,
            gap_identifier=self.gap_identifier,
            config=config,
            on_cycle_complete=self._on_synthesis_complete
        )

        # Track synthesis history
        self.synthesis_history: List[SynthesisResults] = []
        self.last_synthesis_time: Optional[datetime] = None

    async def start(self):
        """Start synthesis engine (if auto-synthesis enabled)."""
        if self.scheduler.config.enabled:
            await self.scheduler.start()

    async def stop(self):
        """Stop synthesis engine."""
        await self.scheduler.stop()

    async def synthesize(
        self,
        knowledge_graph: Dict[str, Any],
        force: bool = True
    ) -> SynthesisResults:
        """
        Run synthesis cycle on current paper collection.

        Args:
            knowledge_graph: Knowledge graph from PaperMemorySystem
            force: Force synthesis even if minimum thresholds not met

        Returns:
            Synthesis results with patterns, contradictions, gaps

        Example:
            >>> kg = memory.get_knowledge_graph()
            >>> results = await engine.synthesize(kg)
            >>> print(results.summary())
        """
        results = await self.scheduler.trigger_synthesis(
            knowledge_graph=knowledge_graph,
            force=force
        )

        if results:
            self.synthesis_history.append(results)
            self.last_synthesis_time = datetime.now()

        return results

    def register_paper_query(self, query: Dict[str, Any]):
        """
        Register a user query about papers (for pattern detection).

        Args:
            query: Dict with keys:
                - text: Query text
                - confidence: Confidence score (0-1)
                - timestamp: When query occurred
                - entities: Extracted entities (optional)
                - motifs: Extracted motifs (optional)

        Example:
            >>> engine.register_paper_query({
            ...     'text': 'What is the Transformer architecture?',
            ...     'confidence': 0.95,
            ...     'timestamp': datetime.now(),
            ...     'entities': ['Transformer'],
            ...     'motifs': ['architecture']
            ... })
        """
        self.pattern_synthesizer.add_query(query)

    def register_paper_content(self, paper_title: str, content: str):
        """
        Register paper content for contradiction detection.

        Args:
            paper_title: Title of paper
            content: Paper content (section or full text)

        Example:
            >>> engine.register_paper_content(
            ...     "Attention Is All You Need",
            ...     "The Transformer uses self-attention..."
            ... )
        """
        self.contradiction_detector.add_memory({
            'id': paper_title,
            'content': content,
            'timestamp': datetime.now()
        })

    def get_patterns(self) -> List[SyntheticMemory]:
        """
        Get discovered patterns across papers.

        Returns:
            List of synthesized patterns

        Example:
            >>> patterns = engine.get_patterns()
            >>> for p in patterns:
            ...     print(f"Pattern: {p.content}")
            ...     print(f"  Sources: {len(p.source_memories)} papers")
        """
        if self.synthesis_history:
            return self.synthesis_history[-1].patterns_synthesized
        return []

    def get_contradictions(self) -> List[Contradiction]:
        """
        Get contradictions detected between papers.

        Returns:
            List of contradictions

        Example:
            >>> contradictions = engine.get_contradictions()
            >>> for c in contradictions:
            ...     print(f"{c.type.value}: {c.explanation}")
        """
        if self.synthesis_history:
            return self.synthesis_history[-1].contradictions_detected
        return []

    def get_gaps(self) -> List[KnowledgeGap]:
        """
        Get knowledge gaps identified in paper collection.

        Returns:
            List of knowledge gaps

        Example:
            >>> gaps = engine.get_gaps()
            >>> for g in gaps:
            ...     print(f"Gap: {g.bridge_concept}")
            ...     print(f"  Missing: {', '.join(g.missing_concepts)}")
            ...     print(f"  Suggested: {g.suggested_queries[0]}")
        """
        if self.synthesis_history:
            return self.synthesis_history[-1].gaps_identified
        return []

    def get_synthesis_summary(self) -> Dict[str, Any]:
        """
        Get summary of latest synthesis cycle.

        Returns:
            Dict with synthesis statistics

        Example:
            >>> summary = engine.get_synthesis_summary()
            >>> print(f"Patterns: {summary['num_patterns']}")
            >>> print(f"Contradictions: {summary['num_contradictions']}")
            >>> print(f"Gaps: {summary['num_gaps']}")
        """
        if not self.synthesis_history:
            return {
                'num_patterns': 0,
                'num_contradictions': 0,
                'num_gaps': 0,
                'last_synthesis': None,
                'total_cycles': 0
            }

        latest = self.synthesis_history[-1]

        return {
            'num_patterns': len(latest.patterns_synthesized),
            'num_contradictions': len(latest.contradictions_detected),
            'num_gaps': len(latest.gaps_identified),
            'cycle_duration_ms': latest.cycle_duration_ms,
            'last_synthesis': self.last_synthesis_time.isoformat() if self.last_synthesis_time else None,
            'total_cycles': len(self.synthesis_history)
        }

    def get_insights_for_query(self, query: str) -> Dict[str, List[str]]:
        """
        Get relevant insights (patterns/contradictions/gaps) for a query.

        Args:
            query: User query

        Returns:
            Dict with relevant patterns, contradictions, gaps

        Example:
            >>> insights = engine.get_insights_for_query("attention mechanism")
            >>> for pattern in insights['patterns']:
            ...     print(f"  - {pattern}")
        """
        query_lower = query.lower()

        relevant_patterns = []
        relevant_contradictions = []
        relevant_gaps = []

        # Find relevant patterns
        for pattern in self.get_patterns():
            if any(word in pattern.content.lower() for word in query_lower.split()):
                relevant_patterns.append(pattern.content)

        # Find relevant contradictions
        for contradiction in self.get_contradictions():
            if any(word in contradiction.explanation.lower() for word in query_lower.split()):
                relevant_contradictions.append(contradiction.explanation)

        # Find relevant gaps
        for gap in self.get_gaps():
            if any(word in gap.bridge_concept.lower() for word in query_lower.split()):
                gap_str = f"{gap.bridge_concept}: missing {', '.join(gap.missing_concepts[:3])}"
                relevant_gaps.append(gap_str)

        return {
            'patterns': relevant_patterns,
            'contradictions': relevant_contradictions,
            'gaps': relevant_gaps
        }

    def _on_synthesis_complete(self, results: SynthesisResults):
        """Callback when synthesis cycle completes."""
        print(f"[Synthesis] Cycle complete: {len(results.patterns_synthesized)} patterns, "
              f"{len(results.contradictions_detected)} contradictions, "
              f"{len(results.gaps_identified)} gaps")


# ============================================================================
# Utility Functions
# ============================================================================

async def synthesis_example():
    """
    Example workflow: synthesize insights from papers.

    Example:
        >>> await synthesis_example()
    """
    engine = ResearchSynthesisEngine()
    await engine.start()

    # Register some sample queries
    engine.register_paper_query({
        'text': 'What is the Transformer architecture?',
        'confidence': 0.95,
        'timestamp': datetime.now(),
        'entities': ['Transformer'],
        'motifs': ['architecture']
    })

    engine.register_paper_query({
        'text': 'How does self-attention work?',
        'confidence': 0.92,
        'timestamp': datetime.now(),
        'entities': ['self-attention'],
        'motifs': ['mechanism']
    })

    # Register paper content
    engine.register_paper_content(
        "Attention Is All You Need",
        "The Transformer uses self-attention mechanisms..."
    )

    # Create sample knowledge graph
    kg = {
        'nodes': ['Transformer', 'BERT', 'GPT', 'self-attention'],
        'edges': [
            ('BERT', 'Transformer', 'BASED_ON'),
            ('GPT', 'Transformer', 'BASED_ON')
        ],
        'node_counts': {
            'Transformer': 10,
            'BERT': 5,
            'GPT': 5,
            'self-attention': 8
        }
    }

    # Run synthesis
    results = await engine.synthesize(kg)

    print("\n=== Synthesis Results ===")
    print(results.summary())

    await engine.stop()
