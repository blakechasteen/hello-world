"""
Background Memory Consolidation
================================

Based on LangMem research: Background thread converts episodic → semantic memories.

Key Insights from LangMem:
1. "Two-path design" - Hot path (fast queries) + Background path (consolidation)
2. "Sleep-like consolidation" - Async thread extracts semantic facts from episodes
3. "Reduce memory bloat" - Consolidate 100s of episodes into 10s of facts

From Graphiti:
- Deduplicate entities (merge similar entities)
- Extract relationships (entity → entity edges)
- Temporal summarization (summarize time windows)

Architecture:
- Consolidation loop runs every 60 minutes (configurable)
- Extracts semantic facts from recent episodic memories
- Stores facts in AGENT scope (TEMPORARY lifecycle, 30 days)
- Optionally prunes consolidated episodes from SESSION scope

Consolidation Strategies:
1. ENTITY_EXTRACTION - Extract entities and relationships
2. FACT_EXTRACTION - Extract semantic facts
3. SUMMARIZATION - Summarize episodes into summaries
4. DEDUPLICATION - Merge duplicate/similar memories
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from HoloLoom.memory.graph import KG, KGEdge

logger = logging.getLogger(__name__)


# ============================================================================
# Consolidation Types
# ============================================================================

class ConsolidationStrategy(Enum):
    """Consolidation strategy (what to extract from episodes)."""
    ENTITY_EXTRACTION = "entity_extraction"  # Extract entities + relationships
    FACT_EXTRACTION = "fact_extraction"      # Extract semantic facts
    SUMMARIZATION = "summarization"          # Summarize episodes
    DEDUPLICATION = "deduplication"          # Merge duplicates


@dataclass
class ConsolidationResult:
    """Result from consolidation operation."""
    strategy: ConsolidationStrategy
    input_episodes: int
    output_facts: int
    facts_stored: List[str]  # Memory IDs of stored facts
    episodes_pruned: int
    consolidation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LLM Integration (Production - Week 3)
# ============================================================================

# Import production LLM consolidator (Week 3)
try:
    from HoloLoom.memory.llm_consolidator import (
        ProductionLLMConsolidator,
        LLMConfig,
        create_production_consolidator
    )
    PRODUCTION_LLM_AVAILABLE = True
except ImportError:
    logger.warning("Production LLM consolidator not available, using fallback")
    PRODUCTION_LLM_AVAILABLE = False


class LLMConsolidator:
    """
    LLM-based consolidation with production integration (Week 3).

    Week 2: Basic rule-based fallback
    Week 3: Production LLM integration (OpenAI, Anthropic, Ollama, vLLM)

    This class now wraps ProductionLLMConsolidator for backward compatibility.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM consolidator.

        Args:
            llm_provider: "openai", "anthropic", "ollama", "vllm", or None (rule-based)
            llm_model: Model name (None = use provider default)
            api_key: API key (None = read from environment)
        """
        self.llm_provider = llm_provider
        self.available = llm_provider is not None

        # Initialize production consolidator if available
        if PRODUCTION_LLM_AVAILABLE and llm_provider:
            try:
                self.production_consolidator = create_production_consolidator(
                    provider=llm_provider,
                    model=llm_model,
                    api_key=api_key,
                    enable_fallback=True
                )
                logger.info(f"Initialized production LLM consolidator: {llm_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize production consolidator: {e}")
                self.production_consolidator = None
        else:
            self.production_consolidator = None

    async def extract_facts(self, episodes: List[MemoryShard]) -> List[str]:
        """
        Extract semantic facts from episodic memories.

        Week 3: Uses production LLM consolidator if available.

        Args:
            episodes: List of episodic memories

        Returns:
            List of extracted facts (strings)
        """
        # Use production consolidator if available
        if self.production_consolidator:
            return await self.production_consolidator.extract_facts(episodes)

        # Fallback to rule-based (Week 2)
        return await self._extract_facts_fallback(episodes)

    async def _extract_facts_fallback(self, episodes: List[MemoryShard]) -> List[str]:
        """Rule-based fact extraction (Week 2 fallback)."""
        facts = set()
        for episode in episodes:
            sentences = episode.text.split('. ')
            for sentence in sentences:
                if len(sentence) > 20:
                    facts.add(sentence.strip())
        return list(facts)[:10]

    async def extract_entities(
        self,
        episodes: List[MemoryShard]
    ) -> List[tuple[str, str, str]]:
        """
        Extract entity relationships from episodes.

        Week 3: Uses production LLM consolidator if available.

        Args:
            episodes: List of episodic memories

        Returns:
            List of (src, dst, relation_type) tuples
        """
        # Use production consolidator if available
        if self.production_consolidator:
            return await self.production_consolidator.extract_entities(episodes)

        # Fallback to rule-based (Week 2)
        return await self._extract_entities_fallback(episodes)

    async def _extract_entities_fallback(
        self,
        episodes: List[MemoryShard]
    ) -> List[tuple[str, str, str]]:
        """Rule-based entity extraction (Week 2 fallback)."""
        edges = []
        for episode in episodes:
            entities = episode.entities
            for i in range(len(entities) - 1):
                edges.append((entities[i], entities[i + 1], "MENTIONS"))
        return edges[:20]

    async def deduplicate(
        self,
        memories: List[MemoryShard]
    ) -> List[MemoryShard]:
        """
        Deduplicate similar memories.

        Week 3: Uses production LLM consolidator if available.

        Args:
            memories: List of memories to deduplicate

        Returns:
            Deduplicated list (unique memories)
        """
        # Use production consolidator if available
        if self.production_consolidator:
            return await self.production_consolidator.deduplicate(memories)

        # Fallback to rule-based (Week 2)
        return await self._deduplicate_fallback(memories)

    async def _deduplicate_fallback(
        self,
        memories: List[MemoryShard]
    ) -> List[MemoryShard]:
        """Rule-based deduplication (Week 2 fallback)."""
        seen_texts = set()
        unique_memories = []
        for mem in memories:
            if mem.text not in seen_texts:
                seen_texts.add(mem.text)
                unique_memories.append(mem)
        return unique_memories

    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics (Week 3)."""
        if self.production_consolidator:
            return self.production_consolidator.get_statistics()
        return {
            "provider": self.llm_provider or "none",
            "total_requests": 0,
            "total_cost_usd": 0.0
        }


# ============================================================================
# Background Consolidation Engine
# ============================================================================

class MemoryConsolidator:
    """
    Background consolidation: Episodic → Semantic conversion.

    From LangMem: "Background thread runs every 60 minutes to extract semantic facts"

    Architecture:
    1. Consolidation loop runs every N minutes (default: 60)
    2. Fetches recent episodic memories (SESSION scope)
    3. Extracts semantic facts using LLM or rules
    4. Stores facts in AGENT scope (30-day TTL)
    5. Optionally prunes consolidated episodes

    Integration:
    - Uses ContextStreamManager for multi-level memory
    - Uses LLMConsolidator for fact extraction
    - Runs as async background task
    """

    def __init__(
        self,
        stream_manager: ContextStreamManager,
        knowledge_graph: Optional[KG] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        consolidation_interval_minutes: int = 60,
        prune_consolidated_episodes: bool = False
    ):
        """
        Initialize memory consolidator.

        Args:
            stream_manager: Multi-level memory manager
            knowledge_graph: Optional KG for entity extraction
            llm_provider: "openai", "anthropic", "ollama", "vllm", or None (Week 3)
            llm_model: Model name (None = use provider default) (Week 3)
            llm_api_key: API key (None = read from environment) (Week 3)
            consolidation_interval_minutes: How often to consolidate (default: 60)
            prune_consolidated_episodes: Delete episodes after consolidation
        """
        self.stream_manager = stream_manager
        self.kg = knowledge_graph or KG()
        self.llm = LLMConsolidator(llm_provider, llm_model, llm_api_key)
        self.consolidation_interval = timedelta(minutes=consolidation_interval_minutes)
        self.prune_episodes = prune_consolidated_episodes

        # Background task
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.total_consolidations = 0
        self.total_facts_extracted = 0
        self.total_episodes_pruned = 0

    async def start_background_consolidation(self):
        """Start background consolidation loop."""
        if self._running:
            logger.warning("Background consolidation already running")
            return

        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        logger.info(
            f"Started background consolidation (interval: {self.consolidation_interval})"
        )

    async def stop_background_consolidation(self):
        """Stop background consolidation."""
        self._running = False

        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

            logger.info("Stopped background consolidation")

    async def _consolidation_loop(self):
        """
        Background loop: Consolidate episodic memories every N minutes.

        From LangMem: "Background path extracts semantic facts while system is idle"
        """
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval.total_seconds())

                # Run consolidation (statistics updated internally)
                result = await self.consolidate_recent_episodes()

                logger.info(
                    f"Background consolidation complete: "
                    f"episodes={result.input_episodes}, facts={result.output_facts}, "
                    f"pruned={result.episodes_pruned}, time={result.consolidation_time_ms:.1f}ms"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}", exc_info=True)

    async def consolidate_recent_episodes(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours: int = 24
    ) -> ConsolidationResult:
        """
        Consolidate recent episodic memories into semantic facts.

        Args:
            strategy: Consolidation strategy (default: FACT_EXTRACTION)
            lookback_hours: How far back to look for episodes (default: 24)

        Returns:
            ConsolidationResult with facts_stored, episodes_pruned, etc.
        """
        start_time = datetime.now()

        # Get recent episodic memories (SESSION scope)
        session_memories = self.stream_manager.get_all_memories(
            scopes=[MemoryScope.SESSION]
        )

        # Filter by timestamp (last N hours)
        cutoff_time = start_time - timedelta(hours=lookback_hours)
        recent_episodes = [
            m for m in session_memories
            if m.metadata.get("timestamp") and
            datetime.fromisoformat(m.metadata["timestamp"]) > cutoff_time
        ]

        if not recent_episodes:
            logger.info("No recent episodes to consolidate")
            return ConsolidationResult(
                strategy=strategy,
                input_episodes=0,
                output_facts=0,
                facts_stored=[],
                episodes_pruned=0,
                consolidation_time_ms=0.0
            )

        # Apply consolidation strategy
        if strategy == ConsolidationStrategy.FACT_EXTRACTION:
            result = await self._consolidate_facts(recent_episodes)
        elif strategy == ConsolidationStrategy.ENTITY_EXTRACTION:
            result = await self._consolidate_entities(recent_episodes)
        elif strategy == ConsolidationStrategy.SUMMARIZATION:
            result = await self._consolidate_summarization(recent_episodes)
        elif strategy == ConsolidationStrategy.DEDUPLICATION:
            result = await self._consolidate_deduplication(recent_episodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        consolidation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.consolidation_time_ms = consolidation_time_ms

        # Update statistics (whether called directly or from background loop)
        if result.input_episodes > 0:  # Only count consolidations that processed episodes
            self.total_consolidations += 1
            self.total_facts_extracted += result.output_facts
            self.total_episodes_pruned += result.episodes_pruned

        return result

    async def _consolidate_facts(
        self,
        episodes: List[MemoryShard]
    ) -> ConsolidationResult:
        """Extract semantic facts from episodes."""
        # Extract facts using LLM or rules
        fact_texts = await self.llm.extract_facts(episodes)

        # Store facts in AGENT scope (30-day TTL)
        facts_stored = []
        for fact_text in fact_texts:
            fact_id = f"fact_{datetime.now().timestamp()}"
            fact = MemoryShard(
                id=fact_id,
                text=fact_text,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "type": "semantic_fact",
                    "consolidated_from": len(episodes),
                    "importance": 0.8,  # Facts are important
                    "project_id": "consolidated"  # Route to AGENT scope (project_facts)
                }
            )

            # Route to AGENT scope (project_facts stream via project_id metadata)
            stream_name = await self.stream_manager.route_memory(fact)
            facts_stored.append(fact_id)

        # Optionally prune consolidated episodes
        episodes_pruned = 0
        if self.prune_episodes:
            session_stream = self.stream_manager.get_stream("session_state")
            if session_stream:
                episode_ids = {e.id for e in episodes}
                before_count = len(session_stream.memories)
                session_stream.memories = [
                    m for m in session_stream.memories
                    if m.id not in episode_ids
                ]
                episodes_pruned = before_count - len(session_stream.memories)

        return ConsolidationResult(
            strategy=ConsolidationStrategy.FACT_EXTRACTION,
            input_episodes=len(episodes),
            output_facts=len(fact_texts),
            facts_stored=facts_stored,
            episodes_pruned=episodes_pruned,
            consolidation_time_ms=0.0  # Set by caller
        )

    async def _consolidate_entities(
        self,
        episodes: List[MemoryShard]
    ) -> ConsolidationResult:
        """Extract entities and relationships from episodes."""
        # Extract entity edges using LLM or rules
        entity_edges = await self.llm.extract_entities(episodes)

        # Add edges to knowledge graph
        kg_edges = []
        for src, dst, rel_type in entity_edges:
            edge = KGEdge(
                src=src,
                dst=dst,
                type=rel_type,
                weight=1.0,
                metadata={
                    "extracted_from": "consolidation",
                    "episode_count": len(episodes)
                }
            )
            kg_edges.append(edge)

        self.kg.add_edges(kg_edges)

        return ConsolidationResult(
            strategy=ConsolidationStrategy.ENTITY_EXTRACTION,
            input_episodes=len(episodes),
            output_facts=len(entity_edges),
            facts_stored=[],
            episodes_pruned=0,
            consolidation_time_ms=0.0,
            metadata={"edges_added": len(kg_edges)}
        )

    async def _consolidate_summarization(
        self,
        episodes: List[MemoryShard]
    ) -> ConsolidationResult:
        """Summarize episodes into condensed summaries."""
        # Group episodes by time window (e.g., 1-hour windows)
        # For now: Create single summary

        combined_text = " ".join([e.text for e in episodes[:10]])  # Limit to 10
        summary_text = f"Summary of {len(episodes)} episodes: {combined_text[:200]}..."

        summary_id = f"summary_{datetime.now().timestamp()}"
        summary = MemoryShard(
            id=summary_id,
            text=summary_text,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "type": "summary",
                "episodes_summarized": len(episodes)
            }
        )

        await self.stream_manager.route_memory(summary)

        return ConsolidationResult(
            strategy=ConsolidationStrategy.SUMMARIZATION,
            input_episodes=len(episodes),
            output_facts=1,
            facts_stored=[summary_id],
            episodes_pruned=0,
            consolidation_time_ms=0.0
        )

    async def _consolidate_deduplication(
        self,
        episodes: List[MemoryShard]
    ) -> ConsolidationResult:
        """Deduplicate similar episodes."""
        unique_episodes = await self.llm.deduplicate(episodes)

        duplicates_removed = len(episodes) - len(unique_episodes)

        return ConsolidationResult(
            strategy=ConsolidationStrategy.DEDUPLICATION,
            input_episodes=len(episodes),
            output_facts=len(unique_episodes),
            facts_stored=[],
            episodes_pruned=duplicates_removed,
            consolidation_time_ms=0.0,
            metadata={"duplicates_removed": duplicates_removed}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics (Week 3: includes LLM usage)."""
        stats = {
            "total_consolidations": self.total_consolidations,
            "total_facts_extracted": self.total_facts_extracted,
            "total_episodes_pruned": self.total_episodes_pruned,
            "consolidation_interval_minutes": self.consolidation_interval.total_seconds() / 60,
            "prune_episodes_enabled": self.prune_episodes,
            "llm_provider": self.llm.llm_provider
        }

        # Add LLM usage statistics (Week 3)
        llm_stats = self.llm.get_statistics()
        stats["llm_usage"] = llm_stats

        return stats
