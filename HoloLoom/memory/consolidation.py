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

from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.validation import MemoryValidator
from HoloLoom.memory.error_recovery import safe_execute, get_error_aggregator

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


# ============================================================================
# Sleep-Based Memory Consolidation (Human-Like Memory Processing)
# ============================================================================

@dataclass
class ConsolidationConfig:
    """
    Configuration for sleep-based memory consolidation.

    Implements human-like memory processing:
    - Access pattern tracking (frequency, recency)
    - Exponential decay for rarely accessed memories
    - Promotion of frequently accessed memories to long-term storage
    - Archival of contradicted/deprecated memories
    - Sleep-based consolidation during idle periods
    """
    enabled: bool = True
    idle_threshold_hours: float = 24.0  # Trigger consolidation after N hours idle
    decay_rate: float = 0.95  # Daily decay (5% per day)
    promotion_threshold_accesses: int = 5  # Promote if accessed 5+ times
    promotion_window_days: int = 30  # In last 30 days
    archive_threshold: float = 0.1  # Archive if importance < 0.1
    contradiction_detection: bool = True  # Enable contradiction detection


@dataclass
class MemoryAccessStats:
    """Track memory access patterns for consolidation decisions."""
    memory_id: str
    access_count: int = 0
    first_access: Optional[datetime] = None
    last_access: Optional[datetime] = None
    access_times: List[datetime] = field(default_factory=list)
    base_importance: float = 1.0  # Initial importance (0.0-1.0)
    current_importance: float = 1.0  # Decayed importance
    promoted_to_long_term: bool = False
    archived: bool = False
    archive_reason: Optional[str] = None


class SleepBasedConsolidation:
    """
    Human-like sleep-based memory consolidation.

    Mimics human memory processing during sleep:
    1. **Access Pattern Tracking**: Records when memories are accessed
    2. **Decay Algorithm**: Rarely accessed memories lose importance over time
    3. **Promotion**: Frequently accessed memories promoted to long-term storage
    4. **Archival**: Low-importance memories archived (not deleted)
    5. **Sleep Triggers**: Consolidation during idle periods (no queries)

    Integration with HoloLoom:
    - Uses AwarenessGraph for activation tracking
    - Uses LifecycleManager for scope management (SESSION → AGENT → USER)
    - Uses KG for contradiction detection
    - Complements existing LangMem consolidation

    Philosophy:
    "Forgetting is a feature, not a bug. Low-importance memories don't
    disappear—they're archived for forensic analysis, but removed from
    active retrieval."

    Implemented: 2025-11-17
    """

    def __init__(
        self,
        stream_manager: ContextStreamManager,
        knowledge_graph: Optional[KG] = None,
        config: Optional[ConsolidationConfig] = None
    ):
        """
        Initialize sleep-based consolidation.

        Args:
            stream_manager: Multi-level memory manager
            knowledge_graph: Optional KG for contradiction detection
            config: Consolidation configuration
        """
        self.stream_manager = stream_manager
        self.kg = knowledge_graph or KG()
        self.config = config or ConsolidationConfig()

        # Access tracking
        self.access_stats: Dict[str, MemoryAccessStats] = {}

        # Last query timestamp (for idle detection)
        self.last_query_time: Optional[datetime] = None

        # Consolidation statistics
        self.total_consolidations = 0
        self.total_memories_decayed = 0
        self.total_memories_promoted = 0
        self.total_memories_archived = 0

        # Background task
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False

    # =========================================================================
    # Access Pattern Tracking
    # =========================================================================

    def record_access(self, memory_id: str, importance: float = 1.0):
        """
        Record memory access for consolidation decisions.

        Args:
            memory_id: Memory identifier
            importance: Base importance (0.0-1.0)
        """
        now = datetime.now()

        if memory_id not in self.access_stats:
            self.access_stats[memory_id] = MemoryAccessStats(
                memory_id=memory_id,
                first_access=now,
                base_importance=importance
            )

        stats = self.access_stats[memory_id]
        stats.access_count += 1
        stats.last_access = now
        stats.access_times.append(now)

        # Update last query time (for idle detection)
        self.last_query_time = now

        logger.debug(f"Recorded access to memory {memory_id} (count: {stats.access_count})")

    def compute_importance_decay(self, memory_id: str) -> float:
        """
        Compute decayed importance based on time since last access.

        Formula: importance = base_importance × (decay_rate ^ days_since_access)

        Args:
            memory_id: Memory identifier

        Returns:
            Decayed importance (0.0-1.0)
        """
        if memory_id not in self.access_stats:
            return 1.0  # Default: full importance

        stats = self.access_stats[memory_id]

        if stats.last_access is None:
            return stats.base_importance

        # Days since last access
        days_since_access = (datetime.now() - stats.last_access).total_seconds() / 86400.0

        # Exponential decay
        decayed_importance = stats.base_importance * (self.config.decay_rate ** days_since_access)

        # Update current importance
        stats.current_importance = decayed_importance

        return decayed_importance

    # =========================================================================
    # Promotion Algorithm
    # =========================================================================

    async def promote_to_long_term(self, memory_id: str) -> bool:
        """
        Promote frequently accessed memory to long-term storage.

        Promotion criteria:
        - Accessed 5+ times in last 30 days (configurable)
        - Move from SESSION scope → AGENT scope
        - Mark as promoted to prevent re-promotion

        Args:
            memory_id: Memory identifier

        Returns:
            True if promoted, False otherwise
        """
        if memory_id not in self.access_stats:
            logger.warning(f"Cannot promote {memory_id}: No access stats")
            return False

        stats = self.access_stats[memory_id]

        # Already promoted?
        if stats.promoted_to_long_term:
            logger.debug(f"Memory {memory_id} already promoted")
            return False

        # Check promotion criteria
        window_start = datetime.now() - timedelta(days=self.config.promotion_window_days)
        recent_accesses = [
            t for t in stats.access_times
            if t >= window_start
        ]

        if len(recent_accesses) < self.config.promotion_threshold_accesses:
            logger.debug(
                f"Memory {memory_id} does not meet promotion threshold "
                f"({len(recent_accesses)} < {self.config.promotion_threshold_accesses})"
            )
            return False

        # Find memory in SESSION scope
        session_memories = self.stream_manager.get_all_memories(
            scopes=[MemoryScope.SESSION]
        )
        memory = next((m for m in session_memories if m.id == memory_id), None)

        if memory is None:
            logger.warning(f"Memory {memory_id} not found in SESSION scope")
            return False

        # Create promoted copy in AGENT scope (30-day TTL)
        promoted_memory = MemoryShard(
            id=f"{memory_id}_promoted",
            text=memory.text,
            episode=memory.episode,
            entities=memory.entities,
            motifs=memory.motifs,
            scales=memory.scales,
            metadata={
                **memory.metadata,
                "promoted_from": memory_id,
                "promoted_at": datetime.now().isoformat(),
                "access_count": stats.access_count,
                "importance": stats.current_importance,
                "project_id": "promoted",  # Route to AGENT scope
                "timestamp": datetime.now().isoformat()
            }
        )

        # Store in AGENT scope
        await self.stream_manager.route_memory(promoted_memory)

        # Mark as promoted
        stats.promoted_to_long_term = True

        # Update statistics
        self.total_memories_promoted += 1

        logger.info(
            f"Promoted memory {memory_id} to long-term storage "
            f"(accesses: {stats.access_count}, importance: {stats.current_importance:.2f})"
        )

        return True

    # =========================================================================
    # Decay & Archival
    # =========================================================================

    async def decay_memory(self, memory_id: str, decay_factor: float):
        """
        Apply decay to memory importance (manual decay).

        Args:
            memory_id: Memory identifier
            decay_factor: Multiplicative decay (e.g., 0.9 = 10% reduction)
        """
        if memory_id not in self.access_stats:
            logger.warning(f"Cannot decay {memory_id}: No access stats")
            return

        stats = self.access_stats[memory_id]
        stats.base_importance *= decay_factor
        stats.current_importance = self.compute_importance_decay(memory_id)

        self.total_memories_decayed += 1

        logger.debug(
            f"Decayed memory {memory_id} by {decay_factor:.2f} "
            f"(new importance: {stats.current_importance:.3f})"
        )

    async def archive_contradicted(self, memory_id: str, reason: str):
        """
        Archive memory that has been contradicted or deprecated.

        Archival strategy:
        - Move to USER scope with "archived" metadata
        - Mark as archived (not deleted) for forensic analysis
        - Record reason for archival

        Args:
            memory_id: Memory identifier
            reason: Why memory was archived
        """
        if memory_id not in self.access_stats:
            # Initialize stats for archival
            self.access_stats[memory_id] = MemoryAccessStats(
                memory_id=memory_id,
                base_importance=0.0
            )

        stats = self.access_stats[memory_id]

        # Already archived?
        if stats.archived:
            logger.debug(f"Memory {memory_id} already archived")
            return

        # Find memory in any scope
        all_memories = self.stream_manager.get_all_memories(
            scopes=[MemoryScope.SESSION, MemoryScope.AGENT, MemoryScope.USER]
        )
        memory = next((m for m in all_memories if m.id == memory_id), None)

        if memory is None:
            logger.warning(f"Memory {memory_id} not found for archival")
            return

        # Create archived copy in USER scope (permanent storage)
        archived_memory = MemoryShard(
            id=f"{memory_id}_archived",
            text=memory.text,
            episode=memory.episode,
            entities=memory.entities,
            motifs=memory.motifs,
            scales=memory.scales,
            metadata={
                **memory.metadata,
                "archived_from": memory_id,
                "archived_at": datetime.now().isoformat(),
                "archive_reason": reason,
                "archived": True,
                "importance": 0.0,
                "user_id": "archived",  # Route to USER scope
                "timestamp": datetime.now().isoformat()
            }
        )

        # Store in USER scope (PERMANENT lifecycle)
        await self.stream_manager.route_memory(archived_memory)

        # Mark as archived
        stats.archived = True
        stats.archive_reason = reason
        stats.current_importance = 0.0

        # Update statistics
        self.total_memories_archived += 1

        logger.info(f"Archived memory {memory_id}: {reason}")

    # =========================================================================
    # Sleep-Based Consolidation (Idle Detection)
    # =========================================================================

    def is_idle(self) -> bool:
        """
        Check if system has been idle long enough to trigger consolidation.

        Returns:
            True if idle threshold exceeded
        """
        if self.last_query_time is None:
            return True  # Never queried = idle

        hours_since_query = (datetime.now() - self.last_query_time).total_seconds() / 3600.0
        return hours_since_query >= self.config.idle_threshold_hours

    async def consolidate_during_idle(
        self,
        idle_threshold_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Consolidate memories during idle period (mimics human sleep).

        Consolidation operations:
        1. Apply decay to all memories based on time since last access
        2. Promote frequently accessed memories to long-term storage
        3. Archive low-importance memories
        4. Detect and archive contradicted memories (optional)

        Args:
            idle_threshold_hours: Override default idle threshold

        Returns:
            Consolidation statistics
        """
        threshold = idle_threshold_hours or self.config.idle_threshold_hours

        # Check if idle
        if not self.is_idle():
            hours_since = (datetime.now() - self.last_query_time).total_seconds() / 3600.0
            logger.info(
                f"System not idle yet ({hours_since:.1f}h < {threshold:.1f}h threshold)"
            )
            return {
                "consolidated": False,
                "reason": "not_idle",
                "hours_since_last_query": hours_since
            }

        logger.info("Starting sleep-based consolidation (system idle)")

        start_time = datetime.now()

        # Get all memories
        all_memories = self.stream_manager.get_all_memories(
            scopes=[MemoryScope.SESSION, MemoryScope.AGENT]
        )

        decayed_count = 0
        promoted_count = 0
        archived_count = 0
        contradiction_count = 0

        # Phase 1: Apply decay
        logger.info("Phase 1: Applying decay to all memories")
        for memory in all_memories:
            importance = self.compute_importance_decay(memory.id)

            if importance < self.config.archive_threshold:
                # Archive low-importance memory
                await self.archive_contradicted(
                    memory.id,
                    reason=f"Low importance ({importance:.3f} < {self.config.archive_threshold})"
                )
                archived_count += 1
            else:
                decayed_count += 1

        # Phase 2: Promote frequently accessed memories
        logger.info("Phase 2: Promoting frequently accessed memories")
        for memory_id, stats in self.access_stats.items():
            if not stats.promoted_to_long_term and not stats.archived:
                promoted = await self.promote_to_long_term(memory_id)
                if promoted:
                    promoted_count += 1

        # Phase 3: Detect and archive contradictions (optional)
        if self.config.contradiction_detection:
            logger.info("Phase 3: Detecting contradictions")
            contradiction_count = await self._detect_and_archive_contradictions(all_memories)

        consolidation_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update statistics
        self.total_consolidations += 1

        stats = {
            "consolidated": True,
            "total_memories": len(all_memories),
            "decayed_count": decayed_count,
            "promoted_count": promoted_count,
            "archived_count": archived_count,
            "contradiction_count": contradiction_count,
            "consolidation_time_ms": consolidation_time_ms,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Sleep consolidation complete: "
            f"decayed={decayed_count}, promoted={promoted_count}, "
            f"archived={archived_count}, contradictions={contradiction_count}, "
            f"time={consolidation_time_ms:.1f}ms"
        )

        return stats

    async def _detect_and_archive_contradictions(
        self,
        memories: List[MemoryShard]
    ) -> int:
        """
        Detect and archive contradicted memories using knowledge graph.

        Contradiction detection:
        - Check for conflicting edges in knowledge graph
        - Example: (A, B, IS_A) conflicts with (A, C, IS_A) if B != C
        - Archive older memory in favor of newer

        Args:
            memories: List of memories to check

        Returns:
            Number of contradictions detected and archived
        """
        contradiction_count = 0

        # Group memories by entity
        entity_memories: Dict[str, List[MemoryShard]] = {}
        for memory in memories:
            for entity in memory.entities:
                if entity not in entity_memories:
                    entity_memories[entity] = []
                entity_memories[entity].append(memory)

        # Check for contradictions
        for entity, entity_mems in entity_memories.items():
            if len(entity_mems) < 2:
                continue

            # Get edges from KG
            edges = [
                (e.src, e.dst, e.type)
                for e in self.kg.get_edges()
                if e.src == entity or e.dst == entity
            ]

            # Simple contradiction detection: conflicting IS_A edges
            is_a_edges = [(src, dst) for src, dst, edge_type in edges if edge_type == "IS_A"]

            if len(is_a_edges) > 1:
                # Multiple IS_A edges - potential contradiction
                # Archive older memories
                sorted_mems = sorted(
                    entity_mems,
                    key=lambda m: m.metadata.get("timestamp", ""),
                    reverse=True
                )

                for old_memory in sorted_mems[1:]:  # Keep newest, archive rest
                    await self.archive_contradicted(
                        old_memory.id,
                        reason=f"Contradicted by newer memory (entity: {entity})"
                    )
                    contradiction_count += 1

        return contradiction_count

    # =========================================================================
    # Background Task Management
    # =========================================================================

    async def start_background_consolidation(self):
        """Start background sleep-based consolidation loop."""
        if self._running:
            logger.warning("Background consolidation already running")
            return

        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        logger.info(
            f"Started background sleep consolidation "
            f"(idle threshold: {self.config.idle_threshold_hours}h)"
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

            logger.info("Stopped background sleep consolidation")

    async def _consolidation_loop(self):
        """
        Background loop: Check for idle and consolidate.

        Runs every hour to check if system is idle.
        """
        check_interval = 3600  # 1 hour

        while self._running:
            try:
                await asyncio.sleep(check_interval)

                # Check if idle and consolidate
                if self.is_idle():
                    await self.consolidate_during_idle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sleep consolidation loop: {e}", exc_info=True)

    # =========================================================================
    # Statistics & Reporting
    # =========================================================================

    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive consolidation statistics.

        Returns:
            Statistics dictionary with all metrics
        """
        # Access stats summary
        total_accesses = sum(stats.access_count for stats in self.access_stats.values())
        avg_importance = (
            sum(stats.current_importance for stats in self.access_stats.values()) /
            len(self.access_stats)
            if self.access_stats else 0.0
        )

        promoted_memories = [
            stats for stats in self.access_stats.values()
            if stats.promoted_to_long_term
        ]
        archived_memories = [
            stats for stats in self.access_stats.values()
            if stats.archived
        ]

        # Idle status
        hours_since_query = (
            (datetime.now() - self.last_query_time).total_seconds() / 3600.0
            if self.last_query_time else None
        )

        return {
            "total_consolidations": self.total_consolidations,
            "total_memories_tracked": len(self.access_stats),
            "total_accesses": total_accesses,
            "avg_importance": avg_importance,
            "total_promoted": len(promoted_memories),
            "total_archived": len(archived_memories),
            "total_decayed": self.total_memories_decayed,
            "idle_threshold_hours": self.config.idle_threshold_hours,
            "hours_since_last_query": hours_since_query,
            "is_idle": self.is_idle(),
            "config": {
                "enabled": self.config.enabled,
                "decay_rate": self.config.decay_rate,
                "promotion_threshold": self.config.promotion_threshold_accesses,
                "promotion_window_days": self.config.promotion_window_days,
                "archive_threshold": self.config.archive_threshold,
                "contradiction_detection": self.config.contradiction_detection
            }
        }

