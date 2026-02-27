"""
Integrated Memory System (Weeks 1-4 Complete Integration)
===========================================================

This module provides a unified interface combining all memory system components:
- Week 1: Multi-level memory (scoping + lifecycles)
- Week 2: Agent-controlled tools + background consolidation
- Week 3: LLM-powered semantic fact extraction
- Week 4: Hybrid retrieval (semantic + BM25 + graph)

Usage:
    system = create_integrated_memory_system(config)

    # Store memory
    await system.store("Important information", importance=0.9)

    # Hybrid retrieval
    results = await system.retrieve("search query", limit=10)

    # Background consolidation
    await system.start_consolidation()

Architecture:
    IntegratedMemorySystem orchestrates:
    - ContextStreamManager (Week 1)
    - AgentMemoryTools (Week 2)
    - MemoryConsolidator (Week 3)
    - HybridRetriever (Week 4)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging

from HoloLoom.core.protocols.types import MemoryShard
from HoloLoom.core.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from HoloLoom.agentic.memory_tools import AgentMemoryTools, MemorySearchResult
from HoloLoom.core.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationStrategy,
    ConsolidationResult
)
from HoloLoom.core.memory.hybrid_retrieval import (
    HybridRetriever,
    create_hybrid_retriever,
    RetrievalResult
)
from HoloLoom.core.memory.graph import KG

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IntegratedMemoryConfig:
    """Configuration for integrated memory system."""

    # Week 3: LLM consolidation
    llm_provider: Optional[str] = None  # "openai", "anthropic", "ollama", None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    consolidation_interval_minutes: int = 60
    enable_consolidation: bool = False  # Opt-in

    # Week 4: Hybrid retrieval
    enable_semantic_search: bool = True
    enable_bm25_search: bool = True
    enable_graph_search: bool = True
    semantic_model: str = "all-MiniLM-L6-v2"

    # Memory management
    enable_archival: bool = True
    prune_consolidated_episodes: bool = False


# ============================================================================
# Integrated Memory System
# ============================================================================

class IntegratedMemorySystem:
    """
    Unified memory system combining Weeks 1-4.

    Provides simple API for:
    - Storing memories (agent-controlled)
    - Retrieving memories (hybrid search)
    - Background consolidation (episodic → semantic)
    - Multi-level scoping (USER/SESSION/AGENT/WORKING)
    """

    def __init__(
        self,
        config: Optional[IntegratedMemoryConfig] = None,
        stream_manager: Optional[ContextStreamManager] = None,
        knowledge_graph: Optional[KG] = None
    ):
        """
        Initialize integrated memory system.

        Args:
            config: System configuration
            stream_manager: Optional existing stream manager
            knowledge_graph: Optional existing knowledge graph
        """
        self.config = config or IntegratedMemoryConfig()

        # Week 1: Multi-level memory manager
        self.stream_manager = stream_manager or ContextStreamManager()

        # Knowledge graph (populated by consolidation)
        self.kg = knowledge_graph or KG()

        # Week 2: Agent memory tools
        self.agent_tools = AgentMemoryTools(
            stream_manager=self.stream_manager,
            knowledge_graph=self.kg,
            enable_archival=self.config.enable_archival
        )

        # Week 3: Background consolidation
        self.consolidator = MemoryConsolidator(
            stream_manager=self.stream_manager,
            knowledge_graph=self.kg,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            llm_api_key=self.config.llm_api_key,
            consolidation_interval_minutes=self.config.consolidation_interval_minutes,
            prune_consolidated_episodes=self.config.prune_consolidated_episodes
        )

        # Week 4: Hybrid retrieval
        self.hybrid_retriever = create_hybrid_retriever(
            kg=self.kg,
            semantic_model=self.config.semantic_model,
            enable_all=True  # Enable all methods
        )

        # Override individual retrievers based on config
        if not self.config.enable_semantic_search:
            self.hybrid_retriever.semantic_retriever = None
        if not self.config.enable_bm25_search:
            self.hybrid_retriever.bm25_retriever = None
        if not self.config.enable_graph_search:
            self.hybrid_retriever.graph_retriever = None

        logger.info(
            f"Initialized integrated memory system: "
            f"llm={self.config.llm_provider or 'none'}, "
            f"retrieval=[{self._enabled_retrievers()}]"
        )

    def _enabled_retrievers(self) -> str:
        """Get list of enabled retrieval methods."""
        methods = []
        if self.config.enable_semantic_search:
            methods.append("semantic")
        if self.config.enable_bm25_search:
            methods.append("bm25")
        if self.config.enable_graph_search:
            methods.append("graph")
        return "+".join(methods) if methods else "none"

    # ========================================================================
    # Week 2: Agent Memory Operations
    # ========================================================================

    async def store(
        self,
        content: str,
        scope: Optional[MemoryScope] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None,
        motifs: Optional[List[str]] = None,
        importance: float = 0.5
    ):
        """
        Store memory (Week 2 agent tools).

        Args:
            content: Memory content
            scope: Memory scope (None = auto-route)
            metadata: Optional metadata
            entities: Optional entities for knowledge graph
            motifs: Optional motifs/tags
            importance: Importance score (0.0-1.0)

        Returns:
            MemoryStoreResult
        """
        return await self.agent_tools.store(
            content=content,
            scope=scope,
            metadata=metadata,
            entities=entities,
            motifs=motifs,
            importance=importance
        )

    async def update(self, memory_id: str, new_content: str, metadata: Optional[Dict[str, Any]] = None):
        """Update memory (Week 2 - temporal invalidation)."""
        return await self.agent_tools.update(memory_id, new_content, metadata)

    async def delete(self, memory_id: str, reason: str = "user_requested"):
        """Delete memory (Week 2 - soft delete with archival)."""
        return await self.agent_tools.delete(memory_id, reason)

    # ========================================================================
    # Week 4: Hybrid Retrieval
    # ========================================================================

    async def retrieve(
        self,
        query: str,
        scopes: Optional[List[MemoryScope]] = None,
        lifecycles: Optional[List[LifeCycle]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> RetrievalResult:
        """
        Hybrid retrieval (Week 4 - semantic + BM25 + graph).

        Args:
            query: Search query
            scopes: Filter by scopes (None = all)
            lifecycles: Filter by lifecycles (None = all)
            limit: Maximum results
            min_importance: Minimum importance threshold

        Returns:
            RetrievalResult with hybrid rankings
        """
        # Get candidate memories (filtered by scope/lifecycle/importance)
        candidates = self.stream_manager.get_all_memories(
            scopes=scopes,
            lifecycles=lifecycles
        )

        # Filter by importance
        candidates = [
            m for m in candidates
            if m.metadata.get("importance", 0.5) >= min_importance
        ]

        # Hybrid retrieval
        result = await self.hybrid_retriever.retrieve(
            query=query,
            memories=candidates,
            limit=limit
        )

        logger.info(
            f"Hybrid retrieval: query='{query[:50]}...', "
            f"candidates={len(candidates)}, results={len(result.memories)}, "
            f"time={result.retrieval_time_ms:.1f}ms"
        )

        return result

    async def search_simple(
        self,
        query: str,
        scopes: Optional[List[MemoryScope]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> MemorySearchResult:
        """
        Simple search (Week 2 agent tools - keyword matching).

        Use this for backward compatibility or when hybrid retrieval is overkill.

        Args:
            query: Search query
            scopes: Filter by scopes
            limit: Maximum results
            min_importance: Minimum importance

        Returns:
            MemorySearchResult (Week 2 format)
        """
        return await self.agent_tools.search(
            query=query,
            scopes=scopes,
            limit=limit,
            min_importance=min_importance,
            strategy="hybrid"  # Uses hybrid internally
        )

    # ========================================================================
    # Week 3: Background Consolidation
    # ========================================================================

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours: int = 24
    ) -> ConsolidationResult:
        """
        Manual consolidation (Week 3 - episodic → semantic).

        Args:
            strategy: Consolidation strategy
            lookback_hours: How far back to consolidate

        Returns:
            ConsolidationResult
        """
        return await self.consolidator.consolidate_recent_episodes(
            strategy=strategy,
            lookback_hours=lookback_hours
        )

    async def start_consolidation(self):
        """Start background consolidation loop (Week 3)."""
        if not self.config.enable_consolidation:
            logger.warning("Background consolidation disabled in config")
            return

        await self.consolidator.start_background_consolidation()
        logger.info(
            f"Started background consolidation: "
            f"interval={self.config.consolidation_interval_minutes}min, "
            f"llm={self.config.llm_provider or 'none'}"
        )

    async def stop_consolidation(self):
        """Stop background consolidation loop (Week 3)."""
        await self.consolidator.stop_background_consolidation()
        logger.info("Stopped background consolidation")

    # ========================================================================
    # Performance Enhancements
    # ========================================================================

    def enable_spring_physics(self, config: Optional[Any] = None):
        """
        Enable spring physics for graph retrieval (9.6× faster!).

        Replaces simple BFS with physics-based spreading activation using
        Hooke's Law and Velocity Verlet integration.

        Performance:
            - 9.6× faster (9.35ms vs 90.10ms)
            - More relevant semantic results
            - Energy-conserving dynamics

        Args:
            config: Optional SpringConfig for custom physics parameters

        Example:
            ```python
            system = create_integrated_memory_system()

            # Enable spring physics
            system.enable_spring_physics()

            # Use normally - spring physics automatically used
            results = await system.retrieve("Bayesian methods", limit=10)
            ```
        """
        try:
            from HoloLoom.core.memory.spring_graph_retriever import (
                enable_spring_physics as _enable_spring
            )
            _enable_spring(self, config=config)
        except ImportError:
            import warnings
            warnings.warn(
                "Spring physics not available. Ensure spring_dynamics.py is present. "
                "Falling back to standard BFS graph retrieval."
            )

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # Week 1: Stream manager stats
        stream_stats = self.stream_manager.get_statistics()

        # Week 2: Agent tools stats
        agent_stats = self.agent_tools.get_statistics()

        # Week 3: Consolidation stats
        consolidation_stats = self.consolidator.get_statistics()

        # Week 4: Knowledge graph stats
        kg_stats = {
            "nodes": len(self.kg.G.nodes),
            "edges": len(self.kg.G.edges)
        }

        return {
            "streams": stream_stats,
            "agent_tools": agent_stats,
            "consolidation": consolidation_stats,
            "knowledge_graph": kg_stats,
            "config": {
                "llm_provider": self.config.llm_provider,
                "consolidation_enabled": self.config.enable_consolidation,
                "retrieval_methods": self._enabled_retrievers()
            }
        }

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def close(self):
        """Clean up resources."""
        # Stop consolidation if running
        if self.consolidator._running:
            await self.stop_consolidation()

        logger.info("Integrated memory system closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ============================================================================
# Factory Functions
# ============================================================================

def create_integrated_memory_system(
    llm_provider: Optional[str] = None,
    enable_consolidation: bool = False,
    enable_all_retrieval: bool = True
) -> IntegratedMemorySystem:
    """
    Create integrated memory system with sensible defaults.

    Args:
        llm_provider: LLM provider for consolidation (None = rule-based)
        enable_consolidation: Enable background consolidation
        enable_all_retrieval: Enable all retrieval methods

    Returns:
        IntegratedMemorySystem
    """
    config = IntegratedMemoryConfig(
        llm_provider=llm_provider,
        enable_consolidation=enable_consolidation,
        enable_semantic_search=enable_all_retrieval,
        enable_bm25_search=enable_all_retrieval,
        enable_graph_search=enable_all_retrieval
    )

    return IntegratedMemorySystem(config=config)


def create_production_memory_system(
    llm_provider: str = "openai",
    llm_model: str = "gpt-3.5-turbo"
) -> IntegratedMemorySystem:
    """
    Create production memory system with recommended settings.

    Args:
        llm_provider: LLM provider (openai, anthropic, ollama)
        llm_model: Model name

    Returns:
        IntegratedMemorySystem
    """
    config = IntegratedMemoryConfig(
        # Week 3: Production LLM
        llm_provider=llm_provider,
        llm_model=llm_model,
        consolidation_interval_minutes=60,
        enable_consolidation=True,
        prune_consolidated_episodes=False,  # Keep for audit trail

        # Week 4: Full hybrid retrieval
        enable_semantic_search=True,
        enable_bm25_search=True,
        enable_graph_search=True,

        # Memory management
        enable_archival=True  # Soft-delete for audit trail
    )

    return IntegratedMemorySystem(config=config)
