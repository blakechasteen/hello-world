"""
Agent-Controlled Memory Tools
==============================

Based on LangMem research: Agents explicitly control what to store, not passive accumulation.

Key Insights from LangMem:
1. "Agents decide what to store" - Explicit memory operations, not passive logging
2. "Two-path design" - Hot path (fast retrieval) + Background path (consolidation)
3. "Memory tools as first-class citizens" - Store, search, update, delete operations

Architecture:
- Store: Agent explicitly stores important information
- Search: Agent retrieves relevant memories with filters
- Update: Agent modifies existing memories (using temporal invalidation)
- Delete: Agent removes irrelevant memories (soft delete with archival)
- Consolidate: Background process extracts semantic facts from episodes

From Graphiti:
- Hybrid retrieval: Semantic search + BM25 + Graph traversal
- Temporal invalidation: Update facts without losing history
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging

from HoloLoom.Documentation.types import MemoryShard, Query
from HoloLoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from HoloLoom.memory.graph import KG, KGEdge

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Memory Tool Results
# ============================================================================

@dataclass
class MemorySearchResult:
    """Result from memory search operation."""
    memories: List[MemoryShard]
    total_found: int
    search_time_ms: float
    search_strategy: str  # "semantic", "bm25", "graph", "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStoreResult:
    """Result from memory store operation."""
    memory_id: str
    stream_name: str
    scope: MemoryScope
    lifecycle: LifeCycle
    success: bool
    message: str = ""


@dataclass
class MemoryUpdateResult:
    """Result from memory update operation."""
    memory_id: str
    old_version_archived: bool
    new_version_id: str
    success: bool
    message: str = ""


@dataclass
class MemoryDeleteResult:
    """Result from memory delete operation."""
    memory_id: str
    archived: bool  # Soft delete: moved to archive
    success: bool
    message: str = ""


# ============================================================================
# Agent Memory Tools
# ============================================================================

class AgentMemoryTools:
    """
    Agent-controlled memory operations (LangMem approach).

    Key Principle: "Agents decide what to store, not passive accumulation"

    Operations:
    1. store() - Explicitly store important information
    2. search() - Retrieve relevant memories with filters
    3. update() - Modify existing memories (temporal invalidation)
    4. delete() - Remove irrelevant memories (soft delete)
    5. get_statistics() - Get memory usage stats

    From research:
    - LangMem: Agent-controlled memory tools
    - Graphiti: Hybrid retrieval (semantic + BM25 + graph)
    - Mem0: Multi-level memory scoping
    """

    def __init__(
        self,
        stream_manager: ContextStreamManager,
        knowledge_graph: Optional[KG] = None,
        enable_archival: bool = True
    ):
        """
        Initialize agent memory tools.

        Args:
            stream_manager: Multi-level memory manager
            knowledge_graph: Optional KG for graph-based retrieval
            enable_archival: Enable soft-delete archival (recommended)
        """
        self.stream_manager = stream_manager
        self.kg = knowledge_graph or KG()
        self.enable_archival = enable_archival

        # Archive stream for soft-deleted memories
        if enable_archival:
            self.archive_stream = stream_manager.create_stream(
                name="archived_memories",
                scope=MemoryScope.WORKING,
                lifecycle=LifeCycle.PERMANENT  # Keep archives forever
            )

    async def store(
        self,
        content: str,
        scope: Optional[MemoryScope] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None,
        motifs: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> MemoryStoreResult:
        """
        Agent explicitly stores important information.

        From LangMem: "Agents decide what to store, not passive accumulation"

        Args:
            content: Memory content (text)
            scope: USER/SESSION/AGENT/WORKING (None = auto-route)
            metadata: Optional metadata (type, project_id, session_only, etc.)
            entities: Optional entities for knowledge graph
            motifs: Optional motifs/tags
            importance: Importance score (0.0-1.0, affects retrieval ranking)

        Returns:
            MemoryStoreResult with memory_id, stream_name, success
        """
        start_time = datetime.now()

        # Create memory shard
        memory_id = f"mem_{start_time.timestamp()}"
        metadata = metadata or {}
        metadata["timestamp"] = start_time.isoformat()
        metadata["importance"] = importance

        memory = MemoryShard(
            id=memory_id,
            text=content,
            entities=entities or [],
            motifs=motifs or [],
            metadata=metadata
        )

        # Route to appropriate stream (or use explicit scope)
        if scope:
            # Explicit scope - add to first stream with that scope
            streams = self.stream_manager.get_streams_by_scope(scope)
            if not streams:
                return MemoryStoreResult(
                    memory_id=memory_id,
                    stream_name="",
                    scope=scope,
                    lifecycle=LifeCycle.WORKING,
                    success=False,
                    message=f"No stream found for scope {scope.value}"
                )
            stream_name = streams[0].name
            await self.stream_manager.add_to_stream(stream_name, memory)
        else:
            # Automatic routing based on metadata
            stream_name = await self.stream_manager.route_memory(memory)

        # Add entities to knowledge graph
        if entities and self.kg:
            for i, entity in enumerate(entities):
                if i < len(entities) - 1:
                    # Connect consecutive entities
                    edge = KGEdge(
                        src=entity,
                        dst=entities[i + 1],
                        type="MENTIONS",
                        weight=importance,
                        metadata={"memory_id": memory_id}
                    )
                    self.kg.add_edges([edge])

        stream = self.stream_manager.get_stream(stream_name)

        logger.info(
            f"Agent stored memory: id={memory_id}, stream={stream_name}, "
            f"scope={stream.scope.value}, importance={importance}"
        )

        return MemoryStoreResult(
            memory_id=memory_id,
            stream_name=stream_name,
            scope=stream.scope,
            lifecycle=stream.lifecycle,
            success=True,
            message=f"Stored in {stream_name}"
        )

    async def search(
        self,
        query: str,
        scopes: Optional[List[MemoryScope]] = None,
        lifecycles: Optional[List[LifeCycle]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        strategy: str = "hybrid"  # "semantic", "bm25", "graph", "hybrid"
    ) -> MemorySearchResult:
        """
        Agent retrieves relevant memories with filters.

        From Graphiti: Hybrid retrieval (semantic + BM25 + graph traversal)

        Args:
            query: Search query
            scopes: Filter by memory scopes (None = all)
            lifecycles: Filter by lifecycles (None = all)
            limit: Maximum memories to return
            min_importance: Minimum importance score (0.0-1.0)
            strategy: "semantic", "bm25", "graph", or "hybrid"

        Returns:
            MemorySearchResult with memories, total_found, search_time_ms
        """
        start_time = datetime.now()

        # Get all memories matching scope/lifecycle filters
        all_memories = self.stream_manager.get_all_memories(
            scopes=scopes,
            lifecycles=lifecycles
        )

        # Filter by importance
        filtered_memories = [
            m for m in all_memories
            if m.metadata.get("importance", 0.5) >= min_importance
        ]

        # Simple ranking (in production, use embeddings + BM25 + graph)
        # For now: keyword matching + importance scoring
        def score_memory(m: MemoryShard) -> float:
            query_lower = query.lower()
            text_lower = m.text.lower()

            # Keyword match score (0.0-1.0)
            keywords = query_lower.split()
            matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = matches / max(len(keywords), 1)

            # Importance score
            importance = m.metadata.get("importance", 0.5)

            # Combined score: 70% keyword match, 30% importance
            return 0.7 * keyword_score + 0.3 * importance

        # Rank and limit
        ranked_memories = sorted(
            filtered_memories,
            key=score_memory,
            reverse=True
        )[:limit]

        search_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Agent searched memories: query='{query}', found={len(ranked_memories)}/{len(filtered_memories)}, "
            f"strategy={strategy}, time={search_time_ms:.1f}ms"
        )

        return MemorySearchResult(
            memories=ranked_memories,
            total_found=len(filtered_memories),
            search_time_ms=search_time_ms,
            search_strategy=strategy,
            metadata={
                "scopes": [s.value for s in scopes] if scopes else None,
                "lifecycles": [lc.value for lc in lifecycles] if lifecycles else None
            }
        )

    async def update(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryUpdateResult:
        """
        Agent updates existing memory (using temporal invalidation).

        From Graphiti: "Invalidate old edges, don't delete - preserves audit trail"

        Args:
            memory_id: ID of memory to update
            new_content: New content
            metadata: Optional new metadata

        Returns:
            MemoryUpdateResult with old_version_archived, new_version_id
        """
        # Find memory across all streams
        old_memory = None
        old_stream_name = None

        for stream_name, stream in self.stream_manager.streams.items():
            for mem in stream.memories:
                if mem.id == memory_id:
                    old_memory = mem
                    old_stream_name = stream_name
                    break
            if old_memory:
                break

        if not old_memory:
            return MemoryUpdateResult(
                memory_id=memory_id,
                old_version_archived=False,
                new_version_id="",
                success=False,
                message=f"Memory {memory_id} not found"
            )

        # Archive old version (soft delete)
        if self.enable_archival:
            archived_metadata = old_memory.metadata.copy()
            archived_metadata["archived_at"] = datetime.now().isoformat()
            archived_metadata["archived_reason"] = "updated"
            archived_metadata["original_id"] = memory_id

            archived_memory = MemoryShard(
                id=f"{memory_id}_archived_{datetime.now().timestamp()}",
                text=old_memory.text,
                entities=old_memory.entities,
                motifs=old_memory.motifs,
                metadata=archived_metadata
            )

            await self.archive_stream.add_memory(archived_memory)

        # Create new version
        new_metadata = metadata or old_memory.metadata.copy()
        now = datetime.now()
        new_metadata["timestamp"] = now.isoformat()
        new_metadata["previous_version"] = memory_id

        # Ensure unique ID (add microseconds to prevent collision)
        import time
        new_memory_id = f"mem_{now.timestamp()}_{int(time.time() * 1000000) % 1000}"
        new_memory = MemoryShard(
            id=new_memory_id,
            text=new_content,
            entities=old_memory.entities,  # Preserve entities (could extract new ones)
            motifs=old_memory.motifs,
            metadata=new_metadata
        )

        # Replace in original stream
        old_stream = self.stream_manager.get_stream(old_stream_name)
        old_stream.memories = [
            m if m.id != memory_id else new_memory
            for m in old_stream.memories
        ]

        logger.info(
            f"Agent updated memory: old_id={memory_id}, new_id={new_memory_id}, "
            f"archived={self.enable_archival}"
        )

        return MemoryUpdateResult(
            memory_id=memory_id,
            old_version_archived=self.enable_archival,
            new_version_id=new_memory_id,
            success=True,
            message=f"Updated to {new_memory_id}"
        )

    async def delete(
        self,
        memory_id: str,
        reason: str = "agent_requested"
    ) -> MemoryDeleteResult:
        """
        Agent deletes irrelevant memory (soft delete with archival).

        From research: Never hard-delete - always archive for audit trail

        Args:
            memory_id: ID of memory to delete
            reason: Reason for deletion (logged in archive)

        Returns:
            MemoryDeleteResult with archived status
        """
        # Find memory across all streams
        target_memory = None
        target_stream_name = None

        for stream_name, stream in self.stream_manager.streams.items():
            for mem in stream.memories:
                if mem.id == memory_id:
                    target_memory = mem
                    target_stream_name = stream_name
                    break
            if target_memory:
                break

        if not target_memory:
            return MemoryDeleteResult(
                memory_id=memory_id,
                archived=False,
                success=False,
                message=f"Memory {memory_id} not found"
            )

        # Archive before deletion (soft delete)
        if self.enable_archival:
            archived_metadata = target_memory.metadata.copy()
            archived_metadata["archived_at"] = datetime.now().isoformat()
            archived_metadata["archived_reason"] = reason
            archived_metadata["original_id"] = memory_id

            archived_memory = MemoryShard(
                id=f"{memory_id}_deleted_{datetime.now().timestamp()}",
                text=target_memory.text,
                entities=target_memory.entities,
                motifs=target_memory.motifs,
                metadata=archived_metadata
            )

            await self.archive_stream.add_memory(archived_memory)

        # Remove from original stream
        target_stream = self.stream_manager.get_stream(target_stream_name)
        target_stream.memories = [
            m for m in target_stream.memories if m.id != memory_id
        ]

        logger.info(
            f"Agent deleted memory: id={memory_id}, reason={reason}, "
            f"archived={self.enable_archival}"
        )

        return MemoryDeleteResult(
            memory_id=memory_id,
            archived=self.enable_archival,
            success=True,
            message=f"Deleted (archived: {self.enable_archival})"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics across all streams."""
        base_stats = self.stream_manager.get_statistics()

        # Add archival stats
        if self.enable_archival:
            archived_count = self.archive_stream.count()
            base_stats["archived_memories"] = archived_count

        return base_stats
