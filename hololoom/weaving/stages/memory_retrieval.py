"""
Memory Retrieval Stage
======================
Step 6 of the weaving cycle: Memory retrieval with multipass crawling.

This stage retrieves relevant context from memory using:
- Multipass memory crawling (if available)
- Traditional retriever (fallback)
- Graph traversal for connected knowledge
"""

import logging
import time
from typing import Any

from hololoom.loom.command import PatternSpec
from hololoom.protocols.types import MemoryShard, Query
from hololoom.weaving.protocols import StageResult


class MemoryRetrievalStage:
    """
    Memory Retrieval Stage (Step 6).

    Retrieves relevant memory shards using intelligent multipass crawling
    or traditional retrieval methods.
    """

    def __init__(
        self,
        memory_backend: Any | None = None,
        retriever: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the memory retrieval stage.

        Args:
            memory_backend: Dynamic memory backend (preferred)
            retriever: Traditional static retriever (fallback)
            logger: Logger instance
        """
        self.memory = memory_backend
        self.retriever = retriever
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        query: Query,
        context: dict[str, Any],
        pattern_spec: PatternSpec,
    ) -> StageResult:
        """
        Execute memory retrieval.

        Args:
            query: The user query
            context: Shared context
            pattern_spec: Pattern specification for retrieval

        Returns:
            StageResult with retrieved memory shards
        """
        start_time = time.time()

        try:
            # Get complexity and provenance from context
            complexity = context.get("complexity")
            provenance = context.get("provenance")

            # Retrieve memory shards
            shards, shard_texts, hits = await self._retrieve_memories(
                query, pattern_spec, complexity, provenance
            )

            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(f"  [6] Retrieved {len(shards)} context shards")

            return StageResult(
                stage_name="memory_retrieval",
                duration_ms=duration_ms,
                data={
                    "shards": shards,
                    "shard_texts": shard_texts,
                    "hits": hits,
                    "shard_count": len(shards),
                },
                metadata={
                    "retrieval_method": self._get_retrieval_method(),
                    "retrieval_k": pattern_spec.retrieval_k,
                    "multipass_enabled": self.memory is not None,
                }
            )

        except Exception as e:
            self.logger.error(f"Memory retrieval failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return StageResult(
                stage_name="memory_retrieval",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                data={"shards": [], "shard_texts": [], "hits": []}
            )

    async def _retrieve_memories(
        self,
        query: Query,
        pattern_spec: PatternSpec,
        complexity: Any | None = None,
        provenance: Any | None = None,
    ) -> tuple[list[MemoryShard], list[str], list[tuple[MemoryShard, float]]]:
        """
        Retrieve memories using the appropriate method.

        Args:
            query: The user query
            pattern_spec: Pattern specification
            complexity: Complexity level
            provenance: Provenance tracking

        Returns:
            Tuple of (shards, shard_texts, hits)
        """
        if self.memory:
            # Use multipass memory crawling for intelligent retrieval
            shards = await self._multipass_memory_crawl(
                query, complexity, provenance, pattern_spec
            )
            shard_texts = [shard.text for shard in shards]
            hits = [(shard, 1.0) for shard in shards]
            self.logger.info(f"  [6] Multipass crawl retrieved {len(shards)} shards")

        elif self.retriever:
            # Fallback: Traditional static shard retrieval
            hits = await self.retriever.search(
                query=query.text,
                k=pattern_spec.retrieval_k,
                fast=(pattern_spec.retrieval_mode == "fast")
            )
            shards = [shard for shard, _ in hits]
            shard_texts = [shard.text for shard in shards]
            self.logger.info(f"  [6] Legacy retriever fetched {len(shards)} shards")

        else:
            # No memory source available
            self.logger.warning("No memory source available")
            shards = []
            shard_texts = []
            hits = []

        return shards, shard_texts, hits

    async def _multipass_memory_crawl(
        self,
        query: Query,
        complexity: Any | None,
        provenance: Any | None,
        pattern_spec: PatternSpec,
    ) -> list[MemoryShard]:
        """
        Perform multipass memory crawling.

        This is a simplified version - the actual implementation would be
        more sophisticated with graph traversal, gated retrieval, etc.

        Args:
            query: The user query
            complexity: Complexity level
            provenance: Provenance tracking
            pattern_spec: Pattern specification

        Returns:
            List of retrieved memory shards
        """
        # Import the actual multipass retrieval function
        try:
            from hololoom.orchestrator.retrieval import multipass_memory_crawl

            # Create a mock orchestrator context with necessary attributes
            class MockOrchestrator:
                def __init__(self, memory, logger):
                    self.memory = memory
                    self.logger = logger
                    self.cfg = type('Config', (), {
                        'multipass_graph_crawling': True,
                        'max_crawl_depth': 2,
                        'max_crawl_breadth': 5
                    })()

            mock_orchestrator = MockOrchestrator(self.memory, self.logger)
            return await multipass_memory_crawl(
                mock_orchestrator, query, complexity, provenance
            )

        except ImportError:
            # Fallback to simple memory query
            self.logger.warning("Multipass crawling not available, using simple query")
            return await self._simple_memory_query(query, pattern_spec)

    async def _simple_memory_query(
        self,
        query: Query,
        pattern_spec: PatternSpec
    ) -> list[MemoryShard]:
        """
        Simple memory query fallback.

        Args:
            query: The user query
            pattern_spec: Pattern specification

        Returns:
            List of memory shards
        """
        if hasattr(self.memory, 'search'):
            results = await self.memory.search(
                query.text,
                k=pattern_spec.retrieval_k
            )
            return [r for r in results if isinstance(r, MemoryShard)]
        return []

    def _get_retrieval_method(self) -> str:
        """Get the retrieval method being used."""
        if self.memory:
            return "multipass_crawling"
        elif self.retriever:
            return "static_retriever"
        else:
            return "none"

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "memory_retrieval"

    def get_required_context_keys(self) -> list[str]:
        """Get required context keys."""
        return []  # Optional: complexity, provenance

    def get_output_context_keys(self) -> list[str]:
        """Get output context keys."""
        return ["shards", "shard_texts", "hits"]
