#!/usr/bin/env python3
"""
HoloLoom RAG Integration
=========================
Integrates RAG-ingested context with HoloLoom's memory system
for context-aware code generation.

This module bridges the gap between RAG ingestion and LLM code generation
by using HoloLoom's unified memory API (experience/recall/reflect).

Author: Claude Code
Date: November 16, 2025
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from HoloLoom.hololoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.Documentation.types import MemoryShard, Query


class HoloLoomRAGIntegration:
    """
    Integrates RAG ingestion with HoloLoom's memory system.

    Architecture:
    1. RAG engines ingest context → MemoryShards
    2. Shards stored in HoloLoom via experience()
    3. Relevant context retrieved via recall()
    4. Context passed to LLM for code generation

    This creates a true RAG loop:
    External Sources → HoloLoom Memory → LLM Context → Enhanced Code
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize HoloLoom RAG integration"""
        self.config = config or Config.fast()
        self.hololoom: Optional[HoloLoom] = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize HoloLoom memory system"""
        if not self._initialized:
            self.hololoom = HoloLoom(config=self.config)
            await self.hololoom.__aenter__()
            self._initialized = True

    async def cleanup(self):
        """Cleanup HoloLoom resources"""
        if self._initialized and self.hololoom:
            await self.hololoom.__aexit__(None, None, None)
            self._initialized = False

    async def ingest_shards(self, shards: List[MemoryShard], source: str) -> int:
        """
        Ingest MemoryShards into HoloLoom memory.

        Args:
            shards: List of MemoryShards from RAG ingestion
            source: Source type (codebase, api, documentation, forum)

        Returns:
            Number of shards successfully ingested
        """
        if not self._initialized:
            await self.initialize()

        ingested_count = 0

        for shard in shards:
            try:
                # Experience the shard in HoloLoom
                # This stores it in the awareness graph with semantic embeddings
                memory = await self.hololoom.experience(shard.text)
                ingested_count += 1
            except Exception as e:
                # Log error but continue with other shards
                print(f"Warning: Failed to ingest shard {shard.id}: {e}")

        return ingested_count

    async def recall_relevant_context(
        self,
        query: str,
        max_items: int = 10,
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant context from HoloLoom memory.

        Args:
            query: Query string (e.g., code description or question)
            max_items: Maximum number of relevant items to return
            min_relevance: Minimum relevance threshold (0.0-1.0)

        Returns:
            List of relevant context items with scores
        """
        if not self._initialized:
            await self.initialize()

        # Use HoloLoom's recall to find relevant memories
        memories = await self.hololoom.recall(
            query=query,
            k=max_items
        )

        # Format results with relevance scores
        context_items = []
        for memory in memories:
            # HoloLoom memories have text content and metadata
            context_items.append({
                "text": memory.text,
                "source": memory.metadata.get("source", "unknown"),
                "entities": memory.entities if hasattr(memory, 'entities') else [],
                "relevance": memory.metadata.get("score", 1.0)
            })

        # Filter by relevance threshold
        filtered = [
            item for item in context_items
            if item["relevance"] >= min_relevance
        ]

        return filtered

    async def get_enriched_context_for_code_generation(
        self,
        description: str,
        language: Optional[str] = None,
        max_context_items: int = 5
    ) -> str:
        """
        Get enriched context for code generation.

        Recalls relevant code examples, API specs, documentation,
        and forum solutions to enhance code generation.

        Args:
            description: Code generation description
            language: Programming language (filters context)
            max_context_items: Maximum context items to include

        Returns:
            Formatted context string to prepend to LLM prompt
        """
        # Build query with language filter
        query = description
        if language:
            query = f"{language} {description}"

        # Recall relevant context
        context_items = await self.recall_relevant_context(
            query=query,
            max_items=max_context_items,
            min_relevance=0.6
        )

        if not context_items:
            return ""

        # Format context for LLM
        context_parts = [
            "# Relevant Context from Codebase/APIs/Documentation:",
            ""
        ]

        for idx, item in enumerate(context_items, 1):
            source = item["source"]
            text = item["text"]
            relevance = item["relevance"]

            context_parts.append(f"## Context {idx} (from {source}, relevance: {relevance:.2f})")
            context_parts.append(text)
            context_parts.append("")

        return "\n".join(context_parts)

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get HoloLoom awareness graph metrics.

        Returns:
            Dictionary with activation, coherence, temporal metrics
        """
        if not self._initialized:
            return {"error": "HoloLoom not initialized"}

        return self.hololoom.get_metrics()

    async def get_summary(self) -> str:
        """
        Get human-readable summary of HoloLoom memory state.

        Returns:
            Summary string
        """
        if not self._initialized:
            return "HoloLoom not initialized"

        return self.hololoom.summary()


# ============================================================================
# Helper Functions
# ============================================================================

async def ingest_rag_shards_to_hololoom(
    shards: List[MemoryShard],
    source: str,
    config: Optional[Config] = None
) -> int:
    """
    Helper function to ingest RAG shards into HoloLoom.

    Usage:
        shards = codebase_engine.entities_to_memory_shards(entities)
        count = await ingest_rag_shards_to_hololoom(shards, "codebase")

    Args:
        shards: MemoryShards from RAG ingestion
        source: Source type (codebase, api, documentation, forum)
        config: Optional HoloLoom config

    Returns:
        Number of shards ingested
    """
    async with HoloLoomRAGIntegration(config) as integration:
        return await integration.ingest_shards(shards, source)


async def get_rag_context_for_code_generation(
    description: str,
    language: Optional[str] = None,
    max_items: int = 5,
    config: Optional[Config] = None
) -> str:
    """
    Helper function to get RAG context for code generation.

    Usage:
        context = await get_rag_context_for_code_generation(
            "Create a FastAPI endpoint",
            language="python",
            max_items=5
        )
        # Pass context to LLM prompt

    Args:
        description: Code generation description
        language: Programming language
        max_items: Maximum context items
        config: Optional HoloLoom config

    Returns:
        Formatted context string
    """
    async with HoloLoomRAGIntegration(config) as integration:
        return await integration.get_enriched_context_for_code_generation(
            description, language, max_items
        )


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example of HoloLoom RAG integration"""

    # Create integration
    async with HoloLoomRAGIntegration() as integration:

        # Example 1: Ingest codebase shards
        print("Example 1: Ingesting codebase...")
        codebase_shards = [
            MemoryShard(
                id="code_001",
                text="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                entities=["fibonacci", "recursion"],
                motifs=["algorithm", "python"],
                metadata={"source": "codebase", "file": "utils.py"}
            ),
            MemoryShard(
                id="code_002",
                text="class FastAPIEndpoint:\n    def __init__(self, path, method): ...",
                entities=["FastAPIEndpoint", "class"],
                motifs=["api", "python", "fastapi"],
                metadata={"source": "codebase", "file": "api.py"}
            )
        ]

        count = await integration.ingest_shards(codebase_shards, "codebase")
        print(f"Ingested {count} shards")

        # Example 2: Recall relevant context
        print("\nExample 2: Recalling context...")
        context = await integration.recall_relevant_context(
            "How do I create a FastAPI endpoint?",
            max_items=5
        )

        print(f"Found {len(context)} relevant items:")
        for item in context:
            print(f"  - {item['text'][:50]}... (relevance: {item['relevance']:.2f})")

        # Example 3: Get enriched context for code generation
        print("\nExample 3: Getting enriched context...")
        enriched = await integration.get_enriched_context_for_code_generation(
            description="Create a recursive fibonacci function",
            language="python",
            max_context_items=3
        )

        print("Enriched context:")
        print(enriched)

        # Example 4: Get metrics
        print("\nExample 4: Memory metrics...")
        metrics = await integration.get_metrics()
        print(f"Active nodes: {metrics.get('activation', {}).get('active_nodes', 0)}")
        print(f"Total nodes: {metrics.get('activation', {}).get('total_nodes', 0)}")


if __name__ == "__main__":
    print("HoloLoom RAG Integration Example")
    print("=" * 80)
    asyncio.run(example_usage())
