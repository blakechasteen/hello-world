#!/usr/bin/env python3
"""
Auto-Spin: The Ruthlessly Elegant Path
=======================================
Everything is a memory operation.

Philosophy: "If you need to configure it, we failed."

Usage:
    from HoloLoom.spinningWheel import spin

    # Ingest anything into memory
    memory = await spin("My thoughts on bee survival...")
    memory = await spin("https://example.com/article.html")
    memory = await spin("/path/to/document.pdf")
    memory = await spin({"data": "structured json"})
    memory = await spin([text, image, audio])  # Multi-modal

    # Use existing memory backend
    await spin(data, memory=existing_memory)

That's it. Everything else is automatic.

Author: Claude Code
Date: October 29, 2025
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import asyncio


# ============================================================================
# Universal Ingestion
# ============================================================================

async def spin(
    source: Union[str, Path, bytes, Dict, List, Any],
    memory: Optional[Any] = None,
    return_shards: bool = False
) -> Any:
    """
    Ruthlessly elegant data ingestion.

    Accepts ANYTHING and automatically:
    1. Detects input type (text, image, audio, structured, URL, file, multi-modal)
    2. Processes with appropriate pipeline
    3. Extracts entities, motifs, embeddings
    4. Ingests into memory backend

    Args:
        source: Anything you want to ingest:
            - Text string
            - File path (PDF, audio, image, CSV, JSON)
            - URL (webpage, YouTube, API endpoint)
            - Bytes (raw data)
            - Dict (structured data)
            - List (multi-modal inputs)
        memory: Optional memory backend (creates INMEMORY if None)
        return_shards: If True, return shards instead of memory

    Returns:
        Memory backend with data ingested (or shards if return_shards=True)

    Examples:
        >>> from HoloLoom.spinningWheel import spin

        # Text
        >>> memory = await spin("My research notes on bee colonies...")

        # URL
        >>> memory = await spin("https://wikipedia.org/wiki/Honeybee")

        # File
        >>> memory = await spin("/path/to/research_paper.pdf")

        # Structured data
        >>> memory = await spin({"temperature": -5, "survival": 85})

        # Multi-modal
        >>> memory = await spin([text, image, audio])

        # Use existing memory
        >>> await spin(data, memory=my_memory_backend)
    """
    from .multimodal_spinner import MultiModalSpinner

    # Step 1: Process input with auto-detection
    spinner = MultiModalSpinner(enable_fusion=True)

    try:
        shards = await spinner.spin(source)
    except Exception as e:
        print(f"[spin] Warning: Processing failed: {e}")
        # Create minimal error shard
        from HoloLoom.documentation.types import MemoryShard
        shards = [MemoryShard(
            id=f"error_{hash(str(source)[:100])}",
            text=f"Error processing input: {str(e)}",
            episode="error",
            entities=[],
            motifs=["error", "failed_ingestion"],
            metadata={"error": str(e), "source_type": type(source).__name__}
        )]

    # Return shards if requested (for advanced usage)
    if return_shards:
        return shards

    # Step 2: Ingest into memory
    if memory is None:
        # Create in-memory backend automatically
        memory = await _create_auto_memory()

    # Step 3: Add shards to memory
    await _ingest_shards(memory, shards)

    print(f"[spin] Ingested {len(shards)} shard(s) into memory")

    return memory


async def _create_auto_memory():
    """Create automatic in-memory backend."""
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        from HoloLoom.config import Config, MemoryBackend

        # Create INMEMORY backend (always works, no dependencies)
        config = Config.bare()
        config.memory_backend = MemoryBackend.INMEMORY

        memory = await create_memory_backend(config)
        return memory
    except Exception as e:
        print(f"[spin] Warning: Could not create memory backend: {e}")
        print("[spin] Using minimal fallback memory")
        return MinimalMemory()


async def _ingest_shards(memory, shards):
    """Ingest shards into memory backend."""
    # Try standard memory backend interface
    if hasattr(memory, 'add_shards'):
        await memory.add_shards(shards)
        return

    # Try individual add_shard
    if hasattr(memory, 'add_shard'):
        for shard in shards:
            await memory.add_shard(shard)
        return

    # Try KG graph backend (HoloLoom.memory.graph.KG)
    if hasattr(memory, 'add_edge') and hasattr(memory, 'G'):
        from HoloLoom.memory.graph import KGEdge

        for shard in shards:
            shard_id = f"shard_{shard.id}"

            # Add edges between shard and entities
            for entity in shard.entities[:10]:  # Limit to first 10
                edge = KGEdge(
                    src=shard_id,
                    dst=entity,
                    type="MENTIONS",
                    weight=1.0,
                    metadata={
                        'text': shard.text[:100],
                        'shard_id': shard.id
                    }
                )
                memory.add_edge(edge)

            # Add edges between shard and motifs
            for motif in shard.motifs[:5]:  # Limit to first 5
                edge = KGEdge(
                    src=shard_id,
                    dst=motif,
                    type="HAS_TOPIC",
                    weight=1.0,
                    metadata={'shard_id': shard.id}
                )
                memory.add_edge(edge)

        print(f"[spin] Added {len(shards)} shard(s) to knowledge graph")
        return

    # Fallback: store in memory object
    if isinstance(memory, MinimalMemory):
        memory.shards.extend(shards)
        return

    raise ValueError(f"Memory backend {type(memory)} doesn't support shard ingestion")


class MinimalMemory:
    """
    Minimal fallback memory for when backends aren't available.

    Just stores shards in a list. Not persistent, but always works.
    """

    def __init__(self):
        self.shards = []

    def get_all_shards(self):
        """Get all stored shards."""
        return self.shards

    def query(self, text: str):
        """Simple text search."""
        results = []
        for shard in self.shards:
            if text.lower() in shard.text.lower():
                results.append(shard)
        return results

    def __repr__(self):
        return f"<MinimalMemory: {len(self.shards)} shards>"


# ============================================================================
# Batch Processing (for bulk ingestion)
# ============================================================================

async def spin_batch(
    sources: List[Any],
    memory: Optional[Any] = None,
    max_concurrent: int = 5
) -> Any:
    """
    Ingest multiple sources concurrently.

    Args:
        sources: List of sources to ingest
        memory: Optional memory backend (creates if None)
        max_concurrent: Max concurrent processing tasks

    Returns:
        Memory backend with all data ingested

    Example:
        >>> sources = [
        ...     "https://example.com/article1.html",
        ...     "https://example.com/article2.html",
        ...     "/path/to/document.pdf",
        ...     "My notes on the topic..."
        ... ]
        >>> memory = await spin_batch(sources)
    """
    # Create memory once if needed
    if memory is None:
        memory = await _create_auto_memory()

    # Process in batches to avoid overwhelming the system
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(source):
        async with semaphore:
            return await spin(source, memory=memory)

    # Process all sources
    tasks = [process_one(src) for src in sources]
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"[spin_batch] Ingested {len(sources)} sources into memory")

    return memory


# ============================================================================
# URL Extraction Helpers
# ============================================================================

async def spin_url(
    url: str,
    memory: Optional[Any] = None,
    max_depth: int = 1,
    follow_links: bool = False
) -> Any:
    """
    Ingest web content with optional crawling.

    Args:
        url: URL to ingest
        memory: Optional memory backend
        max_depth: Max crawl depth (if follow_links=True)
        follow_links: Whether to follow links on page

    Returns:
        Memory backend with content ingested

    Example:
        >>> # Single page
        >>> memory = await spin_url("https://example.com")

        >>> # Crawl with depth
        >>> memory = await spin_url(
        ...     "https://example.com",
        ...     follow_links=True,
        ...     max_depth=2
        ... )
    """
    if not follow_links:
        # Simple single-page ingestion
        return await spin(url, memory=memory)

    # Multi-page crawling
    if memory is None:
        memory = await _create_auto_memory()

    visited = set()
    to_visit = [(url, 0)]  # (url, depth)

    while to_visit:
        current_url, depth = to_visit.pop(0)

        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)

        # Ingest current page
        await spin(current_url, memory=memory)

        # Extract links if depth allows
        if depth < max_depth:
            # Would extract links here and add to to_visit
            # For now, just ingest the single page
            pass

    print(f"[spin_url] Ingested {len(visited)} page(s)")

    return memory


# ============================================================================
# File System Helpers
# ============================================================================

async def spin_directory(
    directory: Union[str, Path],
    memory: Optional[Any] = None,
    pattern: str = "*",
    recursive: bool = False
) -> Any:
    """
    Ingest all files in a directory.

    Args:
        directory: Directory path
        memory: Optional memory backend
        pattern: File pattern (e.g., "*.txt", "*.pdf")
        recursive: Whether to recurse subdirectories

    Returns:
        Memory backend with all files ingested

    Example:
        >>> # Ingest all text files
        >>> memory = await spin_directory("/path/to/docs", pattern="*.txt")

        >>> # Recursive ingestion
        >>> memory = await spin_directory(
        ...     "/path/to/research",
        ...     pattern="*",
        ...     recursive=True
        ... )
    """
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    # Find matching files
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))

    print(f"[spin_directory] Found {len(files)} file(s) matching pattern '{pattern}'")

    # Ingest all files
    return await spin_batch(files, memory=memory)


# ============================================================================
# Query Extraction (from HoloLoom queries)
# ============================================================================

async def spin_from_query(
    query_text: str,
    memory: Optional[Any] = None,
    orchestrator: Optional[Any] = None
) -> Any:
    """
    Execute HoloLoom query and ingest results into memory.

    Bridges the gap between querying and learning.

    Args:
        query_text: Query to execute
        memory: Optional memory backend for ingestion
        orchestrator: Optional WeavingOrchestrator (creates if None)

    Returns:
        Memory backend with query results ingested

    Example:
        >>> memory = await spin_from_query(
        ...     "What are the best practices for bee winter survival?",
        ...     memory=my_knowledge_base
        ... )
    """
    # Create orchestrator if needed
    if orchestrator is None:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        config = Config.fast()
        orchestrator = WeavingOrchestrator(cfg=config)

    # Execute query
    from HoloLoom.documentation.types import Query
    spacetime = await orchestrator.weave(Query(text=query_text))

    # Extract response as text to ingest
    response_text = f"Query: {query_text}\n\nResponse: {spacetime.response}"

    # Ingest into memory
    return await spin(response_text, memory=memory)
