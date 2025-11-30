"""
Interleaved Expansion + Generation (Phase 3 + Phase 4)
=======================================================

Combines streaming context expansion (Phase 2) with LLM generation.
Instead of retrieving all context THEN generating, this interleaves both:

1. Start generation with first context chunk (seed nodes)
2. Continue expanding graph in background
3. Feed new chunks to LLM as discovered
4. Stream BOTH expansion chunks AND generation tokens

Phase 3 MVP: Background generation with batched token yielding
Phase 4: True concurrent token yielding (<100ms to first token)

Expected Benefits:
- 40-60% lower end-to-end latency
- Progressive results (show answers immediately)
- More efficient token usage (can stop early if answer found)
- Better user experience (no waiting for full retrieval)
- <100ms latency to first token (Phase 4)

Author: Claude Code
Date: 2025-11-25
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

from HoloLoom.memory.streaming_expansion import (
    ContextChunk,
    StreamingContextBuilder,
    stream_context_expansion
)


# ============================================================================
# Data Structures
# ============================================================================

class StreamItemType(str, Enum):
    """Type of item in interleaved stream."""
    CONTEXT_CHUNK = "context_chunk"
    GENERATION_TOKEN = "generation_token"
    METADATA = "metadata"


class StreamMode(str, Enum):
    """Streaming mode for token yielding."""
    BATCHED = "batched"      # Phase 3 MVP: Collect all tokens, yield at end
    CONCURRENT = "concurrent"  # Phase 4: Yield tokens as generated (true interleaving)


@dataclass
class GenerationToken:
    """
    A single token from LLM generation.

    Attributes:
        token: The generated token text
        cumulative_text: Full text generated so far
        token_index: Index of this token (0-based)
        is_final: True if this is the last token
        metadata: Additional metadata (confidence, logprobs, etc.)
    """
    token: str
    cumulative_text: str
    token_index: int
    is_final: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetadata:
    """
    Metadata about the interleaved stream.

    Attributes:
        event_type: Type of metadata event
        data: Event data
    """
    event_type: str  # "expansion_start", "generation_start", "expansion_complete", etc.
    data: Dict[str, Any] = field(default_factory=dict)


# Type alias for stream items
StreamItem = Union[ContextChunk, GenerationToken, StreamMetadata]


@dataclass
class InterleavedResult:
    """
    Final result from interleaved expansion + generation.

    Attributes:
        response: Complete generated response
        context_chunks: All context chunks retrieved
        total_tokens_generated: Number of tokens generated
        total_tokens_retrieved: Number of tokens in context
        expansion_time_ms: Time spent on graph expansion
        generation_time_ms: Time spent on generation
        total_time_ms: Total end-to-end time
        metadata: Additional metadata
    """
    response: str
    context_chunks: List[ContextChunk]
    total_tokens_generated: int
    total_tokens_retrieved: int
    expansion_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LLM Protocol (for abstraction)
# ============================================================================

class LLMProtocol:
    """
    Protocol for LLM providers.

    Enables pluggable LLM backends (OpenAI, Anthropic, local models, etc.)
    """

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """
        Stream generation tokens.

        Args:
            prompt: The user query/prompt
            context: Retrieved context to condition on
            max_tokens: Maximum tokens to generate

        Yields:
            Individual tokens as strings
        """
        raise NotImplementedError


# ============================================================================
# Mock LLM (for testing)
# ============================================================================

class MockLLM(LLMProtocol):
    """
    Mock LLM for testing and demos.
    Simulates streaming generation without requiring real API calls.
    """

    def __init__(self, response_template: Optional[str] = None, tokens_per_second: int = 30):
        """
        Initialize mock LLM.

        Args:
            response_template: Template response (default: context-based answer)
            tokens_per_second: Simulation speed (default: 30 tok/s)
        """
        self.response_template = response_template
        self.tokens_per_second = tokens_per_second

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """
        Simulate streaming generation.

        Yields tokens at ~30 tok/s to simulate real LLM.
        """
        if self.response_template:
            response = self.response_template
        else:
            # Generate context-aware response
            response = f"Based on the context, {prompt.lower()} involves multiple concepts. "
            response += f"The key information includes: {context[:100]}... "
            response += "This demonstrates the interleaved expansion and generation working together."

        # Split into tokens (words + punctuation)
        tokens = response.replace(",", " ,").replace(".", " .").split()

        delay = 1.0 / self.tokens_per_second

        for token in tokens[:max_tokens]:
            yield token + " "
            await asyncio.sleep(delay)


# ============================================================================
# Interleaved Stream Manager
# ============================================================================

class InterleavedStreamManager:
    """
    Manages interleaving of expansion chunks and generation tokens.

    Core algorithm:
    1. Start expansion stream (Phase 2)
    2. Wait for first chunk
    3. Start generation with first chunk's context
    4. Interleave both streams:
       - Yield context chunks as discovered
       - Yield generation tokens as produced
    5. Continue until both streams complete
    """

    def __init__(
        self,
        llm: LLMProtocol,
        expansion_builder: Optional[StreamingContextBuilder] = None
    ):
        """
        Initialize interleaved stream manager.

        Args:
            llm: LLM provider (must implement LLMProtocol)
            expansion_builder: Streaming context builder (default: create new)
        """
        self.llm = llm
        self.expansion_builder = expansion_builder or StreamingContextBuilder()

    async def stream_interleaved(
        self,
        query: str,
        seed_nodes: List[str],
        graph: Any,
        token_budget: int = 2000,
        max_generation_tokens: int = 500,
        chunk_size: int = 500,
        min_relevance: float = 0.3,
        max_hops: int = 5,
        importance_scores: Optional[Dict[str, float]] = None,
        node_contents: Optional[Dict[str, str]] = None,
        emit_metadata: bool = False,
        stream_mode: StreamMode = StreamMode.BATCHED
    ) -> AsyncIterator[StreamItem]:
        """
        Interleave context expansion with LLM generation.

        Phase 3 MVP (BATCHED):
        1. Yield first chunk immediately
        2. Start generation in background
        3. Continue expansion, yielding chunks
        4. Drain generation tokens after expansion completes

        Phase 4 (CONCURRENT):
        1. Yield first chunk immediately
        2. Start generation in background with shared queue
        3. Yield tokens as generated (true interleaving)
        4. Continue expansion simultaneously
        5. Both streams complete independently

        Args:
            query: User query
            seed_nodes: Starting nodes for expansion
            graph: Knowledge graph
            token_budget: Token budget for expansion
            max_generation_tokens: Max tokens to generate
            chunk_size: Target chunk size
            min_relevance: Minimum relevance threshold
            max_hops: Maximum hops for expansion
            importance_scores: Node importance scores
            node_contents: Node text contents
            emit_metadata: Whether to emit metadata events
            stream_mode: BATCHED (Phase 3) or CONCURRENT (Phase 4)

        Yields:
            ContextChunk, GenerationToken, or StreamMetadata objects
        """
        import time
        start_time = time.time()

        if emit_metadata:
            yield StreamMetadata("expansion_start", {"query": query, "seed_nodes": seed_nodes})

        # Route to appropriate implementation based on stream_mode
        if stream_mode == StreamMode.CONCURRENT:
            # Phase 4: True concurrent token yielding
            async for item in self._stream_concurrent(
                query=query,
                seed_nodes=seed_nodes,
                graph=graph,
                token_budget=token_budget,
                max_generation_tokens=max_generation_tokens,
                chunk_size=chunk_size,
                min_relevance=min_relevance,
                max_hops=max_hops,
                importance_scores=importance_scores,
                node_contents=node_contents,
                emit_metadata=emit_metadata,
                start_time=start_time
            ):
                yield item
            return

        # Phase 3 MVP: Batched token yielding (original implementation)
        # Start expansion stream
        expansion_stream = self.expansion_builder.stream_expansion(
            query=query,
            seed_nodes=seed_nodes,
            graph=graph,
            token_budget=token_budget,
            chunk_size=chunk_size,
            min_relevance=min_relevance,
            max_hops=max_hops,
            importance_scores=importance_scores,
            node_contents=node_contents
        )

        # Collect first chunk to start generation
        first_chunk = None
        context_chunks = []
        cumulative_context = ""

        async for chunk in expansion_stream:
            # Yield chunk immediately
            yield chunk
            context_chunks.append(chunk)

            # Update context
            for node_id, content in chunk.contents.items():
                if content and content not in cumulative_context:
                    cumulative_context += f"\n{content}"

            # Start generation after first chunk
            if first_chunk is None and not chunk.is_final:
                first_chunk = chunk
                break  # Break to start generation in parallel

        # If we got a first chunk, start generation in background
        generation_tokens = []
        if first_chunk:
            if emit_metadata:
                yield StreamMetadata("generation_start", {
                    "context_tokens": first_chunk.cumulative_tokens,
                    "first_chunk_size": first_chunk.token_count
                })

            # Start generation in background task
            gen_stream = self.llm.generate_stream(
                prompt=query,
                context=cumulative_context,
                max_tokens=max_generation_tokens
            )

            # Collect tokens in background
            async def collect_generation():
                tokens = []
                async for token in gen_stream:
                    tokens.append(token)
                return tokens

            generation_task = asyncio.create_task(collect_generation())

            # Continue expansion (this happens WHILE generation runs)
            async for chunk in expansion_stream:
                yield chunk
                context_chunks.append(chunk)

                # Update context (for potential future enhancement)
                for node_id, content in chunk.contents.items():
                    if content and content not in cumulative_context:
                        cumulative_context += f"\n{content}"

            if emit_metadata:
                yield StreamMetadata("expansion_complete", {
                    "total_chunks": len(context_chunks),
                    "total_tokens": sum(c.token_count for c in context_chunks)
                })

            # Wait for generation to complete
            generation_tokens = await generation_task

            # Yield all generation tokens
            cumulative_text = ""
            for token_index, token in enumerate(generation_tokens):
                cumulative_text += token
                gen_token = GenerationToken(
                    token=token,
                    cumulative_text=cumulative_text,
                    token_index=token_index,
                    is_final=False,
                    metadata={"context_chunks": len(context_chunks)}
                )
                yield gen_token

            # Yield final generation token
            if len(generation_tokens) > 0:
                final_token = GenerationToken(
                    token="",
                    cumulative_text=cumulative_text,
                    token_index=len(generation_tokens),
                    is_final=True,
                    metadata={
                        "context_chunks": len(context_chunks),
                        "total_context_tokens": sum(c.token_count for c in context_chunks)
                    }
                )
                yield final_token

                if emit_metadata:
                    yield StreamMetadata("generation_complete", {
                        "total_tokens": len(generation_tokens),
                        "response_length": len(cumulative_text)
                    })

        else:
            # No first chunk, just consume remaining expansion
            async for chunk in expansion_stream:
                yield chunk
                context_chunks.append(chunk)

        if emit_metadata:
            total_time = (time.time() - start_time) * 1000
            yield StreamMetadata("stream_complete", {
                "total_time_ms": total_time,
                "context_chunks": len(context_chunks),
                "generation_tokens": len(generation_tokens)
            })

    async def _stream_concurrent(
        self,
        query: str,
        seed_nodes: List[str],
        graph: Any,
        token_budget: int,
        max_generation_tokens: int,
        chunk_size: int,
        min_relevance: float,
        max_hops: int,
        importance_scores: Optional[Dict[str, float]],
        node_contents: Optional[Dict[str, str]],
        emit_metadata: bool,
        start_time: float
    ) -> AsyncIterator[StreamItem]:
        """
        Phase 4: True concurrent token yielding implementation.

        Uses async queues to interleave expansion chunks and generation tokens
        as they become available, achieving <100ms latency to first token.
        """
        # Create shared queue for both streams
        output_queue: asyncio.Queue = asyncio.Queue()

        # Track state
        context_chunks = []
        cumulative_context = ""
        generation_tokens = []
        generation_started = False

        # Start expansion stream
        expansion_stream = self.expansion_builder.stream_expansion(
            query=query,
            seed_nodes=seed_nodes,
            graph=graph,
            token_budget=token_budget,
            chunk_size=chunk_size,
            min_relevance=min_relevance,
            max_hops=max_hops,
            importance_scores=importance_scores,
            node_contents=node_contents
        )

        # Track generation task for proper cleanup
        generation_task = None

        async def pump_expansion():
            """Pump expansion chunks into output queue."""
            nonlocal generation_started, cumulative_context, generation_task
            first_chunk = None

            try:
                async for chunk in expansion_stream:
                    # Add to output queue
                    await output_queue.put(("chunk", chunk))
                    context_chunks.append(chunk)

                    # Update context
                    for node_id, content in chunk.contents.items():
                        if content and content not in cumulative_context:
                            cumulative_context += f"\n{content}"

                    # Start generation after first chunk
                    if first_chunk is None and not chunk.is_final and not generation_started:
                        first_chunk = chunk
                        generation_started = True

                        if emit_metadata:
                            await output_queue.put(("metadata", StreamMetadata("generation_start", {
                                "context_tokens": first_chunk.cumulative_tokens,
                                "first_chunk_size": first_chunk.token_count
                            })))

                        # Start generation task (track for cleanup)
                        generation_task = asyncio.create_task(pump_generation(cumulative_context))

                if emit_metadata:
                    await output_queue.put(("metadata", StreamMetadata("expansion_complete", {
                        "total_chunks": len(context_chunks),
                        "total_tokens": sum(c.token_count for c in context_chunks)
                    })))

            finally:
                await output_queue.put(("expansion_done", None))

        async def pump_generation(context: str):
            """Pump generation tokens into output queue."""
            try:
                gen_stream = self.llm.generate_stream(
                    prompt=query,
                    context=context,
                    max_tokens=max_generation_tokens
                )

                cumulative_text = ""
                token_index = 0

                async for token in gen_stream:
                    cumulative_text += token
                    gen_token = GenerationToken(
                        token=token,
                        cumulative_text=cumulative_text,
                        token_index=token_index,
                        is_final=False,
                        metadata={"context_chunks": len(context_chunks)}
                    )
                    await output_queue.put(("token", gen_token))
                    generation_tokens.append(token)
                    token_index += 1

                # Yield final token
                if len(generation_tokens) > 0:
                    final_token = GenerationToken(
                        token="",
                        cumulative_text=cumulative_text,
                        token_index=len(generation_tokens),
                        is_final=True,
                        metadata={
                            "context_chunks": len(context_chunks),
                            "total_context_tokens": sum(c.token_count for c in context_chunks)
                        }
                    )
                    await output_queue.put(("token", final_token))

                    if emit_metadata:
                        await output_queue.put(("metadata", StreamMetadata("generation_complete", {
                            "total_tokens": len(generation_tokens),
                            "response_length": len(cumulative_text)
                        })))

            finally:
                await output_queue.put(("generation_done", None))

        # Start expansion task
        expansion_task = asyncio.create_task(pump_expansion())

        # Consume from output queue
        expansion_done = False
        generation_done = False

        while not (expansion_done and generation_done):
            item_type, item = await output_queue.get()

            if item_type == "expansion_done":
                expansion_done = True
            elif item_type == "generation_done":
                generation_done = True
            elif item_type == "chunk":
                yield item
            elif item_type == "token":
                yield item
            elif item_type == "metadata":
                yield item

        # Wait for both tasks to complete
        await expansion_task
        if generation_task is not None:
            await generation_task

        if emit_metadata:
            import time
            total_time = (time.time() - start_time) * 1000
            yield StreamMetadata("stream_complete", {
                "total_time_ms": total_time,
                "context_chunks": len(context_chunks),
                "generation_tokens": len(generation_tokens)
            })

    async def _interleave_streams(
        self,
        expansion_stream: AsyncIterator[ContextChunk],
        generation_stream: AsyncIterator[str]
    ) -> AsyncIterator[Union[ContextChunk, str]]:
        """
        Interleave two async streams using task-based concurrency.

        Yields items from both streams as they become available.
        """
        # Convert iterators to async queues
        expansion_queue: asyncio.Queue = asyncio.Queue()
        generation_queue: asyncio.Queue = asyncio.Queue()

        async def pump_expansion():
            try:
                async for chunk in expansion_stream:
                    await expansion_queue.put(chunk)
            finally:
                await expansion_queue.put(None)  # Sentinel

        async def pump_generation():
            try:
                async for token in generation_stream:
                    await generation_queue.put(token)
            finally:
                await generation_queue.put(None)  # Sentinel

        # Start both tasks with proper exception handling
        expansion_task = None
        generation_task = None

        try:
            expansion_task = asyncio.create_task(pump_expansion())
            generation_task = asyncio.create_task(pump_generation())

            # Consume from both queues
            expansion_done = False
            generation_done = False

            while not (expansion_done and generation_done):
                # Try to get from both queues with timeout
                tasks = []

                if not expansion_done:
                    tasks.append(asyncio.create_task(expansion_queue.get()))
                if not generation_done:
                    tasks.append(asyncio.create_task(generation_queue.get()))

                if not tasks:
                    break

                # Wait for first item available with timeout
                try:
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=30.0  # 30s timeout to prevent hanging
                    )
                except asyncio.TimeoutError:
                    # Timeout - cancel tasks and raise
                    for task in tasks:
                        task.cancel()
                    raise

                # Cancel pending
                for task in pending:
                    task.cancel()

                # Process done tasks
                for task in done:
                    try:
                        item = task.result()

                        if item is None:
                            # Sentinel - stream done
                            if isinstance(item, type(None)):
                                # Check which stream ended
                                try:
                                    # Try non-blocking get to see if it's from expansion or generation
                                    if not expansion_queue.empty():
                                        expansion_done = True
                                    if not generation_queue.empty():
                                        generation_done = True
                                except:
                                    pass
                        elif isinstance(item, ContextChunk):
                            expansion_done = item.is_final
                            yield item
                        else:
                            # String token
                            yield item
                    except Exception as e:
                        # Handle task exceptions
                        logger.error(f"Task error in concurrent streaming: {e}", exc_info=True)
                        raise

            # Wait for tasks to complete with proper exception handling
            if expansion_task and generation_task:
                # Use asyncio.gather for proper exception propagation
                try:
                    await asyncio.gather(expansion_task, generation_task, return_exceptions=False)
                except Exception as e:
                    logger.error(f"Error waiting for concurrent tasks: {e}", exc_info=True)
                    raise

        except Exception as e:
            # Cleanup on error: cancel all tasks
            if expansion_task and not expansion_task.done():
                expansion_task.cancel()
            if generation_task and not generation_task.done():
                generation_task.cancel()

            # Wait for cancellation
            if expansion_task:
                try:
                    await expansion_task
                except asyncio.CancelledError:
                    pass
            if generation_task:
                try:
                    await generation_task
                except asyncio.CancelledError:
                    pass

            raise

    async def _collect_generation_tokens(self, gen_stream: AsyncIterator[str]) -> List[str]:
        """
        Collect all tokens from generation stream.

        Helper for non-blocking token consumption.
        """
        tokens = []
        async for token in gen_stream:
            tokens.append(token)
        return tokens

    def get_last_result(self) -> Optional[InterleavedResult]:
        """
        Get summary of last interleaved stream.

        Returns:
            InterleavedResult with complete metrics
        """
        # TODO: Track metrics during streaming
        return None


# ============================================================================
# Convenience Function
# ============================================================================

async def stream_interleaved_expansion_generation(
    query: str,
    seed_nodes: List[str],
    graph: Any,
    llm: Optional[LLMProtocol] = None,
    token_budget: int = 2000,
    max_generation_tokens: int = 500,
    stream_mode: StreamMode = StreamMode.BATCHED,
    **kwargs
) -> AsyncIterator[StreamItem]:
    """
    Convenience function for interleaved expansion + generation.

    Args:
        query: User query
        seed_nodes: Starting nodes
        graph: Knowledge graph
        llm: LLM provider (default: MockLLM)
        token_budget: Token budget for expansion
        max_generation_tokens: Max tokens to generate
        stream_mode: BATCHED (Phase 3) or CONCURRENT (Phase 4)
        **kwargs: Additional arguments for stream_interleaved

    Yields:
        ContextChunk, GenerationToken, or StreamMetadata objects
    """
    llm = llm or MockLLM()
    manager = InterleavedStreamManager(llm)

    async for item in manager.stream_interleaved(
        query=query,
        seed_nodes=seed_nodes,
        graph=graph,
        token_budget=token_budget,
        max_generation_tokens=max_generation_tokens,
        stream_mode=stream_mode,
        **kwargs
    ):
        yield item
