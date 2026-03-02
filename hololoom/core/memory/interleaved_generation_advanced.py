"""
Advanced Interleaved Generation Features
=========================================

Extends Phase 4 with three advanced capabilities:

1. **Adaptive Context Feeding**: Update LLM with new chunks during generation
2. **Agentic Graph Navigation**: LLM can request specific nodes by name
3. **Context Summarization**: Compress less relevant chunks to save tokens

These features enable the LLM to actively navigate the knowledge graph and
incorporate new information mid-generation, creating a truly interactive
reasoning loop.

Architecture:
- AdaptiveLLMProtocol: Extended protocol with context update support
- AgenticGraphNavigator: Detects and handles node requests from LLM
- ContextSummarizer: Compresses less important chunks
- AdvancedInterleavedManager: Orchestrator integrating all three features

Author: Claude Code (Opus 4.5)
Date: 2025-11-25
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

from hololoom.memory.interleaved_generation import (
    ContextChunk,
    GenerationToken,
    InterleavedResult,
    LLMProtocol,
    StreamItem,
    StreamMetadata,
    StreamMode
)
from hololoom.memory.streaming_expansion import stream_context_expansion


# ============================================================================
# Feature 1: Adaptive Context Feeding
# ============================================================================

class AdaptiveLLMProtocol(LLMProtocol):
    """
    Extended LLM protocol supporting mid-generation context updates.

    This allows new context chunks to be fed to the LLM while it's still
    generating, enabling it to incorporate new information discovered during
    ongoing graph expansion.
    """

    async def generate_stream_adaptive(
        self,
        prompt: str,
        initial_context: str,
        context_updates: asyncio.Queue,  # Queue of new context chunks
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """
        Stream generation with adaptive context updates.

        Args:
            prompt: The user query/prompt
            initial_context: Initial context to start generation
            context_updates: Queue receiving new ContextChunk objects
            max_tokens: Maximum tokens to generate

        Yields:
            Individual tokens as strings

        The LLM monitors context_updates queue and incorporates new
        chunks into its reasoning as they arrive.
        """
        raise NotImplementedError


@dataclass
class ContextUpdate:
    """
    A context update to send to the LLM mid-generation.

    Attributes:
        chunk: The new context chunk
        priority: Priority level (higher = more important)
        timestamp_ms: When this update was created
    """
    chunk: ContextChunk
    priority: float = 0.5
    timestamp_ms: float = 0.0


# ============================================================================
# Feature 2: Agentic Graph Navigation
# ============================================================================

class NodeRequestType(str, Enum):
    """Type of node request from LLM."""
    BY_NAME = "by_name"          # Request specific node by ID
    BY_RELATIONSHIP = "by_rel"   # Request nodes related to current node
    BY_QUERY = "by_query"        # Natural language query for nodes


@dataclass
class NodeRequest:
    """
    A request from the LLM for additional graph nodes.

    The LLM emits special tokens like:
        <request_node>thompson_sampling</request_node>
        <request_related>exploration</request_related>
        <request_query>What connects to Bayesian methods?</request_query>

    Attributes:
        request_type: Type of node request
        target: Target node ID, relationship type, or query text
        context: Current generation context where request was made
    """
    request_type: NodeRequestType
    target: str
    context: str = ""


class AgenticGraphNavigator:
    """
    Detects and handles node requests from LLM output.

    Parses the token stream for request markers, fetches requested nodes
    from the graph, and feeds them back as context updates.
    """

    # Regex patterns for detecting requests
    REQUEST_PATTERNS = {
        NodeRequestType.BY_NAME: re.compile(r'<request_node>(.*?)</request_node>'),
        NodeRequestType.BY_RELATIONSHIP: re.compile(r'<request_related>(.*?)</request_related>'),
        NodeRequestType.BY_QUERY: re.compile(r'<request_query>(.*?)</request_query>')
    }

    def __init__(self, graph: Any, node_contents: Dict[str, str]):
        """
        Initialize navigator.

        Args:
            graph: Knowledge graph (NetworkX MultiDiGraph)
            node_contents: Mapping of node_id -> content text
        """
        self.graph = graph
        self.node_contents = node_contents
        self.fulfilled_requests: Set[str] = set()  # Avoid duplicate fetches

    def detect_requests(self, cumulative_text: str) -> List[NodeRequest]:
        """
        Detect node requests in generated text.

        Args:
            cumulative_text: Full text generated so far

        Returns:
            List of detected NodeRequest objects
        """
        requests = []

        for request_type, pattern in self.REQUEST_PATTERNS.items():
            matches = pattern.findall(cumulative_text)
            for match in matches:
                target = match.strip()
                request_key = f"{request_type}:{target}"

                # Skip if already fulfilled
                if request_key not in self.fulfilled_requests:
                    requests.append(NodeRequest(
                        request_type=request_type,
                        target=target,
                        context=cumulative_text[-200:]  # Last 200 chars
                    ))
                    self.fulfilled_requests.add(request_key)

        return requests

    def fulfill_request(self, request: NodeRequest) -> Optional[ContextChunk]:
        """
        Fulfill a node request by fetching from graph.

        Args:
            request: The node request to fulfill

        Returns:
            ContextChunk with requested nodes, or None if not found
        """
        if request.request_type == NodeRequestType.BY_NAME:
            return self._fetch_by_name(request.target)
        elif request.request_type == NodeRequestType.BY_RELATIONSHIP:
            return self._fetch_by_relationship(request.target)
        elif request.request_type == NodeRequestType.BY_QUERY:
            return self._fetch_by_query(request.target)
        return None

    def _fetch_by_name(self, node_id: str) -> Optional[ContextChunk]:
        """Fetch a specific node by ID."""
        if node_id not in self.graph.nodes:
            logger.warning(f"Requested node '{node_id}' not found in graph")
            return None

        content = self.node_contents.get(node_id, "")
        if not content:
            logger.warning(f"Node '{node_id}' has no content")
            return None

        return ContextChunk(
            nodes=[node_id],
            relevance_scores={node_id: 1.0},  # Explicitly requested = high relevance
            contents={node_id: content},
            token_count=len(content.split()),
            cumulative_tokens=len(content.split()),
            hop_distance=0,  # Direct request
            chunk_index=0,
            is_final=False
        )

    def _fetch_by_relationship(self, relationship_type: str) -> Optional[ContextChunk]:
        """Fetch nodes connected by a specific relationship."""
        # Not implemented yet - would need current node context
        logger.warning(f"Relationship-based requests not yet implemented: {relationship_type}")
        return None

    def _fetch_by_query(self, query: str) -> Optional[ContextChunk]:
        """Fetch nodes matching a natural language query."""
        # Not implemented yet - would need semantic search
        logger.warning(f"Query-based requests not yet implemented: {query}")
        return None


# ============================================================================
# Feature 3: Context Summarization
# ============================================================================

class SummarizationStrategy(str, Enum):
    """Strategy for summarizing context chunks."""
    IMPORTANCE_BASED = "importance"  # Summarize low-importance chunks
    TOKEN_THRESHOLD = "token_threshold"  # Summarize when exceeds token limit
    DYNAMIC = "dynamic"  # Combine both strategies


@dataclass
class SummarizationConfig:
    """Configuration for context summarization."""
    enabled: bool = True
    strategy: SummarizationStrategy = SummarizationStrategy.DYNAMIC
    chunk_threshold: int = 10  # Summarize when >10 chunks
    token_threshold: int = 2000  # Summarize when >2000 tokens
    importance_cutoff: float = 0.4  # Summarize chunks with relevance <0.4
    compression_ratio: float = 0.3  # Target 30% of original size


class ContextSummarizer:
    """
    Compresses less important context chunks to save tokens.

    Uses importance scores to identify low-value chunks, then either:
    1. Drops them entirely (aggressive)
    2. Replaces with extractive summary (balanced)
    3. Generates LLM-based summary (quality)
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize summarizer.

        Args:
            config: Summarization configuration (default: SummarizationConfig())
        """
        self.config = config or SummarizationConfig()

    def should_summarize(self, chunks: List[ContextChunk]) -> bool:
        """
        Determine if summarization is needed.

        Args:
            chunks: List of context chunks

        Returns:
            True if summarization should be applied
        """
        if not self.config.enabled:
            return False

        total_tokens = sum(chunk.token_count for chunk in chunks)

        if self.config.strategy == SummarizationStrategy.IMPORTANCE_BASED:
            # Count low-importance chunks
            low_importance = sum(
                1 for chunk in chunks
                if chunk.avg_relevance < self.config.importance_cutoff
            )
            return low_importance >= 3  # At least 3 low-importance chunks

        elif self.config.strategy == SummarizationStrategy.TOKEN_THRESHOLD:
            return total_tokens > self.config.token_threshold

        else:  # DYNAMIC
            return (
                len(chunks) > self.config.chunk_threshold or
                total_tokens > self.config.token_threshold
            )

    def summarize(self, chunks: List[ContextChunk]) -> Tuple[List[ContextChunk], ContextChunk]:
        """
        Summarize low-importance chunks into a single compressed chunk.

        Args:
            chunks: List of context chunks to summarize

        Returns:
            (kept_chunks, summarized_chunk) tuple where:
            - kept_chunks: High-importance chunks to keep verbatim
            - summarized_chunk: Compressed summary of low-importance chunks
        """
        # Sort by average relevance
        sorted_chunks = sorted(chunks, key=lambda c: c.avg_relevance, reverse=True)

        # Determine cutoff point
        cutoff_index = int(len(sorted_chunks) * (1 - self.config.compression_ratio))
        cutoff_index = max(1, cutoff_index)  # Keep at least 1 chunk

        kept_chunks = sorted_chunks[:cutoff_index]
        to_summarize = sorted_chunks[cutoff_index:]

        if not to_summarize:
            # Nothing to summarize
            return chunks, None

        # Create summarized chunk (extractive summary)
        all_nodes = []
        all_relevance = {}
        all_contents = {}

        for chunk in to_summarize:
            all_nodes.extend(chunk.nodes)
            all_relevance.update(chunk.relevance_scores)
            all_contents.update(chunk.contents)

        # Extract key sentences (simple extractive summary)
        all_text = " ".join(all_contents.values())
        sentences = all_text.split(". ")

        # Keep first sentence from each node (simple heuristic)
        summary_text = ". ".join([
            sentences[i] if i < len(sentences) else ""
            for i in range(len(all_nodes))
        ])

        # Create summary chunk
        summarized_chunk = ContextChunk(
            nodes=all_nodes,
            relevance_scores=all_relevance,
            contents={"summary": f"[SUMMARY OF {len(all_nodes)} NODES]: {summary_text}"},
            token_count=len(summary_text.split()),
            cumulative_tokens=sum(c.cumulative_tokens for c in chunks),
            hop_distance=max(c.hop_distance for c in to_summarize),
            chunk_index=-1,  # Special marker for summary
            is_final=False
        )

        logger.info(
            f"Summarized {len(to_summarize)} chunks ({sum(c.token_count for c in to_summarize)} tokens) "
            f"into {summarized_chunk.token_count} tokens "
            f"({(1 - summarized_chunk.token_count / sum(c.token_count for c in to_summarize)):.1%} reduction)"
        )

        return kept_chunks, summarized_chunk


# ============================================================================
# Advanced Interleaved Manager
# ============================================================================

class AdvancedInterleavedManager:
    """
    Orchestrator integrating all three advanced features:
    1. Adaptive context feeding
    2. Agentic graph navigation
    3. Context summarization

    Extends the Phase 4 concurrent streaming with active graph navigation
    and intelligent context management.
    """

    def __init__(
        self,
        llm: AdaptiveLLMProtocol,
        navigator: Optional[AgenticGraphNavigator] = None,
        summarizer: Optional[ContextSummarizer] = None,
        enable_adaptive_feeding: bool = True,
        enable_agentic_navigation: bool = True,
        enable_summarization: bool = True
    ):
        """
        Initialize advanced manager.

        Args:
            llm: Adaptive LLM supporting context updates
            navigator: Agentic graph navigator (created if None)
            summarizer: Context summarizer (created if None)
            enable_adaptive_feeding: Enable adaptive context feeding
            enable_agentic_navigation: Enable agentic graph navigation
            enable_summarization: Enable context summarization
        """
        self.llm = llm
        self.navigator = navigator
        self.summarizer = summarizer or ContextSummarizer()

        self.enable_adaptive_feeding = enable_adaptive_feeding
        self.enable_agentic_navigation = enable_agentic_navigation
        self.enable_summarization = enable_summarization

    async def stream_advanced(
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
        emit_metadata: bool = False
    ) -> AsyncIterator[StreamItem]:
        import time
        start_time = time.time()

        if self.enable_agentic_navigation and self.navigator is None:
            if node_contents is None:
                raise ValueError("node_contents required for agentic navigation")
            self.navigator = AgenticGraphNavigator(graph, node_contents)

        expansion_stream = stream_context_expansion(
            query=query, seed_nodes=seed_nodes, graph=graph,
            token_budget=token_budget, chunk_size=chunk_size,
            min_relevance=min_relevance, max_hops=max_hops,
            importance_scores=importance_scores, node_contents=node_contents
        )

        context_updates: asyncio.Queue = asyncio.Queue()
        output_queue: asyncio.Queue = asyncio.Queue()
        all_chunks: List[ContextChunk] = []
        cumulative_context = ""
        generation_tokens: List[str] = []
        generation_started = False

        async def run_expansion():
            nonlocal cumulative_context, generation_started
            first_chunk = None
            try:
                async for chunk in expansion_stream:
                    all_chunks.append(chunk)
                    await output_queue.put(("chunk", chunk))
                    for node_id, content in chunk.contents.items():
                        if content and content not in cumulative_context:
                            cumulative_context += f"\n{content}"
                    if first_chunk is None and not chunk.is_final:
                        first_chunk = chunk
                        if self.enable_summarization and self.summarizer.should_summarize(all_chunks):
                            kept, summary = self.summarizer.summarize(all_chunks)
                            if summary:
                                cumulative_context = summary.contents.get("summary", "")
                                if emit_metadata:
                                    await output_queue.put(("metadata", StreamMetadata("summarization_applied", {
                                        "original_chunks": len(all_chunks), "kept_chunks": len(kept),
                                        "summary_tokens": summary.token_count,
                                        "compression_ratio": summary.token_count / sum(c.token_count for c in all_chunks)
                                    })))
                        generation_started = True
                        if emit_metadata:
                            await output_queue.put(("metadata", StreamMetadata("generation_start", {
                                "context_tokens": first_chunk.cumulative_tokens, "first_chunk_size": first_chunk.token_count
                            })))
                        asyncio.create_task(run_generation(cumulative_context))
                    elif generation_started and self.enable_adaptive_feeding:
                        await context_updates.put(ContextUpdate(chunk=chunk, priority=chunk.avg_relevance,
                            timestamp_ms=(time.time() - start_time) * 1000))
                if emit_metadata:
                    await output_queue.put(("metadata", StreamMetadata("expansion_complete", {
                        "total_chunks": len(all_chunks), "total_tokens": sum(c.token_count for c in all_chunks)
                    })))
            finally:
                if self.enable_adaptive_feeding:
                    await context_updates.put(None)
            await output_queue.put(("expansion_done", None))

        async def run_generation(initial_context: str):
            try:
                if self.enable_adaptive_feeding:
                    gen_stream = self.llm.generate_stream_adaptive(prompt=query, initial_context=initial_context,
                        context_updates=context_updates, max_tokens=max_generation_tokens)
                else:
                    gen_stream = self.llm.generate_stream(prompt=query, context=initial_context, max_tokens=max_generation_tokens)
                cumulative_text = ""
                token_index = 0
                async for token in gen_stream:
                    cumulative_text += token
                    generation_tokens.append(token)
                    if self.enable_agentic_navigation:
                        requests = self.navigator.detect_requests(cumulative_text)
                        for request in requests:
                            chunk = self.navigator.fulfill_request(request)
                            if chunk:
                                await context_updates.put(ContextUpdate(chunk=chunk, priority=1.0,
                                    timestamp_ms=(time.time() - start_time) * 1000))
                                if emit_metadata:
                                    await output_queue.put(("metadata", StreamMetadata("node_request_fulfilled", {
                                        "request_type": request.request_type, "target": request.target, "nodes_fetched": len(chunk.nodes)
                                    })))
                    gen_token = GenerationToken(token=token, cumulative_text=cumulative_text, token_index=token_index, is_final=False,
                        metadata={"context_chunks": len(all_chunks)})
                    await output_queue.put(("token", gen_token))
                    token_index += 1
                if generation_tokens:
                    await output_queue.put(("token", GenerationToken(token="", cumulative_text=cumulative_text,
                        token_index=len(generation_tokens), is_final=True,
                        metadata={"context_chunks": len(all_chunks), "total_context_tokens": sum(c.token_count for c in all_chunks)})))
                    if emit_metadata:
                        await output_queue.put(("metadata", StreamMetadata("generation_complete", {
                            "total_tokens": len(generation_tokens), "response_length": len(cumulative_text)
                        })))
            finally:
                await output_queue.put(("generation_done", None))

        expansion_task = asyncio.create_task(run_expansion())
        expansion_done = False
        generation_done = False

        while not (expansion_done and generation_done):
            try:
                item_type, item = await asyncio.wait_for(output_queue.get(), timeout=30.0)
                if item_type == "expansion_done":
                    expansion_done = True
                    if not generation_started:
                        generation_done = True
                elif item_type == "generation_done":
                    generation_done = True
                elif item_type in ("chunk", "token", "metadata"):
                    yield item
            except asyncio.TimeoutError:
                logger.warning("Stream timeout waiting for output")
                break

        await expansion_task

        if emit_metadata:
            total_time = (time.time() - start_time) * 1000
            yield StreamMetadata("stream_complete", {
                "total_time_ms": total_time, "context_chunks": len(all_chunks),
                "generation_tokens": len(generation_tokens),
                "adaptive_feeding_enabled": self.enable_adaptive_feeding,
                "agentic_navigation_enabled": self.enable_agentic_navigation,
                "summarization_enabled": self.enable_summarization
            })

# ============================================================================
# Convenience Functions
# ============================================================================

async def stream_advanced_generation(
    query: str,
    seed_nodes: List[str],
    graph: Any,
    llm: Optional[AdaptiveLLMProtocol] = None,
    token_budget: int = 2000,
    max_generation_tokens: int = 500,
    enable_adaptive_feeding: bool = True,
    enable_agentic_navigation: bool = True,
    enable_summarization: bool = True,
    **kwargs
) -> AsyncIterator[StreamItem]:
    """
    Convenience function for advanced interleaved generation.

    Args:
        query: User query
        seed_nodes: Starting nodes
        graph: Knowledge graph
        llm: Adaptive LLM (MockAdaptiveLLM used if None)
        token_budget: Maximum context tokens
        max_generation_tokens: Maximum generation tokens
        enable_adaptive_feeding: Enable adaptive context feeding
        enable_agentic_navigation: Enable agentic navigation
        enable_summarization: Enable context summarization
        **kwargs: Additional arguments passed to stream_advanced()

    Yields:
        StreamItem objects (ContextChunk, GenerationToken, StreamMetadata)
    """
    if llm is None:
        # Use mock LLM for testing
        llm = MockAdaptiveLLM()

    manager = AdvancedInterleavedManager(
        llm=llm,
        enable_adaptive_feeding=enable_adaptive_feeding,
        enable_agentic_navigation=enable_agentic_navigation,
        enable_summarization=enable_summarization
    )

    async for item in manager.stream_advanced(
        query=query,
        seed_nodes=seed_nodes,
        graph=graph,
        token_budget=token_budget,
        max_generation_tokens=max_generation_tokens,
        **kwargs
    ):
        yield item


# ============================================================================
# Mock Adaptive LLM (for testing)
# ============================================================================

class MockAdaptiveLLM(AdaptiveLLMProtocol):
    """
    Mock adaptive LLM for testing and demos.
    Simulates adaptive context feeding and node request generation.
    """

    def __init__(self, response_template: Optional[str] = None, tokens_per_second: int = 30):
        self.response_template = response_template
        self.tokens_per_second = tokens_per_second

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Standard generation (fallback)."""
        if self.response_template:
            response = self.response_template
        else:
            response = f"Based on the context, {prompt.lower()} involves multiple concepts."

        tokens = response.replace(",", " ,").replace(".", " .").split()
        delay = 1.0 / self.tokens_per_second

        for token in tokens[:max_tokens]:
            yield token + " "
            await asyncio.sleep(delay)

    async def generate_stream_adaptive(
        self,
        prompt: str,
        initial_context: str,
        context_updates: asyncio.Queue,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """
        Adaptive generation with context updates.

        Simulates LLM incorporating new chunks during generation.
        Occasionally emits node request markers.
        """
        # Start with initial response
        if self.response_template:
            response = self.response_template
        else:
            response = f"Based on the initial context, {prompt.lower()} involves "
            response += f"{initial_context[:50]}... "

        tokens = response.replace(",", " ,").replace(".", " .").split()
        delay = 1.0 / self.tokens_per_second

        token_count = 0

        # Yield tokens, checking for context updates
        for i, token in enumerate(tokens):
            # Check for context update
            try:
                update = context_updates.get_nowait()
                if update is not None:
                    # Incorporate new context
                    new_context_snippet = str(update.chunk.nodes[:2])  # Show first 2 nodes
                    additional = f"[New context: {new_context_snippet}] "
                    for extra_token in additional.split():
                        yield extra_token + " "
                        await asyncio.sleep(delay)
                        token_count += 1
                        if token_count >= max_tokens:
                            return
            except asyncio.QueueEmpty:
                pass

            # Occasionally emit node request (10% chance)
            if i % 10 == 5 and token_count < max_tokens - 5:
                # Emit a node request
                request = "<request_node>bayesian_approach</request_node> "
                for req_token in request.split():
                    yield req_token
                    await asyncio.sleep(delay / 2)  # Faster for special tokens
                    token_count += 1

            yield token + " "
            await asyncio.sleep(delay)
            token_count += 1

            if token_count >= max_tokens:
                return


# ============================================================================
# Production Adaptive LLMs (W1 Remediation - December 2025)
# ============================================================================

class OllamaAdaptiveLLM(AdaptiveLLMProtocol):
    """
    Production Ollama LLM with adaptive context feeding support.

    Wraps OllamaLLM.stream_generate() with context update awareness.
    Note: True mid-generation context injection requires LLM API support.
    Current implementation acknowledges updates but uses initial context.
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self._ollama_llm = None
        self._available = False

        try:
            from hololoom.memory.awareness.llm_integration import OllamaLLM
            self._ollama_llm = OllamaLLM(model=model, base_url=base_url)
            self._available = self._ollama_llm.is_available()
            if self._available:
                logger.info(f"OllamaAdaptiveLLM initialized (model={model})")
        except ImportError:
            logger.warning("OllamaLLM not available - adaptive features disabled")
            self._available = False

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Standard streaming generation."""
        if not self._available or self._ollama_llm is None:
            raise RuntimeError("Ollama not available")

        full_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context above:"""

        system_prompt = "You are a helpful assistant. Answer based on the provided context."

        async for token in self._ollama_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            yield token

    async def generate_stream_adaptive(
        self,
        prompt: str,
        initial_context: str,
        context_updates: asyncio.Queue,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """
        Adaptive streaming generation with context update awareness.

        Monitors context_updates queue and logs new chunks, but uses
        initial context for generation (true mid-generation injection
        would require LLM API support for context modification).
        """
        if not self._available or self._ollama_llm is None:
            raise RuntimeError("Ollama not available")

        full_prompt = f"""Context:
{initial_context}

Question: {prompt}

Answer based on the context above. If new information becomes available,
you may request specific nodes using <request_node>node_id</request_node>."""

        system_prompt = "You are a helpful assistant with access to a knowledge graph. Answer based on the provided context."

        token_count = 0
        context_update_count = 0

        async for token in self._ollama_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            # Check for context updates (non-blocking)
            try:
                update = context_updates.get_nowait()
                if update is not None:
                    context_update_count += 1
                    logger.debug(f"Context update received (#{context_update_count}): {len(update.chunk.nodes)} nodes")
                elif update is None:
                    # End of context stream
                    pass
            except asyncio.QueueEmpty:
                pass

            yield token
            token_count += 1

            if token_count >= max_tokens:
                break

        if context_update_count > 0:
            logger.info(f"Generation complete with {context_update_count} context updates received")


class AnthropicAdaptiveLLM(AdaptiveLLMProtocol):
    """
    Production Anthropic Claude LLM with adaptive context feeding support.

    Wraps AnthropicLLM.stream_generate() with context update awareness.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._anthropic_llm = None
        self._available = False

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Anthropic adaptive LLM unavailable")
            return

        try:
            from hololoom.memory.awareness.llm_integration import AnthropicLLM
            self._anthropic_llm = AnthropicLLM(api_key=self.api_key, model=model)
            self._available = True
            logger.info(f"AnthropicAdaptiveLLM initialized (model={model})")
        except ImportError:
            logger.warning("AnthropicLLM not available - adaptive features disabled")
            self._available = False

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return self._available

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Standard streaming generation."""
        if not self._available or self._anthropic_llm is None:
            raise RuntimeError("Anthropic not available")

        full_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context above:"""

        system_prompt = "You are a helpful assistant. Answer based on the provided context."

        async for token in self._anthropic_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            yield token

    async def generate_stream_adaptive(
        self,
        prompt: str,
        initial_context: str,
        context_updates: asyncio.Queue,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Adaptive streaming generation with context update awareness."""
        if not self._available or self._anthropic_llm is None:
            raise RuntimeError("Anthropic not available")

        full_prompt = f"""Context:
{initial_context}

Question: {prompt}

Answer based on the context above. If you need additional information,
you can request nodes using <request_node>node_id</request_node>."""

        system_prompt = "You are a helpful assistant with access to a knowledge graph."

        token_count = 0
        context_update_count = 0

        async for token in self._anthropic_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            try:
                update = context_updates.get_nowait()
                if update is not None:
                    context_update_count += 1
                    logger.debug(f"Context update received: {len(update.chunk.nodes)} nodes")
            except asyncio.QueueEmpty:
                pass

            yield token
            token_count += 1

            if token_count >= max_tokens:
                break


class OpenAIAdaptiveLLM(AdaptiveLLMProtocol):
    """
    Production OpenAI GPT LLM with adaptive context feeding support.

    Wraps OpenAILLM.stream_generate() with context update awareness.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._openai_llm = None
        self._available = False

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - OpenAI adaptive LLM unavailable")
            return

        try:
            from hololoom.memory.awareness.llm_integration import OpenAILLM
            self._openai_llm = OpenAILLM(api_key=self.api_key, model=model)
            self._available = True
            logger.info(f"OpenAIAdaptiveLLM initialized (model={model})")
        except ImportError:
            logger.warning("OpenAILLM not available - adaptive features disabled")
            self._available = False

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self._available

    async def generate_stream(
        self,
        prompt: str,
        context: str,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Standard streaming generation."""
        if not self._available or self._openai_llm is None:
            raise RuntimeError("OpenAI not available")

        full_prompt = f"""Context:
{context}

Question: {prompt}

Answer based on the context above:"""

        system_prompt = "You are a helpful assistant. Answer based on the provided context."

        async for token in self._openai_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            yield token

    async def generate_stream_adaptive(
        self,
        prompt: str,
        initial_context: str,
        context_updates: asyncio.Queue,
        max_tokens: int = 500
    ) -> AsyncIterator[str]:
        """Adaptive streaming generation with context update awareness."""
        if not self._available or self._openai_llm is None:
            raise RuntimeError("OpenAI not available")

        full_prompt = f"""Context:
{initial_context}

Question: {prompt}

Answer based on the context above. If you need additional information,
you can request nodes using <request_node>node_id</request_node>."""

        system_prompt = "You are a helpful assistant with access to a knowledge graph."

        token_count = 0
        context_update_count = 0

        async for token in self._openai_llm.stream_generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        ):
            try:
                update = context_updates.get_nowait()
                if update is not None:
                    context_update_count += 1
                    logger.debug(f"Context update received: {len(update.chunk.nodes)} nodes")
            except asyncio.QueueEmpty:
                pass

            yield token
            token_count += 1

            if token_count >= max_tokens:
                break


def create_adaptive_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> AdaptiveLLMProtocol:
    """
    Factory function to create an adaptive LLM with automatic fallback.

    Tries to create the requested provider, falls back to MockAdaptiveLLM
    if not available.

    Args:
        provider: LLM provider ("ollama", "anthropic", "openai", "mock")
        model: Model name (uses provider default if None)
        **kwargs: Additional provider-specific arguments

    Returns:
        AdaptiveLLMProtocol implementation

    Example:
        >>> llm = create_adaptive_llm("ollama")
        >>> async for token in llm.generate_stream_adaptive(prompt, context, queue):
        ...     print(token, end="", flush=True)
    """
    provider = provider.lower()

    if provider == "mock":
        return MockAdaptiveLLM(**kwargs)

    if provider == "ollama":
        llm = OllamaAdaptiveLLM(
            model=model or "llama3.2:3b",
            base_url=kwargs.get("base_url")
        )
        if llm.is_available():
            return llm
        logger.warning("Ollama not available, falling back to MockAdaptiveLLM")
        return MockAdaptiveLLM(**kwargs)

    if provider == "anthropic":
        llm = AnthropicAdaptiveLLM(
            api_key=kwargs.get("api_key"),
            model=model or "claude-3-5-sonnet-20241022"
        )
        if llm.is_available():
            return llm
        logger.warning("Anthropic not available, falling back to MockAdaptiveLLM")
        return MockAdaptiveLLM(**kwargs)

    if provider == "openai":
        llm = OpenAIAdaptiveLLM(
            api_key=kwargs.get("api_key"),
            model=model or "gpt-4"
        )
        if llm.is_available():
            return llm
        logger.warning("OpenAI not available, falling back to MockAdaptiveLLM")
        return MockAdaptiveLLM(**kwargs)

    logger.warning(f"Unknown provider '{provider}', falling back to MockAdaptiveLLM")
    return MockAdaptiveLLM(**kwargs)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Feature 1: Adaptive Context Feeding
    "AdaptiveLLMProtocol",
    "ContextUpdate",

    # Feature 2: Agentic Graph Navigation
    "NodeRequestType",
    "NodeRequest",
    "AgenticGraphNavigator",

    # Feature 3: Context Summarization
    "SummarizationStrategy",
    "SummarizationConfig",
    "ContextSummarizer",

    # Main Manager
    "AdvancedInterleavedManager",

    # Convenience
    "stream_advanced_generation",

    # LLM Implementations
    "MockAdaptiveLLM",
    "OllamaAdaptiveLLM",
    "AnthropicAdaptiveLLM",
    "OpenAIAdaptiveLLM",
    "create_adaptive_llm",
]