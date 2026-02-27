"""
Streaming Response Support for HoloLoom RAG
===========================================

Enables token-by-token LLM generation for real-time user experience.

Features:
- Token-by-token streaming for supported LLM providers
- Automatic fallback to regular queries if streaming unavailable
- Cache full response after streaming completes
- Metadata tracking: latency, tokens per second
- Only works with "direct" reasoning mode

Usage:
    async with SimpleRAG() as rag:
        async for token in rag.query_stream("What is Thompson Sampling?"):
            print(token.text, end='', flush=True)

        # Last token has is_final=True and full metadata
"""

from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple
import time
import logging

from HoloLoom.protocols.types import Query

logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    """
    Single token from streaming LLM response.

    Attributes:
        text: The token text (usually 1-4 characters)
        index: Token index in response (0-based)
        cumulative_text: All tokens concatenated so far
        metadata: Dict with latency_ms, tokens_per_sec, etc.
        is_final: True for the last token
    """
    text: str
    index: int
    cumulative_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StreamToken(index={self.index}, len={len(self.cumulative_text)}, "
            f"final={self.is_final})"
        )


class StreamingError(Exception):
    """Error during streaming."""
    pass


async def stream_from_orchestrator(
    orchestrator: Any,
    query: Query,
    context_sources: List[str],
) -> AsyncGenerator[StreamToken, None]:
    """
    Stream tokens from orchestrator's LLM provider.

    Attempts to stream from various LLM providers with graceful fallback.

    Args:
        orchestrator: WeavingOrchestrator instance with LLM support
        query: Query object
        context_sources: Retrieved source texts for context

    Yields:
        StreamToken for each token from LLM

    Raises:
        StreamingError: If streaming completely unavailable
    """

    if orchestrator is None:
        raise StreamingError("Orchestrator not available")

    if not hasattr(orchestrator, 'llm') or orchestrator.llm is None:
        raise StreamingError("LLM not initialized in orchestrator")

    # Get LLM provider
    llm_provider = getattr(orchestrator, 'llm_provider', 'unknown')
    llm = orchestrator.llm

    cumulative_text = ""
    token_index = 0
    start_time = time.time()

    try:
        # Try to stream from LLM
        stream_iter = None

        if llm_provider == "ollama" and hasattr(llm, 'stream_generate'):
            # Ollama streaming
            logger.debug("Streaming from Ollama LLM")
            stream_iter = await llm.stream_generate(query.text)

        elif llm_provider == "anthropic" and hasattr(llm, 'messages_stream'):
            # Anthropic streaming
            logger.debug("Streaming from Anthropic LLM")
            stream_iter = await llm.messages_stream(query.text)

        elif llm_provider == "openai" and hasattr(llm, 'create_chat_completion_stream'):
            # OpenAI streaming
            logger.debug("Streaming from OpenAI LLM")
            stream_iter = await llm.create_chat_completion_stream(query.text)

        else:
            raise StreamingError(f"Streaming not supported for {llm_provider}")

        # Yield tokens as they arrive
        if stream_iter is not None:
            async for chunk_text in stream_iter:
                elapsed_sec = time.time() - start_time
                tokens_per_sec = token_index / elapsed_sec if elapsed_sec > 0 else 0

                token = StreamToken(
                    text=chunk_text,
                    index=token_index,
                    cumulative_text=cumulative_text + chunk_text,
                    metadata={
                        'latency_ms': elapsed_sec * 1000,
                        'tokens_per_sec': tokens_per_sec,
                        'llm_provider': llm_provider,
                    },
                    is_final=False
                )

                cumulative_text += chunk_text
                token_index += 1
                yield token

        # Emit final token
        if token_index > 0:
            final_elapsed = time.time() - start_time
            final_tps = token_index / final_elapsed if final_elapsed > 0 else 0

            yield StreamToken(
                text="",
                index=token_index,
                cumulative_text=cumulative_text,
                metadata={
                    'latency_ms': final_elapsed * 1000,
                    'tokens_per_sec': final_tps,
                    'total_tokens': token_index,
                    'llm_provider': llm_provider,
                    'is_final': True,
                },
                is_final=True
            )

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise StreamingError(f"Failed to stream from {llm_provider}: {e}")


class StreamingRAGMixin:
    """
    Mixin to add streaming support to SimpleRAG.

    This is mixed into SimpleRAG to add query_stream() method.
    """

    async def query_stream(
        self,
        question: str,
        mode: str = "direct",
        max_sources: int = 5,
    ) -> AsyncGenerator[StreamToken, None]:
        """
        Stream response token-by-token from LLM.

        Only works with mode="direct" (other modes require multiple LLM calls).
        Automatically falls back to regular query() if streaming unavailable.

        Process:
        1. Skip cache (can't stream cached results)
        2. Recall relevant memories
        3. Stream LLM generation token-by-token
        4. Cache full response when complete

        Args:
            question: Query text
            mode: Reasoning mode (only "direct" supports streaming)
            max_sources: Maximum source documents to retrieve

        Yields:
            StreamToken for each token from LLM

        Example:
            async with SimpleRAG() as rag:
                await rag.ingest("Thompson Sampling uses Bayesian statistics")

                print("Streaming response:", end=' ', flush=True)
                async for token in rag.query_stream("What is Thompson Sampling?"):
                    print(token.text, end='', flush=True)
                print()

        Note:
            - Only "direct" mode supports streaming
            - Falls back to regular query() if LLM unavailable
            - Full response cached after streaming completes
        """

        if self.loom is None:
            raise RuntimeError("SimpleRAG not initialized. Use: async with SimpleRAG() as rag:")

        # Only direct mode supports streaming
        if mode != "direct":
            logger.warning(f"⚠ Streaming only works with mode='direct', got {mode}")
            # Fall back to regular query and stream tokens
            result = await self.query(question, mode=mode, max_sources=max_sources)

            # Emit all tokens at once
            cumulative = ""
            for i, word in enumerate(result.response.split()):
                token = StreamToken(
                    text=word + " " if i < result.response.split().__len__() - 1 else word,
                    index=i,
                    cumulative_text=cumulative + word,
                    metadata={'mode': mode, 'fallback': True},
                    is_final=(i == len(result.response.split()) - 1)
                )
                cumulative += word + " "
                yield token
            return

        logger.debug(f"Streaming query: {question[:50]}...")

        # 1. Skip cache for streaming (can't partially stream cached results)

        # 2. Retrieve memories
        logger.debug(f"Retrieving memories for streaming query...")
        memories = await self.loom.recall(question, limit=max_sources)
        sources = [mem.text for mem in memories[:max_sources]]
        logger.debug(f"Retrieved {len(sources)} sources")

        # 3. Stream from LLM
        if self.orchestrator and sources:
            try:
                # Stream tokens from orchestrator
                query_obj = Query(text=question)
                async for token in stream_from_orchestrator(
                    self.orchestrator,
                    query_obj,
                    sources
                ):
                    yield token

                # Cache full response after streaming
                if token.is_final and self.enable_caching:
                    cache_key = (question, mode)
                    from HoloLoom.rag.simple_rag import RAGResult
                    result = RAGResult(
                        response=token.cumulative_text,
                        sources=sources,
                        confidence=0.8,  # Assume good confidence for streamed responses
                        reasoning_mode=mode,
                        metadata={
                            'n_sources': len(sources),
                            'llm_provider': self.llm_provider if self.orchestrator else None,
                            'cache_hit': False,
                            'streamed': True,
                            'total_tokens': token.metadata.get('total_tokens', 0),
                        }
                    )
                    self._cache[cache_key] = result
                    logger.debug(f"Cached streamed response ({len(token.cumulative_text)} chars)")

            except StreamingError as e:
                logger.warning(f"⚠ Streaming failed: {e}, falling back to regular query")

                # Fall back to regular query
                result = await self.query(question, mode=mode, max_sources=max_sources)

                # Emit fallback tokens
                cumulative = ""
                words = result.response.split()
                for i, word in enumerate(words):
                    space = " " if i < len(words) - 1 else ""
                    token = StreamToken(
                        text=word + space,
                        index=i,
                        cumulative_text=cumulative + word + space,
                        metadata={'fallback': True, 'fallback_reason': str(e)},
                        is_final=(i == len(words) - 1)
                    )
                    cumulative += word + space
                    yield token

        else:
            # No orchestrator or no sources - emit empty stream or fallback
            if not sources:
                logger.warning("No sources found, cannot stream")
                return

            logger.warning("No LLM orchestrator available, cannot stream")
            return
