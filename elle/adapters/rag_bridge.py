"""RAG Bridge for Elle - Connects HoloLoom's RAG system to Elle.

Enables Elle to:
1. Store and retrieve knowledge from HoloLoom's memory system
2. Use RAG-enhanced responses with source citations
3. Build context-aware prompts with retrieved information

Example:
    async with ElleRAGBridge() as rag:
        # Ingest knowledge
        await rag.ingest("Thompson Sampling balances exploration and exploitation")

        # Query with RAG
        result = await rag.query("What is Thompson Sampling?")
        print(result.response)
        print(f"Sources: {len(result.sources)}")

Created: 2025-11-29
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Context retrieved from RAG for prompt building."""
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_section(self, max_sources: int = 3) -> str:
        """
        Format RAG context for prompt injection.

        Returns:
            Formatted context string for prompt
        """
        if not self.sources:
            return ""

        sources_text = "\n".join(
            f"- {source[:200]}..." if len(source) > 200 else f"- {source}"
            for source in self.sources[:max_sources]
        )

        return f"""
RELEVANT CONTEXT (confidence: {self.confidence:.0%}):
{sources_text}
"""


class ElleRAGBridge:
    """
    Bridge connecting Elle to HoloLoom's RAG system.

    Provides knowledge storage and retrieval for context-aware Elle responses.
    Gracefully degrades if HoloLoom RAG is unavailable.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:3b",
        enable_caching: bool = True,
    ):
        """
        Initialize RAG bridge.

        Args:
            llm_provider: LLM provider for RAG ("ollama", "anthropic", "openai")
            llm_model: Model to use for generation
            enable_caching: Enable query caching for repeated queries
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.enable_caching = enable_caching

        self._rag = None
        self._available = False
        self._last_context: Optional[RAGContext] = None

    async def __aenter__(self):
        """Initialize RAG system."""
        try:
            from HoloLoom.rag.simple_rag import SimpleRAG
            from HoloLoom.config import Config

            config = Config.fast()

            self._rag = SimpleRAG(
                config=config,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                enable_caching=self.enable_caching,
            )
            await self._rag.__aenter__()
            self._available = True
            logger.info(f"RAG bridge initialized with {self.llm_provider}/{self.llm_model}")

        except ImportError as e:
            logger.warning(f"HoloLoom RAG not available: {e}")
            logger.warning("Elle will operate without RAG knowledge retrieval")
            self._available = False

        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            logger.warning("Elle will operate without RAG knowledge retrieval")
            self._available = False

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup RAG system."""
        if self._rag:
            try:
                await self._rag.__aexit__(exc_type, exc_val, exc_tb)
                logger.info("RAG bridge cleaned up")
            except Exception as e:
                logger.error(f"Error closing RAG: {e}")

    @property
    def available(self) -> bool:
        """Check if RAG is available."""
        return self._available

    async def ingest(self, content: str) -> bool:
        """
        Ingest content into RAG knowledge base.

        Args:
            content: Text content to store

        Returns:
            True if ingestion succeeded
        """
        if not self._available or not self._rag:
            logger.debug("RAG not available, skipping ingest")
            return False

        try:
            await self._rag.ingest(content)
            logger.debug(f"Ingested: {content[:50]}...")
            return True
        except Exception as e:
            logger.warning(f"RAG ingest failed: {e}")
            return False

    async def ingest_batch(self, contents: List[str]) -> int:
        """
        Ingest multiple items into RAG.

        Args:
            contents: List of text contents

        Returns:
            Number of successfully ingested items
        """
        if not self._available:
            return 0

        count = 0
        for content in contents:
            if await self.ingest(content):
                count += 1
        return count

    async def get_context(
        self,
        query: str,
        max_sources: int = 5,
    ) -> RAGContext:
        """
        Retrieve relevant context for a query.

        Args:
            query: User's query text
            max_sources: Maximum number of sources to retrieve

        Returns:
            RAGContext with retrieved sources and metadata
        """
        if not self._available or not self._rag:
            return RAGContext(query=query)

        try:
            result = await self._rag.query(
                question=query,
                mode="direct",  # Fast retrieval
                max_sources=max_sources,
            )

            context = RAGContext(
                sources=result.sources,
                confidence=result.confidence,
                query=query,
                metadata={
                    "reasoning_mode": result.reasoning_mode,
                    "llm_provider": self.llm_provider,
                    **result.metadata,
                }
            )

            self._last_context = context
            return context

        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return RAGContext(query=query)

    async def query_with_response(
        self,
        query: str,
        max_sources: int = 5,
        mode: str = "verify",
    ) -> Dict[str, Any]:
        """
        Full RAG query with LLM-generated response.

        Args:
            query: User's query
            max_sources: Maximum sources to retrieve
            mode: Reasoning mode ("direct", "verify", "research")

        Returns:
            Dictionary with response, sources, confidence
        """
        if not self._available or not self._rag:
            return {
                "response": None,
                "sources": [],
                "confidence": 0.0,
                "rag_available": False,
            }

        try:
            result = await self._rag.query(
                question=query,
                mode=mode,
                max_sources=max_sources,
            )

            return {
                "response": result.response,
                "sources": result.sources,
                "confidence": result.confidence,
                "reasoning_mode": result.reasoning_mode,
                "rag_available": True,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return {
                "response": None,
                "sources": [],
                "confidence": 0.0,
                "rag_available": True,
                "error": str(e),
            }

    def get_last_context(self) -> Optional[RAGContext]:
        """Get the last retrieved context."""
        return self._last_context

    def get_metrics(self) -> Dict[str, Any]:
        """Get RAG system metrics."""
        if not self._available or not self._rag:
            return {"available": False}

        try:
            metrics = self._rag.get_metrics()
            metrics["available"] = True
            return metrics
        except Exception:
            return {"available": True, "error": "Failed to get metrics"}

    def clear_cache(self) -> None:
        """Clear RAG query cache."""
        if self._rag:
            self._rag.clear_cache()


def create_rag_bridge(
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2:3b",
    enable_caching: bool = True,
) -> ElleRAGBridge:
    """
    Factory function for creating RAG bridge.

    Args:
        llm_provider: LLM provider ("ollama", "anthropic", "openai")
        llm_model: Model name
        enable_caching: Enable query caching

    Returns:
        Configured ElleRAGBridge instance
    """
    return ElleRAGBridge(
        llm_provider=llm_provider,
        llm_model=llm_model,
        enable_caching=enable_caching,
    )
