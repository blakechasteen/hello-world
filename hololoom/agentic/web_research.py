"""
Web-Enhanced Agentic Research
==============================
Extends AgenticOrchestrator with live web search capabilities.

Integrates MatryoshkaWebSearch for Perplexity-style research:
- Multi-query web exploration
- Cited responses with sources
- Memory shard persistence
- Complete provenance tracking

Usage:
    from hololoom.agentic.web_research import WebResearchOrchestrator

    orchestrator = await WebResearchOrchestrator.create(
        config=Config.fused(),
        enable_web_search=True,
        search_api_key="your_serpapi_key"
    )

    result = await orchestrator.research_web(
        query="What are the tradeoffs of Thompson Sampling?",
        max_web_results=10,
        max_steps=5
    )

    # Result includes:
    # - Cited response with [1], [2], [3]
    # - Search results as memory shards
    # - Complete reasoning trace
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from hololoom.protocols.types import Query, MemoryShard
from hololoom.config import Config
from hololoom.search import MatryoshkaWebSearch, SearchConfig, WebSearchResult
from hololoom.agentic.core import (
    AgenticOrchestrator,
    AgenticResult,
    AgenticIntent,
    ReasoningMode
)
from hololoom.recursive import FullLearningEngine
from hololoom.alignment.audit_trail import AuditTrail, DecisionType, OutcomeType

logger = logging.getLogger(__name__)


# ============================================================================
# Web Research Result
# ============================================================================

@dataclass
class WebResearchResult(AgenticResult):
    """
    Extended result with web search metadata.

    Includes everything from AgenticResult plus:
    - Search results with citations
    - Memory shards created
    - Web search statistics
    """
    search_results: List[WebSearchResult] = field(default_factory=list)
    memory_shards: List[MemoryShard] = field(default_factory=list)
    cited_response: str = ""
    web_search_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "response": self.cited_response or self.spacetime.response,
            "confidence": self.spacetime.confidence,
            "reasoning_mode": self.reasoning_mode.value,
            "steps_taken": self.steps_taken,
            "total_queries": self.total_queries,
            "total_duration_ms": self.total_duration_ms,
            "web_results_count": len(self.search_results),
            "memory_shards_count": len(self.memory_shards),
            "citations": [
                {
                    "index": i + 1,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet[:200],
                    "score": r.final_score
                }
                for i, r in enumerate(self.search_results)
            ]
        }


# ============================================================================
# Web Research Orchestrator
# ============================================================================

class WebResearchOrchestrator:
    """
    Agentic orchestrator with web search integration.

    Combines:
    - AgenticOrchestrator (multi-step reasoning)
    - MatryoshkaWebSearch (live web search)
    - Citation system (Perplexity-style)
    - Memory persistence (Neo4j + Qdrant)

    Key Features:
    - RESEARCH mode enhanced with live web search
    - Automatic citation of web sources
    - Memory shard creation from search results
    - Complete audit trail

    Usage:
        orchestrator = await WebResearchOrchestrator.create(
            config=Config.fused(),
            enable_web_search=True,
            search_api_key=os.getenv("SERPAPI_KEY")
        )

        # Web research mode
        result = await orchestrator.research_web(
            query="What is Thompson Sampling?",
            max_web_results=10
        )

        # Standard agentic modes also available
        result = await orchestrator.reason(
            Query(text="Explain Thompson Sampling"),
            mode=ReasoningMode.VERIFY
        )
    """

    def __init__(
        self,
        agentic_orchestrator: AgenticOrchestrator,
        web_search: Optional[MatryoshkaWebSearch] = None,
        enable_web_search: bool = True
    ):
        self.agentic = agentic_orchestrator
        self.web_search = web_search
        self.enable_web_search = enable_web_search and web_search is not None

        self.logger = logging.getLogger(__name__)

    @classmethod
    async def create(
        cls,
        config: Config,
        shards: List[MemoryShard],
        enable_web_search: bool = True,
        search_provider: str = "serpapi",
        search_api_key: Optional[str] = None,
        enable_verification: bool = True,
        enable_goal_tracking: bool = True,
        audit_trail: Optional[AuditTrail] = None
    ) -> "WebResearchOrchestrator":
        """
        Factory method to create web-enabled agentic orchestrator.

        Args:
            config: HoloLoom config
            shards: Initial memory shards
            enable_web_search: Enable live web search
            search_provider: Provider name ("serpapi", "tavily", "mock")
            search_api_key: API key for search provider
            enable_verification: Enable verification loops
            enable_goal_tracking: Enable goal/intent tracking
            audit_trail: Optional audit trail

        Returns:
            WebResearchOrchestrator instance
        """
        # Create learning engine
        learning_engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=True
        )
        await learning_engine.__aenter__()

        # Create agentic orchestrator
        agentic = AgenticOrchestrator(
            learning_engine=learning_engine,
            audit_trail=audit_trail or AuditTrail(),
            enable_verification=enable_verification,
            enable_goal_tracking=enable_goal_tracking
        )

        # Create web search if enabled
        web_search = None
        if enable_web_search:
            # Auto-detect mock mode if no API key
            if not search_api_key and search_provider != "mock":
                logger.warning("No search API key provided, using mock provider")
                search_provider = "mock"

            search_config = SearchConfig(
                provider=search_provider,
                api_key=search_api_key
            )
            web_search = MatryoshkaWebSearch(config=search_config)
            logger.info(f"Web search enabled: {search_provider}")

        return cls(
            agentic_orchestrator=agentic,
            web_search=web_search,
            enable_web_search=enable_web_search
        )

    async def research_web(
        self,
        query: str,
        max_web_results: int = 10,
        max_steps: int = 5,
        enable_citations: bool = True
    ) -> WebResearchResult:
        """
        Research mode with live web search.

        Performs multi-step research enhanced with web search:
        1. Search web with Matryoshka filtering
        2. Convert results to memory shards
        3. Add shards to knowledge base
        4. Synthesize findings with citations
        5. Return complete result with provenance

        Args:
            query: Research query
            max_web_results: Number of web results to retrieve
            max_steps: Maximum research steps
            enable_citations: Add inline citations to response

        Returns:
            WebResearchResult with cited response and sources
        """
        start_time = datetime.now()

        if not self.enable_web_search:
            raise RuntimeError(
                "Web search not enabled. Create with enable_web_search=True"
            )

        self.logger.info(f"[WEB RESEARCH] Query: {query}")

        # ====================================================================
        # Step 1: Search web with Matryoshka filtering
        # ====================================================================
        web_search_start = datetime.now()

        search_results, memory_shards = await self.web_search.search_to_shards(
            query=query,
            max_results=max_web_results
        )

        web_search_time = (datetime.now() - web_search_start).total_seconds() * 1000

        self.logger.info(
            f"[WEB RESEARCH] Found {len(search_results)} results, "
            f"created {len(memory_shards)} memory shards in {web_search_time:.1f}ms"
        )

        # ====================================================================
        # Step 2: Add shards to knowledge base
        # ====================================================================
        # Add to orchestrator's shard collection for immediate retrieval
        self.agentic.learning_engine.orchestrator.shards.extend(memory_shards)

        self.logger.info(f"[WEB RESEARCH] Added {len(memory_shards)} shards to memory")

        # ====================================================================
        # Step 3: Create synthesis query with sources
        # ====================================================================
        synthesis_query = self._create_web_synthesis_query(query, search_results)

        # ====================================================================
        # Step 4: Use agentic reasoning to synthesize
        # ====================================================================
        query_obj = Query(text=synthesis_query)
        agentic_result = await self.agentic.reason(
            query=query_obj,
            mode=ReasoningMode.DIRECT,  # Use DIRECT since we've already done research
            max_steps=max_steps
        )

        # ====================================================================
        # Step 5: Add citations to response
        # ====================================================================
        cited_response = agentic_result.spacetime.response

        if enable_citations:
            cited_response, citations = await self.web_search.search_with_citations(
                query=query,
                response=agentic_result.spacetime.response,
                max_results=max_web_results
            )

            self.logger.info(f"[WEB RESEARCH] Added {len(citations)} citations")

        # ====================================================================
        # Step 6: Build result
        # ====================================================================
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        result = WebResearchResult(
            spacetime=agentic_result.spacetime,
            intent=AgenticIntent(
                intent_id=f"web_research_{int(start_time.timestamp())}",
                original_query=query,
                goal=f"Research: {query}"
            ),
            reasoning_mode=ReasoningMode.RESEARCH,
            search_results=search_results,
            memory_shards=memory_shards,
            cited_response=cited_response,
            web_search_time_ms=web_search_time,
            total_queries=len(search_results) + 1,
            total_duration_ms=duration_ms,
            steps_taken=[
                {
                    "type": "web_search",
                    "query": query,
                    "results_found": len(search_results),
                    "time_ms": web_search_time
                },
                {
                    "type": "synthesis",
                    "confidence": agentic_result.spacetime.confidence,
                    "sources": len(search_results)
                }
            ]
        )

        self.logger.info(
            f"[WEB RESEARCH] Complete in {duration_ms:.1f}ms "
            f"({web_search_time:.1f}ms search, "
            f"{duration_ms - web_search_time:.1f}ms synthesis)"
        )

        return result

    async def reason(
        self,
        query: Query,
        mode: ReasoningMode = ReasoningMode.DIRECT,
        **kwargs
    ) -> AgenticResult:
        """
        Standard agentic reasoning (delegates to agentic orchestrator).

        For web-enhanced research, use research_web() instead.

        Args:
            query: Query object
            mode: Reasoning mode
            **kwargs: Additional arguments

        Returns:
            AgenticResult
        """
        return await self.agentic.reason(query, mode=mode, **kwargs)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_web_synthesis_query(
        self,
        original_query: str,
        search_results: List[WebSearchResult]
    ) -> str:
        """
        Create synthesis query from web search results.

        Includes source citations for better context.
        """
        synthesis = f"""Based on the following web sources, provide a comprehensive answer to: {original_query}

Web Sources:
"""

        for i, result in enumerate(search_results, 1):
            synthesis += f"\n[{i}] {result.title}\n"
            synthesis += f"URL: {result.url}\n"
            synthesis += f"Content: {result.snippet}\n"

            # Include full content if available (truncated)
            if result.full_content and len(result.full_content) > len(result.snippet):
                content_preview = result.full_content[:500]
                synthesis += f"Full text: {content_preview}...\n"

            synthesis += "\n"

        synthesis += f"\nPlease synthesize these sources into a clear, comprehensive answer."

        return synthesis

    async def close(self):
        """Cleanup resources."""
        await self.agentic.close()
        self.logger.info("WebResearchOrchestrator closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
