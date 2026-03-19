"""
HoloLoom Agentic Search Suite
==============================
Comprehensive multi-agent search orchestrator with specialized intelligence.

Architecture:
- SearchOrchestrator: Routes queries to specialized agents
- FactualAgent: Direct fact retrieval
- AnalyticalAgent: Multi-document synthesis and comparison
- MultiHopAgent: Chain-of-thought reasoning across documents
- ExploratoryAgent: Discovery and brainstorming
- Query decomposition: Complex queries → sub-queries → parallel execution
- Result synthesis: Combine multi-agent results into coherent response

This is "agents all the way down" - each search type gets its own intelligent agent.

Usage:
    from hololoom.search import SearchOrchestrator

    orchestrator = SearchOrchestrator()
    result = await orchestrator.search("What's the ROI comparison between bread and brewing?")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class SearchStrategy(Enum):
    """Search strategies mapped to agent types."""
    FACTUAL = "factual"           # Single-document, direct facts
    ANALYTICAL = "analytical"     # Multi-document synthesis
    MULTI_HOP = "multi_hop"      # Chain-of-thought reasoning
    EXPLORATORY = "exploratory"   # Discovery, brainstorming
    HYBRID = "hybrid"            # Combine multiple strategies


@dataclass
class SearchQuery:
    """Structured search query."""
    text: str
    strategy: SearchStrategy | None = None
    max_documents: int = 10
    require_sources: bool = True
    time_limit_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Unified search result."""
    query: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    strategy_used: SearchStrategy
    agent_reasoning: list[str]
    elapsed_ms: float
    sub_queries: list[str] | None = None


# ============================================================================
# Base Agent Protocol
# ============================================================================

class SearchAgent:
    """Base class for specialized search agents."""

    def __init__(self, name: str):
        self.name = name
        self.query_count = 0
        self.total_time_ms = 0.0

    async def search(self, query: SearchQuery) -> SearchResult:
        """Execute search with this agent's strategy."""
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        avg_time = self.total_time_ms / self.query_count if self.query_count > 0 else 0
        return {
            'name': self.name,
            'queries': self.query_count,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': avg_time
        }


# ============================================================================
# Specialized Agents
# ============================================================================

class FactualAgent(SearchAgent):
    """
    Factual Agent: Direct fact retrieval.

    Best for:
    - "What is X?"
    - "When did X happen?"
    - "Who is X?"

    Strategy:
    - Single RAG query
    - Top result with highest confidence
    - Fast (50-100ms)
    """

    def __init__(self):
        super().__init__("FactualAgent")

    async def search(self, query: SearchQuery) -> SearchResult:
        start = datetime.now()
        self.query_count += 1

        logger.info(f"[FactualAgent] Processing: {query.text}")

        # Simple RAG query for direct facts
        # In production, this would call your RAG system
        reasoning = [
            "Identified as factual query",
            "Single RAG retrieval with semantic search",
            "Selected top result by confidence"
        ]

        # Simulate retrieval
        sources = [
            {
                'id': 'doc_001',
                'text': f"Found relevant information about: {query.text}",
                'score': 0.92,
                'metadata': {'type': 'factual'}
            }
        ]

        answer = f"Based on the retrieved documents: {sources[0]['text']}"

        elapsed = (datetime.now() - start).total_seconds() * 1000
        self.total_time_ms += elapsed

        return SearchResult(
            query=query.text,
            answer=answer,
            sources=sources,
            confidence=0.92,
            strategy_used=SearchStrategy.FACTUAL,
            agent_reasoning=reasoning,
            elapsed_ms=elapsed
        )


class AnalyticalAgent(SearchAgent):
    """
    Analytical Agent: Multi-document synthesis and comparison.

    Best for:
    - "Compare X and Y"
    - "What are the tradeoffs?"
    - "Why is X better than Y?"

    Strategy:
    - Multiple RAG queries (one per entity)
    - Cross-document analysis
    - Synthesis of comparisons
    - Moderate speed (200-400ms)
    """

    def __init__(self):
        super().__init__("AnalyticalAgent")

    async def search(self, query: SearchQuery) -> SearchResult:
        start = datetime.now()
        self.query_count += 1

        logger.info(f"[AnalyticalAgent] Processing: {query.text}")

        # Extract entities for comparison
        entities = self._extract_comparison_entities(query.text)

        reasoning = [
            f"Identified as analytical query with {len(entities)} entities",
            f"Entities: {', '.join(entities)}",
            "Retrieving documents for each entity in parallel",
            "Cross-document comparison and synthesis"
        ]

        # Simulate multi-document retrieval
        sources = []
        for i, entity in enumerate(entities):
            sources.append({
                'id': f'doc_{i:03d}',
                'text': f"Information about {entity}: [Details here]",
                'score': 0.85 - (i * 0.05),
                'entity': entity
            })

        # Synthesize comparison
        answer = self._synthesize_comparison(entities, sources)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        self.total_time_ms += elapsed

        return SearchResult(
            query=query.text,
            answer=answer,
            sources=sources,
            confidence=0.87,
            strategy_used=SearchStrategy.ANALYTICAL,
            agent_reasoning=reasoning,
            elapsed_ms=elapsed
        )

    def _extract_comparison_entities(self, query: str) -> list[str]:
        """Extract entities being compared."""
        # Simple keyword extraction (in production, use NER)
        keywords = ['bread', 'brewing', 'honey', 'seeds', 'lettuce']
        return [kw for kw in keywords if kw in query.lower()]

    def _synthesize_comparison(self, entities: list[str], sources: list[dict]) -> str:
        """Synthesize multi-document comparison."""
        if len(entities) >= 2:
            return f"""Comparison of {' vs '.join(entities)}:

Based on analysis of {len(sources)} documents:

{entities[0].title()}:
- Advantages: [From sources]
- Disadvantages: [From sources]
- Metrics: [From sources]

{entities[1].title()}:
- Advantages: [From sources]
- Disadvantages: [From sources]
- Metrics: [From sources]

Recommendation: [Based on analysis]"""
        else:
            return f"Analysis of {entities[0] if entities else 'query'}: [Results]"


class MultiHopAgent(SearchAgent):
    """
    Multi-Hop Agent: Chain-of-thought reasoning across documents.

    Best for:
    - "If I start X, what happens to Y?"
    - "What's the path from A to B?"
    - "Explain the relationship between X and Y"

    Strategy:
    - Query decomposition into steps
    - Sequential retrieval (each step informs next)
    - Chain reasoning together
    - Slower (400-800ms) but thorough
    """

    def __init__(self):
        super().__init__("MultiHopAgent")

    async def search(self, query: SearchQuery) -> SearchResult:
        start = datetime.now()
        self.query_count += 1

        logger.info(f"[MultiHopAgent] Processing: {query.text}")

        # Decompose into reasoning chain
        steps = self._decompose_query(query.text)

        reasoning = [
            "Identified as multi-hop reasoning query",
            f"Decomposed into {len(steps)} reasoning steps"
        ]

        # Execute chain
        sources = []
        chain_results = []

        for i, step in enumerate(steps):
            reasoning.append(f"Step {i+1}: {step}")

            # Simulate retrieval for this step
            step_sources = [
                {
                    'id': f'doc_step{i}_{j:03d}',
                    'text': f"Information for step '{step}': [Details]",
                    'score': 0.88 - (j * 0.02),
                    'step': i+1
                }
                for j in range(2)
            ]
            sources.extend(step_sources)

            # Simulate reasoning
            step_answer = f"From step {i+1}: [Inference based on retrieved docs]"
            chain_results.append(step_answer)
            reasoning.append(f"  → {step_answer}")

        # Final synthesis
        answer = self._synthesize_chain(steps, chain_results)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        self.total_time_ms += elapsed

        return SearchResult(
            query=query.text,
            answer=answer,
            sources=sources,
            confidence=0.84,
            strategy_used=SearchStrategy.MULTI_HOP,
            agent_reasoning=reasoning,
            elapsed_ms=elapsed,
            sub_queries=steps
        )

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose complex query into reasoning steps."""
        # Simple decomposition (in production, use LLM)
        if "if" in query.lower() and "then" in query.lower():
            return [
                "Identify initial condition",
                "Find direct effects",
                "Trace secondary effects",
                "Synthesize outcome"
            ]
        else:
            return [
                "Gather background information",
                "Identify key relationships",
                "Connect concepts",
                "Formulate answer"
            ]

    def _synthesize_chain(self, steps: list[str], results: list[str]) -> str:
        """Synthesize chain-of-thought reasoning."""
        synthesis = "Chain-of-Thought Analysis:\n\n"
        for i, (step, result) in enumerate(zip(steps, results), 1):
            synthesis += f"{i}. {step}\n   {result}\n\n"
        synthesis += "Conclusion: [Based on complete reasoning chain]"
        return synthesis


class ExploratoryAgent(SearchAgent):
    """
    Exploratory Agent: Discovery and brainstorming.

    Best for:
    - "What else should I consider?"
    - "Show me related ideas"
    - "What are alternatives?"

    Strategy:
    - Diverse retrieval (maximize variety)
    - Cluster results by theme
    - Surface unexpected connections
    - Moderate speed (200-400ms)
    """

    def __init__(self):
        super().__init__("ExploratoryAgent")

    async def search(self, query: SearchQuery) -> SearchResult:
        start = datetime.now()
        self.query_count += 1

        logger.info(f"[ExploratoryAgent] Processing: {query.text}")

        reasoning = [
            "Identified as exploratory query",
            "Retrieving diverse results (maximize variety)",
            "Clustering by theme",
            "Surfacing unexpected connections"
        ]

        # Simulate diverse retrieval
        themes = ["Cost", "Time", "Risk", "Opportunity", "Innovation"]
        sources = []

        for i, theme in enumerate(themes):
            sources.append({
                'id': f'doc_theme_{i:03d}',
                'text': f"{theme} perspective: [Insights]",
                'score': 0.75,  # Lower individual scores, but high diversity
                'theme': theme
            })

        # Organize by theme
        answer = self._organize_by_theme(themes, sources)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        self.total_time_ms += elapsed

        return SearchResult(
            query=query.text,
            answer=answer,
            sources=sources,
            confidence=0.78,  # Lower confidence (exploratory)
            strategy_used=SearchStrategy.EXPLORATORY,
            agent_reasoning=reasoning,
            elapsed_ms=elapsed
        )

    def _organize_by_theme(self, themes: list[str], sources: list[dict]) -> str:
        """Organize diverse results by theme."""
        organized = "Exploratory Analysis:\n\n"
        for theme in themes:
            organized += f"## {theme}\n"
            theme_sources = [s for s in sources if s.get('theme') == theme]
            for source in theme_sources:
                organized += f"- {source['text']}\n"
            organized += "\n"
        organized += "**Unexpected Connections**: [Novel insights from cross-theme analysis]"
        return organized


# ============================================================================
# Search Orchestrator
# ============================================================================

class SearchOrchestrator:
    """
    Main orchestrator: Routes queries to appropriate agents.

    Capabilities:
    - Automatic strategy selection
    - Query decomposition for complex queries
    - Parallel agent execution
    - Result synthesis and ranking
    - Performance tracking
    """

    def __init__(self):
        self.agents = {
            SearchStrategy.FACTUAL: FactualAgent(),
            SearchStrategy.ANALYTICAL: AnalyticalAgent(),
            SearchStrategy.MULTI_HOP: MultiHopAgent(),
            SearchStrategy.EXPLORATORY: ExploratoryAgent()
        }

        self.query_history = []
        self.total_queries = 0

    async def search(
        self,
        query: str,
        strategy: SearchStrategy | None = None,
        **kwargs
    ) -> SearchResult:
        """
        Main search endpoint.

        Args:
            query: Search query
            strategy: Optional strategy override (auto-detected if None)
            **kwargs: Additional search parameters

        Returns:
            SearchResult with answer, sources, and reasoning
        """
        start = datetime.now()
        self.total_queries += 1

        # Create structured query
        search_query = SearchQuery(
            text=query,
            strategy=strategy,
            **kwargs
        )

        # Auto-detect strategy if not provided
        if search_query.strategy is None:
            search_query.strategy = self._route_query(query)
            logger.info(f"Auto-routed to: {search_query.strategy.value}")

        # Route to appropriate agent
        agent = self.agents[search_query.strategy]
        result = await agent.search(search_query)

        # Track history
        self.query_history.append({
            'query': query,
            'strategy': search_query.strategy.value,
            'confidence': result.confidence,
            'elapsed_ms': result.elapsed_ms
        })

        total_elapsed = (datetime.now() - start).total_seconds() * 1000
        logger.info(f"Search completed in {total_elapsed:.1f}ms (agent: {result.elapsed_ms:.1f}ms)")

        return result

    async def parallel_search(
        self,
        queries: list[str],
        strategies: list[SearchStrategy] | None = None
    ) -> list[SearchResult]:
        """
        Execute multiple searches in parallel.

        Args:
            queries: List of queries
            strategies: Optional strategies (one per query)

        Returns:
            List of SearchResults
        """
        if strategies is None:
            strategies = [None] * len(queries)

        tasks = [
            self.search(q, strategy=s)
            for q, s in zip(queries, strategies)
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def decompose_and_search(self, complex_query: str) -> SearchResult:
        """
        Decompose complex query into sub-queries and search in parallel.

        Args:
            complex_query: Complex multi-part query

        Returns:
            Synthesized SearchResult
        """
        start = datetime.now()

        # Decompose query
        sub_queries = self._decompose_complex_query(complex_query)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")

        # Search in parallel
        sub_results = await self.parallel_search(sub_queries)

        # Synthesize results
        synthesized = self._synthesize_results(complex_query, sub_results)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        synthesized.elapsed_ms = elapsed

        return synthesized

    def _route_query(self, query: str) -> SearchStrategy:
        """
        Auto-route query to best strategy.

        Uses simple rule-based routing. In production, use a classifier.
        """
        query_lower = query.lower()

        # Multi-hop indicators
        if any(word in query_lower for word in ['if', 'then', 'path', 'relationship', 'chain']):
            return SearchStrategy.MULTI_HOP

        # Analytical indicators
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'tradeoff', 'better']):
            return SearchStrategy.ANALYTICAL

        # Exploratory indicators
        if any(word in query_lower for word in ['alternatives', 'ideas', 'brainstorm', 'what else', 'other']):
            return SearchStrategy.EXPLORATORY

        # Default to factual
        return SearchStrategy.FACTUAL

    def _decompose_complex_query(self, query: str) -> list[str]:
        """
        Decompose complex query into sub-queries.

        Example:
          "Compare bread and brewing, then tell me which has better ROI"
          → ["What is bread baking ROI?", "What is brewing ROI?", "Compare bread vs brewing"]
        """
        # Simple decomposition (in production, use LLM)
        if "compare" in query.lower() and "and" in query.lower():
            # Extract entities
            parts = query.lower().split("and")
            if len(parts) >= 2:
                entity1 = parts[0].replace("compare", "").strip()
                entity2 = parts[1].split(",")[0].strip()
                return [
                    f"What is {entity1}?",
                    f"What is {entity2}?",
                    f"Compare {entity1} and {entity2}"
                ]

        # Fallback: treat as single query
        return [query]

    def _synthesize_results(
        self,
        original_query: str,
        sub_results: list[SearchResult]
    ) -> SearchResult:
        """Synthesize multiple sub-results into single answer."""

        # Combine sources
        all_sources = []
        for result in sub_results:
            all_sources.extend(result.sources)

        # Combine reasoning
        all_reasoning = [f"Decomposed query into {len(sub_results)} parts:"]
        for i, result in enumerate(sub_results, 1):
            all_reasoning.append(f"Part {i}: {result.query}")
            all_reasoning.extend([f"  {r}" for r in result.agent_reasoning])

        # Synthesize answer
        answer = "Synthesized Answer:\n\n"
        for i, result in enumerate(sub_results, 1):
            answer += f"Part {i}: {result.query}\n{result.answer}\n\n"
        answer += "Overall Conclusion: [Synthesis of all parts]"

        # Average confidence
        avg_confidence = sum(r.confidence for r in sub_results) / len(sub_results)

        return SearchResult(
            query=original_query,
            answer=answer,
            sources=all_sources,
            confidence=avg_confidence,
            strategy_used=SearchStrategy.HYBRID,
            agent_reasoning=all_reasoning,
            elapsed_ms=sum(r.elapsed_ms for r in sub_results),
            sub_queries=[r.query for r in sub_results]
        )

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'total_queries': self.total_queries,
            'agents': {
                name: agent.get_stats()
                for name, agent in self.agents.items()
            },
            'recent_queries': self.query_history[-10:]
        }


# ============================================================================
# Example Usage
# ============================================================================

async def demo():
    """Demo the agentic search suite."""
    orchestrator = SearchOrchestrator()

    print("=" * 70)
    print("Agentic Search Suite Demo")
    print("=" * 70)
    print()

    # Example queries
    queries = [
        ("What is the ROI on bread baking?", SearchStrategy.FACTUAL),
        ("Compare bread baking and micro brewing", SearchStrategy.ANALYTICAL),
        ("If I start bread baking, what happens to my time budget?", SearchStrategy.MULTI_HOP),
        ("What alternatives should I consider?", SearchStrategy.EXPLORATORY)
    ]

    for query, expected_strategy in queries:
        print(f"Query: {query}")
        print(f"Expected: {expected_strategy.value}")
        print("-" * 70)

        result = await orchestrator.search(query)

        print(f"Strategy Used: {result.strategy_used.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Time: {result.elapsed_ms:.1f}ms")
        print(f"Sources: {len(result.sources)}")
        print()
        print("Reasoning:")
        for step in result.agent_reasoning:
            print(f"  • {step}")
        print()
        print(f"Answer: {result.answer[:200]}...")
        print()
        print("=" * 70)
        print()

    # Complex query decomposition
    print("Complex Query Demo")
    print("=" * 70)
    complex_query = "Compare bread and brewing, then recommend which one to start with"
    result = await orchestrator.decompose_and_search(complex_query)

    print(f"Query: {complex_query}")
    print(f"Sub-queries: {result.sub_queries}")
    print(f"Time: {result.elapsed_ms:.1f}ms")
    print()

    # Stats
    print("Orchestrator Stats")
    print("=" * 70)
    stats = orchestrator.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    for strategy, agent_stats in stats['agents'].items():
        print(f"\n{strategy.value.title()} Agent:")
        print(f"  Queries: {agent_stats['queries']}")
        print(f"  Avg Time: {agent_stats['avg_time_ms']:.1f}ms")


if __name__ == "__main__":
    asyncio.run(demo())
