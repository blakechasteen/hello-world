"""
HoloLoom Search Module
======================
Matryoshka-powered web search + Multi-agent search orchestration.

Philosophy:
"Fast filtering, slow ranking."

Components:
1. **MatryoshkaWebSearch**: 3-stage adaptive retrieval (10-50× speedup)
2. **SearchOrchestrator**: Multi-agent intelligent search
3. **Citation System**: Perplexity-style inline citations
4. **Search Cache**: TTL-based caching with LRU eviction

Quick Start:
    # Matryoshka web search
    from hololoom.search import MatryoshkaWebSearch, SearchConfig

    config = SearchConfig(provider="serpapi", api_key="...")
    search = MatryoshkaWebSearch(config=config)
    results = await search.search("What is Thompson Sampling?")

    # Multi-agent orchestration
    from hololoom.search import SearchOrchestrator

    orchestrator = SearchOrchestrator()
    result = await orchestrator.search("Compare bread vs brewing ROI")
"""

# Matryoshka web search (new!)
# Multi-agent search suite (existing)
from .agentic_search_suite import (
    AnalyticalAgent,
    ExploratoryAgent,
    FactualAgent,
    MultiHopAgent,
    SearchAgent,
    SearchOrchestrator,
    SearchQuery,
    SearchStrategy,
)
from .agentic_search_suite import SearchResult as AgenticSearchResult
from .cache import SearchCache
from .citation import Citation, CitationFormatter, CitationStyle
from .matryoshka_search import MatryoshkaWebSearch
from .protocol import SearchConfig, SearchProvider, WebSearchResult
from .providers import create_provider

__all__ = [
    # Matryoshka web search
    "SearchProvider",
    "WebSearchResult",
    "SearchConfig",
    "MatryoshkaWebSearch",
    "CitationFormatter",
    "CitationStyle",
    "Citation",
    "SearchCache",
    "create_provider",
    # Multi-agent search
    "SearchOrchestrator",
    "SearchAgent",
    "FactualAgent",
    "AnalyticalAgent",
    "MultiHopAgent",
    "ExploratoryAgent",
    "SearchStrategy",
    "SearchQuery",
    "AgenticSearchResult",
]
