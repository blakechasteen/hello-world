"""
Search Protocol - Abstract Interfaces
======================================
Protocol-based design for search providers.

Philosophy:
"Depend on abstractions, not concretions."

This module defines the contracts that all search providers must implement,
enabling swapping providers without changing orchestrator code.
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SearchConfig:
    """Configuration for web search."""

    # Provider settings
    provider: str = "serpapi"  # "serpapi", "bing", "tavily", "brave"
    api_key: Optional[str] = None

    # Matryoshka settings
    stage1_size: int = 96   # Broad search dimension
    stage2_size: int = 192  # Refinement dimension
    stage3_size: int = 384  # Final ranking dimension

    # Retrieval settings
    stage1_candidates: int = 100  # Broad search limit
    stage2_candidates: int = 20   # Refinement limit
    final_results: int = 10       # Final results

    # Scraping settings
    scrape_full_content: bool = True
    timeout_seconds: int = 10
    max_content_length: int = 100000

    # Caching settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_size: int = 1000

    # Performance settings
    parallel_scraping: bool = True
    max_parallel: int = 5


# ============================================================================
# Data Models
# ============================================================================

class SearchResultType(Enum):
    """Type of search result."""
    ORGANIC = "organic"      # Regular web result
    FEATURED = "featured"    # Featured snippet
    NEWS = "news"           # News article
    VIDEO = "video"         # Video result
    IMAGE = "image"         # Image result
    ACADEMIC = "academic"   # Academic paper


@dataclass
class WebSearchResult:
    """
    Single web search result with multi-scale scores.

    Includes similarity scores at each Matryoshka stage for analysis.
    """
    # Core fields
    url: str
    title: str
    snippet: str

    # Content
    full_content: str = ""
    html_content: str = ""

    # Metadata
    result_type: SearchResultType = SearchResultType.ORGANIC
    domain: str = ""
    published_date: Optional[str] = None
    author: Optional[str] = None

    # Multi-scale similarity scores
    score_96d: float = 0.0   # Coarse similarity (stage 1)
    score_192d: float = 0.0  # Medium similarity (stage 2)
    score_384d: float = 0.0  # Fine similarity (stage 3)
    final_score: float = 0.0  # Combined score

    # Ranking
    original_rank: int = 0    # Rank from search API
    final_rank: int = 0       # Rank after Matryoshka re-ranking

    # Provenance
    search_query: str = ""
    timestamp: str = ""

    def to_citation(self, index: int) -> str:
        """Format as citation [1], [2], etc."""
        return f"[{index}] {self.title} - {self.url}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "full_content": self.full_content[:500] + "..." if len(self.full_content) > 500 else self.full_content,
            "domain": self.domain,
            "result_type": self.result_type.value,
            "scores": {
                "96d": self.score_96d,
                "192d": self.score_192d,
                "384d": self.score_384d,
                "final": self.final_score
            },
            "ranks": {
                "original": self.original_rank,
                "final": self.final_rank
            }
        }


@dataclass
class RawSearchResult:
    """
    Raw result from search API before Matryoshka processing.

    Minimal structure - just what we get from the API.
    """
    url: str
    title: str
    snippet: str
    rank: int = 0
    result_type: SearchResultType = SearchResultType.ORGANIC
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Search Provider Protocol
# ============================================================================

class SearchProvider(Protocol):
    """
    Protocol for search API providers.

    All search providers (SerpAPI, Bing, Tavily, etc.) must implement this interface.
    This enables the MatryoshkaWebSearch to work with any provider.

    Usage:
        class MySearchProvider:
            async def search(self, query: str, limit: int) -> List[RawSearchResult]:
                # Call API
                # Parse results
                # Return list of RawSearchResult
                pass
    """

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs
    ) -> List[RawSearchResult]:
        """
        Execute search query and return raw results.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Provider-specific parameters

        Returns:
            List of RawSearchResult objects
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if provider is available.

        Returns:
            True if provider is working, False otherwise
        """
        ...

    def get_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider name (e.g., "serpapi", "bing")
        """
        ...


# ============================================================================
# Content Scraper Protocol
# ============================================================================

class ContentScraper(Protocol):
    """
    Protocol for web content scrapers.

    Separate from search providers - handles fetching full page content.
    """

    async def scrape(
        self,
        url: str,
        timeout: int = 10
    ) -> tuple[str, str]:
        """
        Scrape full content from URL.

        Args:
            url: URL to scrape
            timeout: Timeout in seconds

        Returns:
            Tuple of (text_content, html_content)
        """
        ...

    async def scrape_batch(
        self,
        urls: List[str],
        max_parallel: int = 5
    ) -> Dict[str, tuple[str, str]]:
        """
        Scrape multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape
            max_parallel: Maximum parallel requests

        Returns:
            Dict mapping url -> (text, html)
        """
        ...
