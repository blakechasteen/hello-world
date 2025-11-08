"""
Web Crawler + Matryoshka Search Integration
===========================================

Combines the best of both worlds:
1. MatryoshkaWebSearch - Fast 3-stage search for finding relevant URLs
2. RecursiveCrawler - Deep exploration of important pages
3. WebsiteSpinner - Content extraction and MemoryShard creation

Architecture:
    Query --> MatryoshkaWebSearch (find seed URLs)
          --> RecursiveCrawler (explore related pages)
          --> WebsiteSpinner (extract content + images)
          --> MemoryShards with full provenance

Use Cases:
- Research: "Find and explore papers on Thompson Sampling"
- Learning: "Gather comprehensive info on beekeeping"
- Investigation: "Deep dive into company X's technology stack"
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinning_wheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.spinning_wheel.recursive_crawler import (
    RecursiveCrawler,
    CrawlConfig,
    LinkInfo
)

from .matryoshka_search import MatryoshkaWebSearch
from .protocol import SearchConfig, WebSearchResult
from .citation import CitationFormatter, CitationStyle


logger = logging.getLogger(__name__)


@dataclass
class WebCrawlerSearchConfig:
    """Configuration for integrated web crawler + search."""

    # Search configuration
    search_provider: str = "mock"  # or "serpapi", "google", etc.
    search_api_key: Optional[str] = None
    max_search_results: int = 10   # How many search results to use as seeds

    # Crawl configuration
    enable_recursive_crawl: bool = True  # Whether to crawl related pages
    max_crawl_depth: int = 2            # How deep to crawl from each seed
    max_pages_per_seed: int = 5         # Max pages to crawl from each seed URL
    max_total_pages: int = 50           # Total page limit across all seeds
    crawl_same_domain_only: bool = False # Stay on seed domain

    # Content extraction
    extract_images: bool = True          # Extract meaningful images
    max_images_per_page: int = 5        # Limit images for crawled pages
    min_content_length: int = 200       # Skip short pages

    # Matryoshka importance thresholds (by depth)
    # Higher thresholds = only follow very relevant links
    importance_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,   # Seed URLs (always crawl)
        1: 0.65,  # Direct links (medium-high importance)
        2: 0.8,   # Second-level links (high importance)
    })

    # Citation formatting
    enable_citations: bool = True
    citation_style: CitationStyle = CitationStyle.INLINE_NUMERIC


class WebCrawlerSearch:
    """
    Integrated web crawler + Matryoshka search system.

    Pipeline:
    1. MatryoshkaWebSearch finds relevant seed URLs (3-stage filtering)
    2. RecursiveCrawler explores related pages from each seed
    3. WebsiteSpinner extracts content + images
    4. Results formatted with citations

    Example:
        config = WebCrawlerSearchConfig(
            search_provider="mock",
            enable_recursive_crawl=True,
            max_crawl_depth=2,
            max_pages_per_seed=5
        )

        crawler = WebCrawlerSearch(config)

        result = await crawler.search_and_crawl(
            query="What is Thompson Sampling?",
            seed_topic="Thompson Sampling Bayesian bandits"
        )

        print(f"Found {len(result.search_results)} search results")
        print(f"Crawled {len(result.crawled_pages)} pages")
        print(f"Created {len(result.shards)} memory shards")
        print(f"Citations: {result.cited_response}")
    """

    def __init__(self, config: Optional[WebCrawlerSearchConfig] = None):
        self.config = config or WebCrawlerSearchConfig()

        # Initialize Matryoshka search
        search_config = SearchConfig(
            provider=self.config.search_provider,
            api_key=self.config.search_api_key,
            final_results=self.config.max_search_results
        )
        self.search = MatryoshkaWebSearch(config=search_config)

        # Initialize citation formatter
        if self.config.enable_citations:
            self.citation_formatter = CitationFormatter(
                style=self.config.citation_style,
                auto_cite=True
            )
        else:
            self.citation_formatter = None

        # Statistics
        self.stats = {
            'total_searches': 0,
            'total_crawls': 0,
            'total_pages_crawled': 0,
            'total_shards_created': 0,
            'search_time_ms': 0,
            'crawl_time_ms': 0,
            'extraction_time_ms': 0
        }

    async def search_and_crawl(
        self,
        query: str,
        seed_topic: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> 'SearchCrawlResult':
        """
        Execute full search + crawl pipeline.

        Args:
            query: Search query
            seed_topic: Topic for recursive crawler (defaults to query)
            max_results: Override max_search_results from config

        Returns:
            SearchCrawlResult with search results, crawled pages, shards, and citations
        """
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("WEB CRAWLER + MATRYOSHKA SEARCH")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info(f"Topic: {seed_topic or query}")
        logger.info(f"Recursive crawl: {self.config.enable_recursive_crawl}")
        logger.info("")

        # Stage 1: Matryoshka Web Search (find seed URLs)
        logger.info("Stage 1: Matryoshka Web Search (3-stage filtering)")
        search_start = datetime.now()

        max_results = max_results or self.config.max_search_results
        search_results = await self.search.search(query, max_results=max_results)

        search_time = (datetime.now() - search_start).total_seconds() * 1000
        self.stats['search_time_ms'] += search_time
        self.stats['total_searches'] += 1

        logger.info(f"  Found {len(search_results)} results in {search_time:.1f}ms")
        logger.info("")

        # If recursive crawl disabled, just convert search results to shards
        if not self.config.enable_recursive_crawl:
            logger.info("Recursive crawl disabled - converting search results only")

            extract_start = datetime.now()
            all_shards = []

            for i, sr in enumerate(search_results, 1):
                logger.info(f"  [{i}/{len(search_results)}] Extracting: {sr.url}")
                shards = await self._extract_content(sr.url, sr.title, sr.snippet)
                all_shards.extend(shards)

            extract_time = (datetime.now() - extract_start).total_seconds() * 1000
            self.stats['extraction_time_ms'] += extract_time

            total_time = (datetime.now() - start_time).total_seconds() * 1000

            return SearchCrawlResult(
                query=query,
                search_results=search_results,
                crawled_pages=[],
                shards=all_shards,
                cited_response=self._format_citations(query, search_results, all_shards),
                total_duration_ms=total_time,
                search_time_ms=search_time,
                crawl_time_ms=0,
                extraction_time_ms=extract_time
            )

        # Stage 2: Recursive Crawl from each seed URL
        logger.info("Stage 2: Recursive Crawl (importance gating)")
        crawl_start = datetime.now()

        all_crawled_pages = []
        pages_budget = self.config.max_total_pages

        for i, search_result in enumerate(search_results, 1):
            if pages_budget <= 0:
                logger.info(f"  Reached total page limit ({self.config.max_total_pages})")
                break

            logger.info(f"  Seed {i}/{len(search_results)}: {search_result.url}")

            # Configure crawler for this seed
            crawl_config = CrawlConfig(
                max_depth=self.config.max_crawl_depth,
                max_pages=min(self.config.max_pages_per_seed, pages_budget),
                same_domain_only=self.config.crawl_same_domain_only,
                importance_thresholds=self.config.importance_thresholds,
                extract_images=self.config.extract_images,
                max_images_per_page=self.config.max_images_per_page,
                min_content_length=self.config.min_content_length,
                rate_limit_seconds=0.5  # Faster rate for crawling
            )

            crawler = RecursiveCrawler(config=crawl_config)

            # Use seed_topic or query for importance scoring
            topic = seed_topic or query

            # Crawl from this seed
            crawled_pages = await crawler.crawl(
                seed_url=search_result.url,
                seed_topic=topic
            )

            all_crawled_pages.extend(crawled_pages)
            pages_budget -= len(crawled_pages)

            logger.info(f"    Crawled {len(crawled_pages)} pages (budget remaining: {pages_budget})")

        crawl_time = (datetime.now() - crawl_start).total_seconds() * 1000
        self.stats['crawl_time_ms'] += crawl_time
        self.stats['total_crawls'] += 1
        self.stats['total_pages_crawled'] += len(all_crawled_pages)

        logger.info(f"  Total pages crawled: {len(all_crawled_pages)} in {crawl_time:.1f}ms")
        logger.info("")

        # Stage 3: Extract all shards (already done by crawler)
        logger.info("Stage 3: Shard Extraction (content + images)")
        extract_start = datetime.now()

        all_shards = []
        for page in all_crawled_pages:
            all_shards.extend(page['shards'])

        extract_time = (datetime.now() - extract_start).total_seconds() * 1000
        self.stats['extraction_time_ms'] += extract_time
        self.stats['total_shards_created'] += len(all_shards)

        logger.info(f"  Extracted {len(all_shards)} shards in {extract_time:.1f}ms")
        logger.info("")

        # Stage 4: Format citations (combine search results + crawled pages)
        logger.info("Stage 4: Citation Formatting")
        cited_response = self._format_citations(query, search_results, all_shards)

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info("=" * 80)
        logger.info("COMPLETE!")
        logger.info(f"  Search: {search_time:.1f}ms")
        logger.info(f"  Crawl: {crawl_time:.1f}ms")
        logger.info(f"  Extract: {extract_time:.1f}ms")
        logger.info(f"  Total: {total_time:.1f}ms")
        logger.info(f"  Pages: {len(all_crawled_pages)}")
        logger.info(f"  Shards: {len(all_shards)}")
        logger.info("=" * 80)

        return SearchCrawlResult(
            query=query,
            search_results=search_results,
            crawled_pages=all_crawled_pages,
            shards=all_shards,
            cited_response=cited_response,
            total_duration_ms=total_time,
            search_time_ms=search_time,
            crawl_time_ms=crawl_time,
            extraction_time_ms=extract_time
        )

    async def _extract_content(
        self,
        url: str,
        title: Optional[str] = None,
        snippet: Optional[str] = None
    ) -> List[MemoryShard]:
        """Extract content from single URL using WebsiteSpinner."""
        try:
            config = WebsiteSpinnerConfig(
                extract_images=self.config.extract_images,
                max_images=self.config.max_images_per_page,
                min_content_length=self.config.min_content_length
            )

            spinner = WebsiteSpinner(config)

            shards = await spinner.spin({
                'url': url,
                'title': title
            })

            return shards

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return []

    def _format_citations(
        self,
        query: str,
        search_results: List[WebSearchResult],
        shards: List[MemoryShard]
    ) -> str:
        """Format comprehensive response with citations."""
        if not self.citation_formatter:
            return f"Query: {query}\n\nFound {len(search_results)} results and created {len(shards)} memory shards."

        # Create comprehensive response
        response_parts = []

        # Summary
        response_parts.append(f"Research results for: {query}")
        response_parts.append(f"\nFound {len(search_results)} relevant sources across {len(shards)} content shards.")
        response_parts.append("\nKey findings:")

        # Extract key content from top shards
        for i, shard in enumerate(shards[:5], 1):
            # Get first sentence or snippet
            text = shard.text.split('.')[0] + '.' if '.' in shard.text else shard.text[:200]
            response_parts.append(f"\n{i}. {text}")

        response = '\n'.join(response_parts)

        # Add citations
        cited_response, citations = self.citation_formatter.add_citations(
            response,
            search_results
        )

        # Add bibliography
        bibliography = self.citation_formatter.format_bibliography(
            citations,
            title="Sources"
        )

        return f"{cited_response}\n\n{bibliography}"

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self.stats,
            'avg_search_time_ms': (
                self.stats['search_time_ms'] / self.stats['total_searches']
                if self.stats['total_searches'] > 0 else 0
            ),
            'avg_crawl_time_ms': (
                self.stats['crawl_time_ms'] / self.stats['total_crawls']
                if self.stats['total_crawls'] > 0 else 0
            ),
            'avg_pages_per_crawl': (
                self.stats['total_pages_crawled'] / self.stats['total_crawls']
                if self.stats['total_crawls'] > 0 else 0
            ),
            'avg_shards_per_page': (
                self.stats['total_shards_created'] / self.stats['total_pages_crawled']
                if self.stats['total_pages_crawled'] > 0 else 0
            )
        }


@dataclass
class SearchCrawlResult:
    """Result of integrated search + crawl operation."""
    query: str
    search_results: List[WebSearchResult]
    crawled_pages: List[Dict[str, Any]]
    shards: List[MemoryShard]
    cited_response: str
    total_duration_ms: float
    search_time_ms: float
    crawl_time_ms: float
    extraction_time_ms: float


# Convenience function
async def search_and_crawl_web(
    query: str,
    seed_topic: Optional[str] = None,
    enable_recursive_crawl: bool = True,
    max_search_results: int = 5,
    max_crawl_depth: int = 2,
    max_pages_per_seed: int = 3,
    search_provider: str = "mock"
) -> SearchCrawlResult:
    """
    Quick function to search and crawl the web.

    Args:
        query: Search query
        seed_topic: Topic for crawler importance scoring
        enable_recursive_crawl: Whether to crawl related pages
        max_search_results: How many search results to use as seeds
        max_crawl_depth: How deep to crawl from each seed
        max_pages_per_seed: Max pages per seed URL
        search_provider: Search provider (mock, serpapi, etc.)

    Returns:
        SearchCrawlResult with all data

    Example:
        # Simple search only
        result = await search_and_crawl_web(
            "What is Thompson Sampling?",
            enable_recursive_crawl=False
        )

        # Deep research
        result = await search_and_crawl_web(
            "beekeeping hive management",
            seed_topic="beekeeping varroa mites treatment",
            enable_recursive_crawl=True,
            max_search_results=3,
            max_crawl_depth=2,
            max_pages_per_seed=5
        )

        print(result.cited_response)
    """
    config = WebCrawlerSearchConfig(
        search_provider=search_provider,
        max_search_results=max_search_results,
        enable_recursive_crawl=enable_recursive_crawl,
        max_crawl_depth=max_crawl_depth,
        max_pages_per_seed=max_pages_per_seed
    )

    crawler = WebCrawlerSearch(config)
    return await crawler.search_and_crawl(query, seed_topic=seed_topic)
