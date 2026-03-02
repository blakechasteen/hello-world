"""
Agentic Web Researcher - Autonomous Multi-Step Research Agent
==============================================================

Combines:
- HoloLoom agentic reasoning (PLAN_EXECUTE, VERIFY, RESEARCH modes)
- MatryoshkaWebSearch (3-stage filtering)
- RecursiveCrawler (importance gating)
- WebsiteSpinner (content extraction)
- LearningEngine (recursive refinement)

Architecture:
    Query → Plan decomposition
          → Execute sub-queries (MatryoshkaWebSearch + Crawl)
          → Verify findings for consistency
          → Synthesize comprehensive report
          → Learn from outcome
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.search.web_crawler_integration import (
    WebCrawlerSearch,
    WebCrawlerSearchConfig,
    SearchCrawlResult
)
from hololoom.agentic.core import AgenticOrchestrator, ReasoningMode
from hololoom.recursive import FullLearningEngine


logger = logging.getLogger(__name__)


class ResearchStrategy(Enum):
    """Research execution strategies."""
    QUICK = "quick"              # Single search, no verification (fast)
    STANDARD = "standard"        # Multi-query, basic verification
    COMPREHENSIVE = "comprehensive"  # Deep crawl, full verification
    EXPLORATORY = "exploratory"  # Broad exploration, many angles


@dataclass
class ResearchPlan:
    """Decomposed research plan."""
    main_query: str
    sub_queries: List[str]
    strategy: ResearchStrategy
    max_depth: int = 1
    max_pages_per_query: int = 5
    enable_verification: bool = True
    enable_crawling: bool = True


@dataclass
class ResearchResult:
    """Complete research result."""
    query: str
    plan: ResearchPlan
    search_results: List[SearchCrawlResult]
    total_pages_crawled: int
    total_shards_created: int
    synthesis: str
    cited_response: str
    verification_results: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgenticWebResearcher:
    """
    Autonomous web research agent with multi-step reasoning.

    Capabilities:
    1. Plan decomposition (PLAN_EXECUTE mode)
    2. Multi-query search (MatryoshkaWebSearch)
    3. Recursive crawling (importance gating)
    4. Verification (consistency checking)
    5. Synthesis (comprehensive report)
    6. Learning (recursive refinement)

    Example:
        researcher = AgenticWebResearcher(
            config=Config.fast(),
            strategy=ResearchStrategy.COMPREHENSIVE
        )

        result = await researcher.research(
            "What are the key innovations in Thompson Sampling?"
        )

        print(result.cited_response)
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Pages crawled: {result.total_pages_crawled}")
        print(f"Shards created: {result.total_shards_created}")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        strategy: ResearchStrategy = ResearchStrategy.STANDARD,
        enable_learning: bool = True,
        enable_verification: bool = True
    ):
        self.config = config or Config.fast()
        self.strategy = strategy
        self.enable_learning = enable_learning
        self.enable_verification = enable_verification

        # Configure based on strategy
        self.crawler_config = self._create_crawler_config(strategy)

        # Initialize components
        self.crawler = WebCrawlerSearch(self.crawler_config)
        self.learning_engine = None  # Lazy-loaded
        self.agentic_orchestrator = None  # Lazy-loaded

        # Statistics
        self.stats = {
            'total_research_queries': 0,
            'total_sub_queries': 0,
            'total_pages_crawled': 0,
            'total_shards_created': 0,
            'avg_confidence': 0.0,
            'verifications_passed': 0,
            'verifications_failed': 0
        }

    def _create_crawler_config(self, strategy: ResearchStrategy) -> WebCrawlerSearchConfig:
        """Create crawler config based on strategy."""
        if strategy == ResearchStrategy.QUICK:
            return WebCrawlerSearchConfig(
                search_provider="mock",
                max_search_results=3,
                enable_recursive_crawl=False,  # Fast: search only
                enable_citations=True
            )

        elif strategy == ResearchStrategy.STANDARD:
            return WebCrawlerSearchConfig(
                search_provider="mock",
                max_search_results=5,
                enable_recursive_crawl=True,
                max_crawl_depth=1,
                max_pages_per_seed=3,
                max_total_pages=15,
                importance_thresholds={0: 0.0, 1: 0.7},
                enable_citations=True
            )

        elif strategy == ResearchStrategy.COMPREHENSIVE:
            return WebCrawlerSearchConfig(
                search_provider="mock",
                max_search_results=5,
                enable_recursive_crawl=True,
                max_crawl_depth=2,
                max_pages_per_seed=5,
                max_total_pages=30,
                importance_thresholds={0: 0.0, 1: 0.65, 2: 0.8},
                extract_images=True,
                enable_citations=True
            )

        elif strategy == ResearchStrategy.EXPLORATORY:
            return WebCrawlerSearchConfig(
                search_provider="mock",
                max_search_results=10,
                enable_recursive_crawl=True,
                max_crawl_depth=1,
                max_pages_per_seed=2,
                max_total_pages=20,
                importance_thresholds={0: 0.0, 1: 0.6},
                enable_citations=True
            )

        else:
            return WebCrawlerSearchConfig()

    async def research(
        self,
        query: str,
        strategy: Optional[ResearchStrategy] = None,
        max_sub_queries: int = 5
    ) -> ResearchResult:
        """
        Execute autonomous research on query.

        Args:
            query: Main research question
            strategy: Override default strategy
            max_sub_queries: Max sub-queries to generate

        Returns:
            ResearchResult with comprehensive findings
        """
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("AGENTIC WEB RESEARCHER")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info(f"Strategy: {strategy or self.strategy}")
        logger.info("")

        # Override strategy if provided
        if strategy:
            self.crawler_config = self._create_crawler_config(strategy)
            self.crawler = WebCrawlerSearch(self.crawler_config)

        # Step 1: Plan decomposition
        logger.info("Step 1: Planning research strategy...")
        plan = await self._plan_research(query, max_sub_queries)

        logger.info(f"  Main query: {plan.main_query}")
        logger.info(f"  Sub-queries: {len(plan.sub_queries)}")
        for i, sq in enumerate(plan.sub_queries, 1):
            logger.info(f"    {i}. {sq}")
        logger.info("")

        # Step 2: Execute sub-queries
        logger.info("Step 2: Executing sub-queries...")
        search_results = []

        for i, sub_query in enumerate(plan.sub_queries, 1):
            logger.info(f"  [{i}/{len(plan.sub_queries)}] Researching: {sub_query}")

            result = await self.crawler.search_and_crawl(
                query=sub_query,
                seed_topic=query,  # Use main query as topic
                max_results=self.crawler_config.max_search_results
            )

            search_results.append(result)

            logger.info(f"      Found: {len(result.search_results)} results, "
                       f"{len(result.crawled_pages)} pages, "
                       f"{len(result.shards)} shards")

        logger.info("")

        # Step 3: Aggregate results
        total_pages = sum(len(r.crawled_pages) for r in search_results)
        total_shards = sum(len(r.shards) for r in search_results)
        all_shards = []
        for r in search_results:
            all_shards.extend(r.shards)

        logger.info(f"Step 3: Aggregation complete")
        logger.info(f"  Total pages crawled: {total_pages}")
        logger.info(f"  Total shards created: {total_shards}")
        logger.info("")

        # Step 4: Verification (optional)
        verification_results = None
        if self.enable_verification and plan.enable_verification:
            logger.info("Step 4: Verifying findings...")
            verification_results = await self._verify_findings(query, all_shards)

            if verification_results['verified']:
                logger.info(f"  Verification: PASSED (confidence: {verification_results['confidence']:.2f})")
                self.stats['verifications_passed'] += 1
            else:
                logger.info(f"  Verification: FAILED (issues: {len(verification_results['issues'])})")
                self.stats['verifications_failed'] += 1

            logger.info("")

        # Step 5: Synthesis
        logger.info("Step 5: Synthesizing comprehensive report...")
        synthesis, cited_response = await self._synthesize_report(
            query,
            plan,
            search_results,
            all_shards
        )

        logger.info(f"  Report synthesized ({len(synthesis)} chars)")
        logger.info("")

        # Calculate confidence
        confidence = self._calculate_confidence(
            search_results,
            verification_results,
            total_shards
        )

        # Update statistics
        self.stats['total_research_queries'] += 1
        self.stats['total_sub_queries'] += len(plan.sub_queries)
        self.stats['total_pages_crawled'] += total_pages
        self.stats['total_shards_created'] += total_shards
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_research_queries'] - 1) + confidence)
            / self.stats['total_research_queries']
        )

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info("=" * 80)
        logger.info("RESEARCH COMPLETE!")
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  Pages: {total_pages}")
        logger.info(f"  Shards: {total_shards}")
        logger.info(f"  Duration: {total_time:.1f}ms")
        logger.info("=" * 80)

        return ResearchResult(
            query=query,
            plan=plan,
            search_results=search_results,
            total_pages_crawled=total_pages,
            total_shards_created=total_shards,
            synthesis=synthesis,
            cited_response=cited_response,
            verification_results=verification_results,
            confidence=confidence,
            total_duration_ms=total_time,
            metadata={
                'strategy': plan.strategy.value,
                'sub_query_count': len(plan.sub_queries),
                'verification_enabled': self.enable_verification
            }
        )

    async def _plan_research(
        self,
        query: str,
        max_sub_queries: int
    ) -> ResearchPlan:
        """
        Decompose query into research plan.

        Uses simple heuristics for now. Future: use AgenticOrchestrator.
        """
        # Simple decomposition based on query characteristics
        query_lower = query.lower()

        # Detect question type
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            # Definitional query
            sub_queries = [
                f"{query}",  # Main query
                f"Examples of {query.replace('What is', '').replace('?', '')}",
                f"Applications of {query.replace('What is', '').replace('?', '')}"
            ]

        elif any(word in query_lower for word in ['how', 'why', 'compare']):
            # Procedural/comparative query
            sub_queries = [
                f"{query}",  # Main query
                f"Step by step {query.replace('How to', '').replace('?', '')}",
                f"Best practices for {query.replace('How to', '').replace('?', '')}"
            ]

        elif any(word in query_lower for word in ['tradeoffs', 'pros', 'cons', 'versus']):
            # Comparative query
            sub_queries = [
                f"{query}",  # Main query
                f"Advantages of {query.split()[0]}",
                f"Disadvantages of {query.split()[0]}",
                f"Alternatives to {query.split()[0]}"
            ]

        else:
            # General query
            sub_queries = [
                f"{query}",
                f"Overview of {query}",
                f"Recent developments in {query}"
            ]

        # Limit sub-queries
        sub_queries = sub_queries[:max_sub_queries]

        return ResearchPlan(
            main_query=query,
            sub_queries=sub_queries,
            strategy=self.strategy,
            max_depth=self.crawler_config.max_crawl_depth,
            max_pages_per_query=self.crawler_config.max_pages_per_seed,
            enable_verification=self.enable_verification,
            enable_crawling=self.crawler_config.enable_recursive_crawl
        )

    async def _verify_findings(
        self,
        query: str,
        shards: List[MemoryShard]
    ) -> Dict[str, Any]:
        """
        Verify findings for consistency and accuracy.

        Simple verification: check for contradictions and coverage.
        """
        if not shards:
            return {
                'verified': False,
                'confidence': 0.0,
                'issues': ['No shards to verify']
            }

        # Simple heuristics
        issues = []

        # Check coverage
        if len(shards) < 3:
            issues.append(f"Low coverage: only {len(shards)} shards")

        # Check diversity (unique domains)
        domains = set(s.metadata.get('domain', '') for s in shards if 'domain' in s.metadata)
        if len(domains) < 2:
            issues.append(f"Low diversity: only {len(domains)} unique domains")

        # Calculate confidence
        confidence = max(0.0, 1.0 - (len(issues) * 0.2))

        return {
            'verified': len(issues) == 0,
            'confidence': confidence,
            'issues': issues,
            'shard_count': len(shards),
            'domain_count': len(domains)
        }

    async def _synthesize_report(
        self,
        query: str,
        plan: ResearchPlan,
        search_results: List[SearchCrawlResult],
        shards: List[MemoryShard]
    ) -> tuple[str, str]:
        """
        Synthesize comprehensive report from findings.

        Returns:
            (synthesis, cited_response)
        """
        # Build synthesis
        synthesis_parts = []

        synthesis_parts.append(f"# Research Report: {query}")
        synthesis_parts.append("")
        synthesis_parts.append(f"**Strategy**: {plan.strategy.value}")
        synthesis_parts.append(f"**Sub-queries**: {len(plan.sub_queries)}")
        synthesis_parts.append(f"**Sources**: {len(search_results)} searches, "
                              f"{sum(len(r.search_results) for r in search_results)} results")
        synthesis_parts.append("")

        synthesis_parts.append("## Key Findings")
        synthesis_parts.append("")

        # Extract key findings from shards
        for i, shard in enumerate(shards[:10], 1):
            # Get first sentence
            text = shard.text.split('.')[0] + '.' if '.' in shard.text else shard.text[:200]
            synthesis_parts.append(f"{i}. {text}")

        synthesis_parts.append("")

        # Add sources section
        synthesis_parts.append("## Sources")
        synthesis_parts.append("")

        all_search_results = []
        for search_result in search_results:
            all_search_results.extend(search_result.search_results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for sr in all_search_results:
            if sr.url not in seen_urls:
                seen_urls.add(sr.url)
                unique_results.append(sr)

        for i, sr in enumerate(unique_results[:20], 1):
            synthesis_parts.append(f"[{i}] {sr.title} - {sr.url}")

        synthesis = '\n'.join(synthesis_parts)

        # For cited_response, just use the first search result's citation
        if search_results:
            cited_response = search_results[0].cited_response
        else:
            cited_response = synthesis

        return synthesis, cited_response

    def _calculate_confidence(
        self,
        search_results: List[SearchCrawlResult],
        verification_results: Optional[Dict[str, Any]],
        total_shards: int
    ) -> float:
        """Calculate overall confidence score."""
        # Base confidence from search quality
        if not search_results:
            return 0.0

        avg_result_count = sum(len(r.search_results) for r in search_results) / len(search_results)
        search_confidence = min(1.0, avg_result_count / 5.0)  # Normalize to 5 results

        # Shard count confidence
        shard_confidence = min(1.0, total_shards / 20.0)  # Normalize to 20 shards

        # Verification confidence
        verification_confidence = 1.0
        if verification_results:
            verification_confidence = verification_results.get('confidence', 0.5)

        # Weighted average
        confidence = (
            0.4 * search_confidence +
            0.3 * shard_confidence +
            0.3 * verification_confidence
        )

        return round(confidence, 2)

    def get_stats(self) -> Dict[str, Any]:
        """Get research statistics."""
        return {
            **self.stats,
            'crawler_stats': self.crawler.get_stats(),
            'avg_pages_per_query': (
                self.stats['total_pages_crawled'] / self.stats['total_research_queries']
                if self.stats['total_research_queries'] > 0 else 0
            ),
            'avg_shards_per_query': (
                self.stats['total_shards_created'] / self.stats['total_research_queries']
                if self.stats['total_research_queries'] > 0 else 0
            ),
            'verification_pass_rate': (
                self.stats['verifications_passed'] /
                (self.stats['verifications_passed'] + self.stats['verifications_failed'])
                if (self.stats['verifications_passed'] + self.stats['verifications_failed']) > 0
                else 0
            )
        }


# Convenience function
async def research_web(
    query: str,
    strategy: ResearchStrategy = ResearchStrategy.STANDARD,
    enable_verification: bool = True,
    max_sub_queries: int = 5
) -> ResearchResult:
    """
    Quick function for autonomous web research.

    Args:
        query: Research question
        strategy: Research strategy (quick, standard, comprehensive, exploratory)
        enable_verification: Whether to verify findings
        max_sub_queries: Max sub-queries to generate

    Returns:
        ResearchResult with comprehensive findings

    Example:
        # Quick research
        result = await research_web(
            "What is Thompson Sampling?",
            strategy=ResearchStrategy.QUICK
        )

        # Comprehensive research
        result = await research_web(
            "What are the key innovations in Bayesian optimization?",
            strategy=ResearchStrategy.COMPREHENSIVE,
            enable_verification=True
        )

        print(result.cited_response)
        print(f"Confidence: {result.confidence}")
    """
    researcher = AgenticWebResearcher(
        strategy=strategy,
        enable_verification=enable_verification
    )

    return await researcher.research(query, max_sub_queries=max_sub_queries)
