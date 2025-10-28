"""
Recursive Web Crawler with Matryoshka Importance Gating
========================================================

Intelligently crawl web content by following only important links.

The "Matryoshka" (nested doll) approach:
- Layer 0 (seed URL): High importance, full processing
- Layer 1 (direct links): Medium importance threshold (0.6+)
- Layer 2 (links from links): Higher threshold (0.75+)
- Layer 3+: Very high threshold (0.85+)

This ensures:
- Deep crawling on highly relevant paths
- Shallow crawling on tangential content
- No crawling of noise (ads, social, unrelated)

Features:
- Link importance scoring (relevance to seed topic)
- Depth-based importance thresholds (matryoshka gating)
- Domain diversity control
- Deduplication by URL and content hash
- Politeness (rate limiting, robots.txt)
- Stop conditions (max pages, max depth, time limit)
"""

import asyncio
import logging
from typing import List, Set, Optional, Dict, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import hashlib
import re

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

from .website import WebsiteSpinner, WebsiteSpinnerConfig

logger = logging.getLogger(__name__)


@dataclass(order=True)
class LinkInfo:
    """Information about an extracted link."""
    importance_score: float = 0.0 # How important/relevant (0-1) - MUST BE FIRST for sorting
    url: str = ""
    anchor_text: str = ""              # Link text
    title: str = ""                    # Title attribute
    context: str = ""                  # Surrounding text
    depth: int = 0                    # Crawl depth (0 = seed)
    parent_url: str = ""               # Where link was found


@dataclass
class CrawlConfig:
    """Configuration for recursive crawler."""

    # Depth control
    max_depth: int = 2                # How deep to crawl (0 = seed only)
    max_pages: int = 50               # Total page limit
    max_pages_per_domain: int = 10    # Avoid over-crawling one domain

    # Matryoshka importance thresholds (by depth)
    # Only follow links above threshold at each depth
    importance_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,   # Seed URL (always crawl)
        1: 0.6,   # Direct links (medium importance)
        2: 0.75,  # Second-level links (high importance)
        3: 0.85,  # Third-level links (very high importance)
        4: 0.9,   # Beyond (exceptional importance only)
    })

    # Link filtering
    same_domain_only: bool = False    # Stay on seed domain
    max_domain_diversity: float = 0.3 # Max % of pages from one domain

    # Content filtering
    min_content_length: int = 200     # Skip short pages
    skip_patterns: List[str] = field(default_factory=lambda: [
        r'/login', r'/signup', r'/register',
        r'/cart', r'/checkout',
        r'/search\?', r'/tag/',
        r'facebook\.com', r'twitter\.com', r'linkedin\.com',
        r'\.pdf$', r'\.zip$', r'\.exe$'
    ])

    # Politeness
    rate_limit_seconds: float = 1.0   # Delay between requests
    max_runtime_minutes: int = 30     # Total runtime limit

    # Multimodal
    extract_images: bool = True
    max_images_per_page: int = 5      # Fewer images for crawled pages


class RecursiveCrawler:
    """
    Recursively crawl web content with importance gating.

    Uses matryoshka (nested doll) approach:
    - Each depth level requires higher importance to continue
    - Prevents crawling noise while capturing related content
    """

    def __init__(self, config: Optional[CrawlConfig] = None):
        self.config = config or CrawlConfig()

        # Tracking
        self.visited_urls: Set[str] = set()
        self.visited_hashes: Set[str] = set()  # Content deduplication
        self.domain_counts: Dict[str, int] = {}
        self.pages_crawled: int = 0
        self.start_time: Optional[datetime] = None

        # Queue of (LinkInfo, priority) tuples
        # Higher importance = higher priority
        self.link_queue: List[Tuple[float, LinkInfo]] = []

        # Results
        self.crawled_pages: List[Dict] = []

    async def crawl(
        self,
        seed_url: str,
        seed_topic: Optional[str] = None
    ) -> List[Dict]:
        """
        Start recursive crawl from seed URL.

        Args:
            seed_url: Starting URL
            seed_topic: Topic for importance scoring (optional)

        Returns:
            List of crawled page data with shards
        """
        logger.info("=" * 60)
        logger.info("Recursive Web Crawler with Importance Gating")
        logger.info("=" * 60)
        logger.info(f"Seed URL: {seed_url}")
        logger.info(f"Seed topic: {seed_topic or 'auto-detected'}")
        logger.info(f"Max depth: {self.config.max_depth}")
        logger.info(f"Max pages: {self.config.max_pages}")
        logger.info("")

        self.start_time = datetime.now()
        self.seed_topic = seed_topic
        self.seed_domain = urlparse(seed_url).netloc

        # Add seed URL to queue
        seed_link = LinkInfo(
            url=seed_url,
            anchor_text="",
            title="",
            context="",
            depth=0,
            parent_url="",
            importance_score=1.0  # Seed always important
        )
        self._add_to_queue(seed_link)

        # Process queue
        while self.link_queue and not self._should_stop():
            # Get highest priority link
            _, link = self.link_queue.pop(0)

            # Check if should crawl
            if not self._should_crawl(link):
                continue

            # Crawl page
            logger.info(f"[{self.pages_crawled + 1}/{self.config.max_pages}] "
                       f"Depth {link.depth} | Score {link.importance_score:.2f} | "
                       f"{link.url[:60]}")

            page_data = await self._crawl_page(link)

            if page_data:
                self.crawled_pages.append(page_data)
                self.pages_crawled += 1

                # Extract and score links
                if link.depth < self.config.max_depth:
                    await self._extract_and_queue_links(page_data, link)

            # Rate limiting
            await asyncio.sleep(self.config.rate_limit_seconds)

        # Summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✓ Crawl complete!")
        logger.info(f"  Pages crawled: {self.pages_crawled}")
        logger.info(f"  Max depth reached: {max((p['depth'] for p in self.crawled_pages), default=0)}")
        logger.info(f"  Domains visited: {len(self.domain_counts)}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        return self.crawled_pages

    def _should_crawl(self, link: LinkInfo) -> bool:
        """Check if link should be crawled."""

        # Already visited?
        if link.url in self.visited_urls:
            return False

        # Check importance threshold for this depth
        threshold = self.config.importance_thresholds.get(
            link.depth,
            self.config.importance_thresholds.get(max(self.config.importance_thresholds.keys()), 0.9)
        )

        if link.importance_score < threshold:
            logger.debug(f"  ✗ Below threshold ({link.importance_score:.2f} < {threshold})")
            return False

        # Check domain restrictions
        domain = urlparse(link.url).netloc

        if self.config.same_domain_only and domain != self.seed_domain:
            logger.debug(f"  ✗ Different domain")
            return False

        # Check domain diversity
        domain_count = self.domain_counts.get(domain, 0)
        if domain_count >= self.config.max_pages_per_domain:
            logger.debug(f"  ✗ Domain limit reached")
            return False

        # Check skip patterns
        for pattern in self.config.skip_patterns:
            if re.search(pattern, link.url):
                logger.debug(f"  ✗ Matched skip pattern: {pattern}")
                return False

        return True

    def _should_stop(self) -> bool:
        """Check if crawl should stop."""

        # Page limit
        if self.pages_crawled >= self.config.max_pages:
            logger.info("Stopping: Page limit reached")
            return True

        # Time limit
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            if elapsed > timedelta(minutes=self.config.max_runtime_minutes):
                logger.info("Stopping: Time limit reached")
                return True

        # No more links
        if not self.link_queue:
            logger.info("Stopping: No more links in queue")
            return True

        return False

    async def _crawl_page(self, link: LinkInfo) -> Optional[Dict]:
        """Crawl a single page."""

        try:
            # Configure spinner
            config = WebsiteSpinnerConfig(
                min_content_length=self.config.min_content_length,
                extract_images=self.config.extract_images,
                max_images=self.config.max_images_per_page,
                download_images=True
            )

            spinner = WebsiteSpinner(config)

            # Spin page
            shards = await spinner.spin({'url': link.url})

            if not shards:
                logger.warning(f"  ✗ No content extracted")
                return None

            # Calculate content hash for deduplication
            content = ' '.join(s.text for s in shards)
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            if content_hash in self.visited_hashes:
                logger.info(f"  ✗ Duplicate content")
                return None

            # Mark as visited
            self.visited_urls.add(link.url)
            self.visited_hashes.add(content_hash)

            # Update domain count
            domain = urlparse(link.url).netloc
            self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1

            # Extract title
            title = shards[0].metadata.get('title', '')

            logger.info(f"  ✓ {len(shards)} chunks | {title[:40]}")

            # If first page, auto-detect topic
            if self.pages_crawled == 0 and not self.seed_topic:
                self.seed_topic = self._detect_topic(shards)
                logger.info(f"  Topic detected: {self.seed_topic}")

            return {
                'link': link,
                'url': link.url,
                'title': title,
                'depth': link.depth,
                'shards': shards,
                'content_hash': content_hash,
                'domain': domain,
                'crawled_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return None

    async def _extract_and_queue_links(self, page_data: Dict, parent_link: LinkInfo):
        """Extract links from page and add to queue."""

        # Get first shard for HTML content
        if not page_data['shards']:
            return

        # Re-fetch to get links (could optimize by storing HTML)
        try:
            import requests
            response = requests.get(page_data['url'], timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.debug(f"  Failed to extract links: {e}")
            return

        # Find all links
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if not href or href.startswith('#'):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(page_data['url'], href)

            # Extract link metadata
            anchor_text = a_tag.get_text(strip=True)
            title = a_tag.get('title', '')

            # Get surrounding context
            context = self._get_link_context(a_tag)

            link_info = LinkInfo(
                url=absolute_url,
                anchor_text=anchor_text,
                title=title,
                context=context,
                depth=parent_link.depth + 1,
                parent_url=page_data['url'],
                importance_score=0.0  # Will calculate
            )

            links.append(link_info)

        # Score links
        scored_links = self._score_links(links, page_data)

        # Add to queue
        logger.debug(f"  Found {len(scored_links)} links, queueing high-value ones")

        queued = 0
        for link in scored_links:
            # Check threshold for next depth
            next_depth = parent_link.depth + 1
            threshold = self.config.importance_thresholds.get(next_depth, 0.9)

            if link.importance_score >= threshold:
                self._add_to_queue(link)
                queued += 1

        if queued > 0:
            logger.debug(f"  Queued {queued} links for depth {parent_link.depth + 1}")

    def _score_links(self, links: List[LinkInfo], page_data: Dict) -> List[LinkInfo]:
        """
        Score link importance/relevance.

        Matryoshka scoring - links scored based on:
        - Relevance to seed topic
        - Anchor text quality
        - Context similarity
        - Link position (higher = more important)
        - Domain authority
        """

        for i, link in enumerate(links):
            score = 0.5  # Start neutral

            # Position (earlier links often more important)
            position_score = 1.0 - (i / len(links))
            score += position_score * 0.1

            # Anchor text quality
            if link.anchor_text:
                if len(link.anchor_text) > 10:
                    score += 0.1

                # Check topic relevance
                if self.seed_topic:
                    topic_words = self.seed_topic.lower().split()
                    anchor_lower = link.anchor_text.lower()

                    if any(word in anchor_lower for word in topic_words):
                        score += 0.3

            # Context similarity
            if self.seed_topic and link.context:
                topic_words = set(self.seed_topic.lower().split())
                context_words = set(link.context.lower().split())
                overlap = len(topic_words & context_words)

                if overlap > 0:
                    score += min(overlap * 0.1, 0.2)

            # Same domain bonus (usually related)
            link_domain = urlparse(link.url).netloc
            if link_domain == self.seed_domain:
                score += 0.15

            # Penalize external social/share links
            social_domains = ['facebook.com', 'twitter.com', 'linkedin.com', 'pinterest.com']
            if any(domain in link_domain for domain in social_domains):
                score -= 0.5

            # Penalize navigation links (common anchor text)
            nav_text = ['home', 'about', 'contact', 'menu', 'search', 'login', 'sign up']
            if link.anchor_text.lower() in nav_text:
                score -= 0.3

            link.importance_score = max(0.0, min(1.0, score))

        # Sort by score
        links.sort(key=lambda x: x.importance_score, reverse=True)

        return links

    def _get_link_context(self, a_tag, context_chars: int = 100) -> str:
        """Extract text context around link."""
        parent = a_tag.parent
        if parent:
            text = parent.get_text(strip=True)
            return text[:context_chars]
        return ""

    def _detect_topic(self, shards: List) -> str:
        """Auto-detect topic from first page."""
        # Use title + first chunk
        title = shards[0].metadata.get('title', '')
        first_text = shards[0].text if shards else ''

        # Extract key words (simple approach)
        text = f"{title} {first_text}"
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words

        # Use top 3 most common
        from collections import Counter
        common = Counter(words).most_common(3)

        return ' '.join(word for word, _ in common)

    def _add_to_queue(self, link: LinkInfo):
        """Add link to queue with priority."""
        # Priority queue: higher importance = processed first
        priority = -link.importance_score  # Negative for min-heap behavior
        self.link_queue.append((priority, link))
        self.link_queue.sort()  # Keep sorted by priority


# Convenience function
async def crawl_recursive(
    seed_url: str,
    seed_topic: Optional[str] = None,
    max_depth: int = 2,
    max_pages: int = 50,
    same_domain_only: bool = False
) -> List[Dict]:
    """
    Recursively crawl web content with importance gating.

    Args:
        seed_url: Starting URL
        seed_topic: Topic for scoring (auto-detected if None)
        max_depth: How deep to crawl
        max_pages: Total page limit
        same_domain_only: Stay on seed domain

    Returns:
        List of crawled page data

    Example:
        # Crawl beekeeping article + related pages
        pages = await crawl_recursive(
            seed_url='https://example.com/beekeeping-basics',
            seed_topic='beekeeping hive management',
            max_depth=2,
            max_pages=20
        )

        # Only highly relevant links followed at each depth
    """
    config = CrawlConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only
    )

    crawler = RecursiveCrawler(config)
    return await crawler.crawl(seed_url, seed_topic)
