"""
Recursive Web Crawler with Matryoshka Importance Gating

Implements hierarchical importance-based crawling where:
- Importance thresholds increase with depth (like nested dolls)
- Prevents crawl explosion by filtering unimportant branches early
- Combines topic relevance, link quality, and structural signals

Created: 2025-12-17
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib

from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceSignals,
    ImportanceScore,
)
from HoloLoom.protocols.types import MemoryShard

# Import WebsiteSpinner for page fetching
try:
    from HoloLoom.spinningWheel.website import WebsiteSpinner, WebPageContent
    WEBSITE_SPINNER_AVAILABLE = True
except ImportError:
    WEBSITE_SPINNER_AVAILABLE = False


@dataclass
class LinkInfo:
    """Information about a discovered link."""
    url: str
    text: str
    context: str  # Surrounding text
    source_url: str
    depth: int
    position: str  # 'main', 'nav', 'sidebar', 'footer'
    importance_score: float = 0.0


@dataclass
class CrawlConfig:
    """Configuration for recursive crawling."""
    max_depth: int = 3
    max_pages: int = 50
    max_pages_per_domain: int = 20

    # Matryoshka importance thresholds (increase with depth)
    base_threshold: float = 0.3  # Depth 0
    threshold_increment: float = 0.15  # Per depth level
    max_threshold: float = 0.85  # Cap

    # Link scoring weights
    relevance_weight: float = 0.4
    position_weight: float = 0.2
    text_quality_weight: float = 0.2
    domain_weight: float = 0.2

    # Filtering
    same_domain_only: bool = False
    allowed_domains: Optional[List[str]] = None
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'login', r'signup', r'register', r'cart', r'checkout',
        r'privacy', r'terms', r'cookie', r'gdpr',
        r'\.pdf$', r'\.zip$', r'\.exe$', r'\.dmg$',
        r'facebook\.com', r'twitter\.com', r'linkedin\.com',
        r'instagram\.com', r'youtube\.com/watch',
    ])

    # Rate limiting
    delay_between_requests: float = 1.0  # seconds

    # Topic for relevance scoring
    seed_topic: str = ""


@dataclass
class CrawlState:
    """State tracking for crawl session."""
    visited_urls: Set[str] = field(default_factory=set)
    url_queue: List[LinkInfo] = field(default_factory=list)
    pages_crawled: int = 0
    pages_per_domain: Dict[str, int] = field(default_factory=dict)
    discovered_links: int = 0
    filtered_links: int = 0
    shards_created: int = 0


class MatryoshkaImportanceGate:
    """
    Hierarchical importance gating inspired by Matryoshka embeddings.

    Like nested Russian dolls, each level is more selective:
    - Outer level (depth 0): Accept most relevant content
    - Inner levels (depth 1, 2, 3...): Increasingly strict filtering

    This prevents crawl explosion by cutting off unimportant branches early.
    """

    def __init__(self, config: CrawlConfig):
        self.config = config

        # Position importance multipliers
        self.position_scores = {
            'main': 1.0,      # Main content area
            'article': 0.95,  # Article body
            'nav': 0.6,       # Navigation
            'sidebar': 0.5,   # Sidebar
            'footer': 0.3,    # Footer
            'header': 0.4,    # Header
            'unknown': 0.5,
        }

        # Compile blocked patterns
        self.blocked_re = [re.compile(p, re.I) for p in config.blocked_patterns]

    def get_threshold_for_depth(self, depth: int) -> float:
        """Get importance threshold for given depth (matryoshka gating)."""
        threshold = self.config.base_threshold + (depth * self.config.threshold_increment)
        return min(threshold, self.config.max_threshold)

    def score_link(
        self,
        link: LinkInfo,
        seed_topic: str,
        source_content: str = ""
    ) -> float:
        """
        Score a link's importance for crawling.

        Combines multiple signals:
        - Topic relevance (how related to seed topic)
        - Position quality (main content vs sidebar)
        - Link text quality (descriptive vs generic)
        - Domain authority (simplified)
        """
        scores = {}

        # 1. Topic relevance (0-1)
        scores['relevance'] = self._score_relevance(link, seed_topic, source_content)

        # 2. Position quality (0-1)
        scores['position'] = self.position_scores.get(link.position, 0.5)

        # 3. Link text quality (0-1)
        scores['text_quality'] = self._score_link_text(link.text)

        # 4. Domain score (0-1)
        scores['domain'] = self._score_domain(link.url)

        # Weighted combination
        final_score = (
            scores['relevance'] * self.config.relevance_weight +
            scores['position'] * self.config.position_weight +
            scores['text_quality'] * self.config.text_quality_weight +
            scores['domain'] * self.config.domain_weight
        )

        return final_score

    def _score_relevance(
        self,
        link: LinkInfo,
        seed_topic: str,
        source_content: str
    ) -> float:
        """Score topic relevance using keyword overlap."""
        if not seed_topic:
            return 0.5  # Neutral if no topic

        # Tokenize
        topic_words = set(seed_topic.lower().split())
        link_text_words = set(link.text.lower().split())
        context_words = set(link.context.lower().split())
        url_words = set(re.split(r'[/\-_.]', link.url.lower()))

        # Check overlap
        text_overlap = len(topic_words & link_text_words)
        context_overlap = len(topic_words & context_words)
        url_overlap = len(topic_words & url_words)

        # Normalize (assuming topic has at least 1 word)
        max_possible = len(topic_words)
        if max_possible == 0:
            return 0.5

        # Weight: link text > context > URL
        score = (
            (text_overlap / max_possible) * 0.5 +
            (min(context_overlap, 3) / 3) * 0.3 +
            (min(url_overlap, 2) / 2) * 0.2
        )

        return min(1.0, score)

    def _score_link_text(self, text: str) -> float:
        """Score link text quality (descriptive vs generic)."""
        text = text.strip().lower()

        # Generic/poor link text
        poor_texts = {
            'click here', 'read more', 'learn more', 'here',
            'link', 'this', 'more', 'continue', 'next',
            '>', '>>', '...', '[more]', '[link]'
        }

        if text in poor_texts or len(text) < 3:
            return 0.2

        # Very short
        if len(text) < 10:
            return 0.4

        # Good descriptive text
        if len(text) > 20 and len(text) < 100:
            return 0.9

        # Moderate length
        return 0.7

    def _score_domain(self, url: str) -> float:
        """Simple domain scoring (could be enhanced with authority data)."""
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return 0.5

        # Prefer certain TLDs
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return 0.9
        if domain.endswith('.org'):
            return 0.8

        # Penalize certain patterns
        if 'blog' in domain or 'wordpress' in domain:
            return 0.6

        return 0.7

    def should_crawl(self, link: LinkInfo, state: CrawlState) -> Tuple[bool, str]:
        """
        Determine if a link should be crawled (matryoshka gating).

        Returns:
            (should_crawl, reason)
        """
        url = link.url

        # Already visited
        if url in state.visited_urls:
            return False, "already_visited"

        # URL normalization check
        normalized = self._normalize_url(url)
        if normalized in state.visited_urls:
            return False, "already_visited_normalized"

        # Blocked patterns
        for pattern in self.blocked_re:
            if pattern.search(url):
                return False, f"blocked_pattern:{pattern.pattern}"

        # Domain limits
        try:
            domain = urlparse(url).netloc
        except Exception:
            return False, "invalid_url"

        if state.pages_per_domain.get(domain, 0) >= self.config.max_pages_per_domain:
            return False, "domain_limit"

        # Same domain check
        if self.config.same_domain_only:
            source_domain = urlparse(link.source_url).netloc
            if domain != source_domain:
                return False, "different_domain"

        # Allowed domains check
        if self.config.allowed_domains:
            if not any(d in domain for d in self.config.allowed_domains):
                return False, "not_allowed_domain"

        # Max pages limit
        if state.pages_crawled >= self.config.max_pages:
            return False, "max_pages_reached"

        # Max depth limit
        if link.depth > self.config.max_depth:
            return False, "max_depth_exceeded"

        # MATRYOSHKA GATING: Check importance against depth-based threshold
        threshold = self.get_threshold_for_depth(link.depth)
        if link.importance_score < threshold:
            return False, f"below_threshold:{link.importance_score:.2f}<{threshold:.2f}"

        return True, "accepted"

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        try:
            parsed = urlparse(url)
            # Remove trailing slash, fragments, common tracking params
            path = parsed.path.rstrip('/')
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        except Exception:
            return url


class RecursiveCrawlerSpinner(BaseSpinner):
    """
    Recursive web crawler with matryoshka importance gating.

    Uses hierarchical importance thresholds that increase with depth,
    preventing crawl explosion while capturing relevant content.

    Example:
        ```python
        config = CrawlConfig(
            max_depth=3,
            max_pages=50,
            seed_topic="machine learning transformers"
        )
        crawler = RecursiveCrawlerSpinner(config)
        result = await crawler.spin("https://example.com/ml-guide")

        # Access crawled content
        for shard in result.shards:
            print(f"{shard.metadata['url']}: {shard.text[:100]}...")
        ```
    """

    def __init__(self, config: Optional[CrawlConfig] = None):
        super().__init__(name="recursive_crawler")
        self.config = config or CrawlConfig()
        self.gate = MatryoshkaImportanceGate(self.config)

        if WEBSITE_SPINNER_AVAILABLE:
            self.page_spinner = WebsiteSpinner()
        else:
            self.page_spinner = None

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=True,  # Supports streaming crawl
            batch_processing=True,
            importance_scoring=True,  # Matryoshka gating
            supported_formats=['url', 'urls'],
            max_input_size=100 * 1024 * 1024,  # 100MB total
        )

    def is_available(self) -> bool:
        return WEBSITE_SPINNER_AVAILABLE

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Recursively crawl starting from seed URL(s).

        Args:
            source: Single URL string or list of URLs
            **kwargs:
                topic: Override seed_topic from config
                max_depth: Override max_depth
                max_pages: Override max_pages

        Returns:
            List of MemoryShards from crawled pages
        """
        if not self.page_spinner:
            raise RuntimeError("WebsiteSpinner not available - install required dependencies")

        # Handle single URL or list
        if isinstance(source, str):
            seed_urls = [source]
        elif isinstance(source, list):
            seed_urls = source
        else:
            raise ValueError(f"Expected URL string or list, got {type(source)}")

        # Override config from kwargs
        topic = kwargs.get('topic', self.config.seed_topic)
        max_depth = kwargs.get('max_depth', self.config.max_depth)
        max_pages = kwargs.get('max_pages', self.config.max_pages)

        # Update config temporarily
        original_topic = self.config.seed_topic
        original_depth = self.config.max_depth
        original_pages = self.config.max_pages

        self.config.seed_topic = topic
        self.config.max_depth = max_depth
        self.config.max_pages = max_pages

        try:
            shards = await self._crawl(seed_urls)
            return shards
        finally:
            # Restore config
            self.config.seed_topic = original_topic
            self.config.max_depth = original_depth
            self.config.max_pages = original_pages

    async def _crawl(self, seed_urls: List[str]) -> List[MemoryShard]:
        """Execute the recursive crawl."""
        state = CrawlState()
        all_shards: List[MemoryShard] = []

        # Initialize queue with seed URLs (depth 0, high importance)
        for url in seed_urls:
            link = LinkInfo(
                url=url,
                text=url,  # Use URL as text for seeds
                context="",
                source_url="",
                depth=0,
                position='main',
                importance_score=1.0  # Seeds always pass
            )
            state.url_queue.append(link)

        # BFS crawl with importance gating
        while state.url_queue and state.pages_crawled < self.config.max_pages:
            # Get next link from queue
            current_link = state.url_queue.pop(0)

            # Check if should crawl (matryoshka gating)
            should_crawl, reason = self.gate.should_crawl(current_link, state)

            if not should_crawl:
                state.filtered_links += 1
                continue

            # Mark as visited
            state.visited_urls.add(current_link.url)
            state.visited_urls.add(self.gate._normalize_url(current_link.url))

            # Update domain counter
            try:
                domain = urlparse(current_link.url).netloc
                state.pages_per_domain[domain] = state.pages_per_domain.get(domain, 0) + 1
            except Exception:
                pass

            # Fetch and process page
            try:
                page_shards, discovered_links = await self._process_page(
                    current_link,
                    state
                )

                all_shards.extend(page_shards)
                state.pages_crawled += 1
                state.shards_created += len(page_shards)

                # Score and queue discovered links
                for link in discovered_links:
                    # Score importance
                    link.importance_score = self.gate.score_link(
                        link,
                        self.config.seed_topic,
                        ""  # Could pass page content for better relevance
                    )
                    state.discovered_links += 1

                    # Add to queue (will be filtered by matryoshka gate later)
                    state.url_queue.append(link)

                # Rate limiting
                if self.config.delay_between_requests > 0:
                    await asyncio.sleep(self.config.delay_between_requests)

            except Exception as e:
                # Log error but continue crawling
                print(f"Error crawling {current_link.url}: {e}")
                continue

        # Add crawl summary shard
        summary_shard = self._create_summary_shard(state, seed_urls)
        all_shards.append(summary_shard)

        return all_shards

    async def _process_page(
        self,
        link: LinkInfo,
        state: CrawlState
    ) -> Tuple[List[MemoryShard], List[LinkInfo]]:
        """
        Process a single page: extract content and discover links.

        Returns:
            (page_shards, discovered_links)
        """
        # Use WebsiteSpinner to fetch page
        result = await self.page_spinner.spin(link.url)

        if not result.success or not result.shards:
            return [], []

        # Get page shards
        page_shards = result.shards

        # Add crawl metadata to shards
        for shard in page_shards:
            shard.metadata['crawl_depth'] = link.depth
            shard.metadata['crawl_source'] = link.source_url
            shard.metadata['crawl_importance'] = link.importance_score
            shard.metadata['crawl_timestamp'] = datetime.now().isoformat()

        # Extract links for further crawling
        discovered_links = await self._extract_links(link, page_shards)

        return page_shards, discovered_links

    async def _extract_links(
        self,
        source_link: LinkInfo,
        page_shards: List[MemoryShard]
    ) -> List[LinkInfo]:
        """Extract links from page content for potential crawling."""
        discovered = []

        # Get main content shard
        main_shard = None
        for shard in page_shards:
            if shard.metadata.get('content_type') == 'webpage':
                main_shard = shard
                break

        if not main_shard:
            return []

        # Parse HTML from metadata if available
        html_content = main_shard.metadata.get('raw_html', '')

        if not html_content:
            # Fall back to link extraction from text
            return self._extract_links_from_text(source_link, main_shard.text)

        # Use BeautifulSoup for proper link extraction
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all links
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']

                # Skip non-http links
                if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    continue

                # Resolve relative URLs
                full_url = urljoin(source_link.url, href)

                # Skip non-http URLs
                if not full_url.startswith(('http://', 'https://')):
                    continue

                # Get link text and context
                link_text = a_tag.get_text(strip=True)
                context = self._get_link_context(a_tag)
                position = self._detect_link_position(a_tag)

                link_info = LinkInfo(
                    url=full_url,
                    text=link_text,
                    context=context,
                    source_url=source_link.url,
                    depth=source_link.depth + 1,
                    position=position,
                )

                discovered.append(link_info)

        except ImportError:
            # Fall back to regex
            return self._extract_links_from_text(source_link, main_shard.text)
        except Exception as e:
            print(f"Error extracting links: {e}")
            return []

        return discovered

    def _get_link_context(self, a_tag) -> str:
        """Get surrounding text context for a link."""
        try:
            # Get parent paragraph or container
            parent = a_tag.find_parent(['p', 'div', 'li', 'td', 'article'])
            if parent:
                text = parent.get_text(strip=True)
                # Limit context length
                return text[:200] if len(text) > 200 else text
        except Exception:
            pass
        return ""

    def _detect_link_position(self, a_tag) -> str:
        """Detect the semantic position of a link on the page."""
        try:
            # Check parent elements for semantic hints
            for parent in a_tag.parents:
                tag_name = parent.name.lower() if parent.name else ''
                classes = ' '.join(parent.get('class', [])).lower()
                element_id = (parent.get('id') or '').lower()

                # Check tag name
                if tag_name in ('nav', 'navigation'):
                    return 'nav'
                if tag_name in ('footer',):
                    return 'footer'
                if tag_name in ('header',):
                    return 'header'
                if tag_name in ('aside',):
                    return 'sidebar'
                if tag_name in ('article', 'main'):
                    return 'main'

                # Check classes/id
                combined = f"{classes} {element_id}"
                if any(x in combined for x in ['nav', 'menu', 'navigation']):
                    return 'nav'
                if any(x in combined for x in ['footer', 'foot']):
                    return 'footer'
                if any(x in combined for x in ['header', 'head']):
                    return 'header'
                if any(x in combined for x in ['sidebar', 'side', 'widget']):
                    return 'sidebar'
                if any(x in combined for x in ['main', 'content', 'article', 'post', 'entry']):
                    return 'main'
        except Exception:
            pass

        return 'unknown'

    def _extract_links_from_text(
        self,
        source_link: LinkInfo,
        text: str
    ) -> List[LinkInfo]:
        """Fallback link extraction using regex."""
        discovered = []

        # Simple URL pattern
        url_pattern = r'https?://[^\s<>"\']+[^\s<>"\'\.,;:!?\)]'

        for match in re.finditer(url_pattern, text):
            url = match.group()

            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            link_info = LinkInfo(
                url=url,
                text=url,  # No anchor text available
                context=context,
                source_url=source_link.url,
                depth=source_link.depth + 1,
                position='unknown',
            )

            discovered.append(link_info)

        return discovered

    def _create_summary_shard(
        self,
        state: CrawlState,
        seed_urls: List[str]
    ) -> MemoryShard:
        """Create a summary shard for the entire crawl."""
        summary_text = f"""Web Crawl Summary
================
Seed URLs: {', '.join(seed_urls)}
Topic: {self.config.seed_topic or 'None specified'}

Statistics:
- Pages crawled: {state.pages_crawled}
- Shards created: {state.shards_created}
- Links discovered: {state.discovered_links}
- Links filtered: {state.filtered_links}
- Domains visited: {len(state.pages_per_domain)}

Top domains:
{chr(10).join(f'- {domain}: {count} pages' for domain, count in sorted(state.pages_per_domain.items(), key=lambda x: -x[1])[:10])}

Matryoshka Gating Thresholds:
- Depth 0: {self.gate.get_threshold_for_depth(0):.2f}
- Depth 1: {self.gate.get_threshold_for_depth(1):.2f}
- Depth 2: {self.gate.get_threshold_for_depth(2):.2f}
- Depth 3: {self.gate.get_threshold_for_depth(3):.2f}
"""

        return MemoryShard(
            id=f"crawl_summary_{hashlib.md5(str(seed_urls).encode()).hexdigest()[:8]}",
            text=summary_text,
            episode="web_crawl",
            entities=['web_crawl', 'summary'] + [urlparse(u).netloc for u in seed_urls],
            motifs=['recursive_crawl', 'matryoshka_gating'],
            metadata={
                'content_type': 'crawl_summary',
                'seed_urls': seed_urls,
                'topic': self.config.seed_topic,
                'pages_crawled': state.pages_crawled,
                'links_discovered': state.discovered_links,
                'links_filtered': state.filtered_links,
                'domains_visited': len(state.pages_per_domain),
                'timestamp': datetime.now().isoformat(),
            }
        )

    def score_importance(self, shard: MemoryShard) -> ImportanceScore:
        """Score the importance of a crawled shard."""
        signals = ImportanceSignals()

        # Length signal
        text_len = len(shard.text)
        signals.length_score = min(1.0, text_len / 2000)

        # Crawl depth (deeper = potentially less relevant)
        depth = shard.metadata.get('crawl_depth', 0)
        signals.custom_signals['depth_penalty'] = max(0, 1.0 - (depth * 0.15))

        # Crawl importance (from matryoshka gating)
        crawl_importance = shard.metadata.get('crawl_importance', 0.5)
        signals.custom_signals['crawl_importance'] = crawl_importance

        # Technical content (code, data)
        if any(marker in shard.text.lower() for marker in ['```', 'def ', 'class ', 'function']):
            signals.technical_score = 0.8
        else:
            signals.technical_score = 0.3

        # Calculate weighted score
        base_score = (
            signals.length_score * 0.2 +
            signals.technical_score * 0.2 +
            signals.custom_signals.get('crawl_importance', 0.5) * 0.4 +
            signals.custom_signals.get('depth_penalty', 1.0) * 0.2
        )

        return ImportanceScore(
            score=base_score,
            signals=signals,
            reason=f"Crawl depth {depth}, importance {crawl_importance:.2f}"
        )


# Convenience functions
async def crawl_website(
    url: str,
    topic: str = "",
    max_depth: int = 2,
    max_pages: int = 30,
    same_domain: bool = True
) -> SpinResult:
    """
    Convenience function to crawl a website with matryoshka importance gating.

    Args:
        url: Starting URL
        topic: Topic keywords for relevance scoring
        max_depth: Maximum crawl depth (default 2)
        max_pages: Maximum pages to crawl (default 30)
        same_domain: Whether to stay on same domain (default True)

    Returns:
        SpinResult with crawled content shards
    """
    config = CrawlConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain,
        seed_topic=topic,
    )

    crawler = RecursiveCrawlerSpinner(config)
    return await crawler.spin(url)


async def crawl_multiple(
    urls: List[str],
    topic: str = "",
    max_depth: int = 2,
    max_pages: int = 50
) -> SpinResult:
    """
    Crawl multiple seed URLs with shared topic.

    Args:
        urls: List of starting URLs
        topic: Topic keywords for relevance scoring
        max_depth: Maximum crawl depth
        max_pages: Total maximum pages across all seeds

    Returns:
        SpinResult with crawled content shards
    """
    config = CrawlConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=False,
        seed_topic=topic,
    )

    crawler = RecursiveCrawlerSpinner(config)
    return await crawler.spin(urls)
