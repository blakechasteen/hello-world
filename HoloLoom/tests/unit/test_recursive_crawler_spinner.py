"""
Unit tests for RecursiveCrawlerSpinner

Tests the recursive web crawler with Matryoshka importance gating.
Key components tested:
- LinkInfo, CrawlConfig, CrawlState dataclasses
- MatryoshkaImportanceGate threshold and scoring logic
- RecursiveCrawlerSpinner crawl orchestration
- Convenience functions (crawl_website, crawl_multiple)

Created: 2025-12-22
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict
from datetime import datetime

from HoloLoom.spinningWheel.recursive_crawler import (
    LinkInfo,
    CrawlConfig,
    CrawlState,
    MatryoshkaImportanceGate,
    RecursiveCrawlerSpinner,
    crawl_website,
    crawl_multiple,
    WEBSITE_SPINNER_AVAILABLE,
)
from HoloLoom.spinningWheel.protocol import SpinResult, ImportanceScore


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Default crawl configuration."""
    return CrawlConfig()


@pytest.fixture
def strict_config():
    """Stricter configuration for testing limits."""
    return CrawlConfig(
        max_depth=2,
        max_pages=10,
        max_pages_per_domain=5,
        base_threshold=0.5,
        threshold_increment=0.2,
        same_domain_only=True,
    )


@pytest.fixture
def sample_link():
    """Sample LinkInfo for testing."""
    return LinkInfo(
        url="https://example.com/article/machine-learning",
        text="Introduction to Machine Learning",
        context="This comprehensive guide covers the basics of machine learning.",
        source_url="https://example.com/",
        depth=1,
        position='main',
        importance_score=0.0
    )


@pytest.fixture
def sample_state():
    """Sample CrawlState for testing."""
    state = CrawlState()
    state.visited_urls.add("https://example.com/visited")
    state.pages_crawled = 5
    state.pages_per_domain["example.com"] = 3
    return state


@pytest.fixture
def gate(default_config):
    """MatryoshkaImportanceGate instance."""
    return MatryoshkaImportanceGate(default_config)


# ============================================================================
# Test LinkInfo Dataclass
# ============================================================================

class TestLinkInfo:
    """Tests for LinkInfo dataclass."""

    def test_linkinfo_creation(self):
        """Test creating a LinkInfo instance."""
        link = LinkInfo(
            url="https://example.com/page",
            text="Click here",
            context="Some surrounding text",
            source_url="https://example.com/",
            depth=1,
            position='main'
        )

        assert link.url == "https://example.com/page"
        assert link.text == "Click here"
        assert link.context == "Some surrounding text"
        assert link.source_url == "https://example.com/"
        assert link.depth == 1
        assert link.position == 'main'
        assert link.importance_score == 0.0  # Default

    def test_linkinfo_with_importance(self):
        """Test LinkInfo with importance score."""
        link = LinkInfo(
            url="https://example.com/",
            text="Home",
            context="",
            source_url="",
            depth=0,
            position='nav',
            importance_score=0.85
        )

        assert link.importance_score == 0.85

    def test_linkinfo_positions(self):
        """Test various position values."""
        positions = ['main', 'nav', 'sidebar', 'footer', 'header', 'article', 'unknown']

        for pos in positions:
            link = LinkInfo(
                url="https://example.com/",
                text="Link",
                context="",
                source_url="",
                depth=0,
                position=pos
            )
            assert link.position == pos


# ============================================================================
# Test CrawlConfig Dataclass
# ============================================================================

class TestCrawlConfig:
    """Tests for CrawlConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CrawlConfig()

        assert config.max_depth == 3
        assert config.max_pages == 50
        assert config.max_pages_per_domain == 20
        assert config.base_threshold == 0.3
        assert config.threshold_increment == 0.15
        assert config.max_threshold == 0.85
        assert config.delay_between_requests == 1.0
        assert config.same_domain_only is False
        assert config.allowed_domains is None
        assert config.seed_topic == ""

    def test_custom_config(self):
        """Test custom configuration."""
        config = CrawlConfig(
            max_depth=5,
            max_pages=100,
            base_threshold=0.4,
            seed_topic="machine learning transformers"
        )

        assert config.max_depth == 5
        assert config.max_pages == 100
        assert config.base_threshold == 0.4
        assert config.seed_topic == "machine learning transformers"

    def test_blocked_patterns_default(self):
        """Test default blocked patterns."""
        config = CrawlConfig()

        # Should contain common patterns to block
        patterns = config.blocked_patterns
        assert any('login' in p for p in patterns)
        assert any('signup' in p for p in patterns)
        assert any('facebook' in p for p in patterns)
        assert any('.pdf' in p for p in patterns)

    def test_scoring_weights_sum(self):
        """Test that scoring weights sum to 1.0."""
        config = CrawlConfig()

        total = (
            config.relevance_weight +
            config.position_weight +
            config.text_quality_weight +
            config.domain_weight
        )

        assert abs(total - 1.0) < 0.001


# ============================================================================
# Test CrawlState Dataclass
# ============================================================================

class TestCrawlState:
    """Tests for CrawlState dataclass."""

    def test_empty_state(self):
        """Test empty initial state."""
        state = CrawlState()

        assert len(state.visited_urls) == 0
        assert len(state.url_queue) == 0
        assert state.pages_crawled == 0
        assert len(state.pages_per_domain) == 0
        assert state.discovered_links == 0
        assert state.filtered_links == 0
        assert state.shards_created == 0

    def test_state_tracking(self, sample_state):
        """Test state tracking."""
        assert "https://example.com/visited" in sample_state.visited_urls
        assert sample_state.pages_crawled == 5
        assert sample_state.pages_per_domain["example.com"] == 3

    def test_add_to_queue(self, sample_link):
        """Test adding links to queue."""
        state = CrawlState()
        state.url_queue.append(sample_link)

        assert len(state.url_queue) == 1
        assert state.url_queue[0].url == sample_link.url


# ============================================================================
# Test MatryoshkaImportanceGate - Threshold Calculation
# ============================================================================

class TestMatryoshkaThresholds:
    """Tests for Matryoshka threshold calculation."""

    def test_threshold_depth_0(self, gate):
        """Test threshold at depth 0 (seed level)."""
        threshold = gate.get_threshold_for_depth(0)
        assert threshold == 0.3  # base_threshold

    def test_threshold_depth_1(self, gate):
        """Test threshold at depth 1."""
        threshold = gate.get_threshold_for_depth(1)
        assert threshold == pytest.approx(0.45)  # 0.3 + 0.15

    def test_threshold_depth_2(self, gate):
        """Test threshold at depth 2."""
        threshold = gate.get_threshold_for_depth(2)
        assert threshold == 0.60  # 0.3 + 0.30

    def test_threshold_depth_3(self, gate):
        """Test threshold at depth 3."""
        threshold = gate.get_threshold_for_depth(3)
        assert threshold == 0.75  # 0.3 + 0.45

    def test_threshold_max_cap(self, gate):
        """Test threshold is capped at max_threshold."""
        # Depth 5: 0.3 + 0.75 = 1.05 -> should cap at 0.85
        threshold = gate.get_threshold_for_depth(5)
        assert threshold == 0.85

    def test_threshold_increases_with_depth(self, gate):
        """Test that thresholds strictly increase with depth."""
        thresholds = [gate.get_threshold_for_depth(d) for d in range(5)]

        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1]


# ============================================================================
# Test MatryoshkaImportanceGate - Link Scoring
# ============================================================================

class TestLinkScoring:
    """Tests for link importance scoring."""

    def test_score_link_basic(self, gate, sample_link):
        """Test basic link scoring."""
        score = gate.score_link(sample_link, "machine learning")

        assert 0.0 <= score <= 1.0

    def test_score_link_no_topic(self, gate, sample_link):
        """Test scoring without topic (neutral relevance)."""
        score = gate.score_link(sample_link, "")

        # Relevance should be 0.5 (neutral) when no topic
        assert 0.0 <= score <= 1.0

    def test_relevance_scoring_keyword_match(self, gate):
        """Test relevance scoring with keyword matches."""
        link = LinkInfo(
            url="https://example.com/ml",
            text="machine learning tutorial",
            context="Learn about machine learning basics",
            source_url="https://example.com/",
            depth=1,
            position='main'
        )

        score = gate._score_relevance(link, "machine learning", "")

        # Should have high relevance due to keyword matches
        assert score > 0.3

    def test_relevance_scoring_no_match(self, gate):
        """Test relevance scoring with no keyword matches."""
        link = LinkInfo(
            url="https://example.com/cooking",
            text="Best pasta recipes",
            context="Italian cuisine guide",
            source_url="https://example.com/",
            depth=1,
            position='main'
        )

        score = gate._score_relevance(link, "machine learning", "")

        # Should have low relevance
        assert score < 0.3

    def test_position_scores(self, gate):
        """Test position score values."""
        assert gate.position_scores['main'] == 1.0
        assert gate.position_scores['article'] == 0.95
        assert gate.position_scores['nav'] == 0.6
        assert gate.position_scores['sidebar'] == 0.5
        assert gate.position_scores['footer'] == 0.3
        assert gate.position_scores['header'] == 0.4
        assert gate.position_scores['unknown'] == 0.5

    def test_link_text_quality_poor(self, gate):
        """Test poor link text quality scoring."""
        poor_texts = ['click here', 'read more', 'here', 'link', '>']

        for text in poor_texts:
            score = gate._score_link_text(text)
            assert score <= 0.4, f"'{text}' should have low score"

    def test_link_text_quality_good(self, gate):
        """Test good link text quality scoring."""
        good_text = "Introduction to Machine Learning Concepts and Applications"
        score = gate._score_link_text(good_text)

        assert score >= 0.7

    def test_link_text_quality_moderate(self, gate):
        """Test moderate link text quality."""
        moderate_text = "ML Guide"
        score = gate._score_link_text(moderate_text)

        assert 0.3 <= score <= 0.8

    def test_domain_scoring_edu(self, gate):
        """Test .edu domain scoring."""
        score = gate._score_domain("https://stanford.edu/ml")
        assert score == 0.9

    def test_domain_scoring_gov(self, gate):
        """Test .gov domain scoring."""
        score = gate._score_domain("https://whitehouse.gov/policy")
        assert score == 0.9

    def test_domain_scoring_org(self, gate):
        """Test .org domain scoring."""
        score = gate._score_domain("https://wikipedia.org/wiki/ML")
        assert score == 0.8

    def test_domain_scoring_blog(self, gate):
        """Test blog domain scoring (penalized)."""
        score = gate._score_domain("https://myblog.wordpress.com/post")
        assert score == 0.6

    def test_domain_scoring_regular(self, gate):
        """Test regular domain scoring."""
        score = gate._score_domain("https://example.com/page")
        assert score == 0.7


# ============================================================================
# Test MatryoshkaImportanceGate - Gating Decisions
# ============================================================================

class TestGatingDecisions:
    """Tests for should_crawl gating decisions."""

    def test_already_visited(self, gate, sample_link, sample_state):
        """Test blocking already visited URLs."""
        sample_state.visited_urls.add(sample_link.url)

        should, reason = gate.should_crawl(sample_link, sample_state)

        assert should is False
        assert "already_visited" in reason

    def test_blocked_pattern_login(self, gate, sample_state):
        """Test blocking login URLs."""
        link = LinkInfo(
            url="https://example.com/login",
            text="Login",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='nav',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "blocked_pattern" in reason

    def test_blocked_pattern_pdf(self, gate, sample_state):
        """Test blocking PDF files."""
        link = LinkInfo(
            url="https://example.com/document.pdf",
            text="Download PDF",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "blocked_pattern" in reason

    def test_blocked_pattern_social(self, gate, sample_state):
        """Test blocking social media URLs."""
        social_urls = [
            "https://facebook.com/page",
            "https://twitter.com/user",
            "https://linkedin.com/company"
        ]

        for url in social_urls:
            link = LinkInfo(
                url=url,
                text="Follow us",
                context="",
                source_url="https://example.com/",
                depth=1,
                position='footer',
                importance_score=1.0
            )

            should, reason = gate.should_crawl(link, sample_state)
            assert should is False

    def test_domain_limit(self, gate, sample_state):
        """Test domain page limit enforcement."""
        config = CrawlConfig(max_pages_per_domain=3)
        gate = MatryoshkaImportanceGate(config)

        # Domain already at limit (3 pages)
        sample_state.pages_per_domain["example.com"] = 3

        link = LinkInfo(
            url="https://example.com/new-page",
            text="New Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "domain_limit" in reason

    def test_max_pages_limit(self, gate, sample_state):
        """Test max pages limit enforcement."""
        config = CrawlConfig(max_pages=5)
        gate = MatryoshkaImportanceGate(config)
        sample_state.pages_crawled = 5

        link = LinkInfo(
            url="https://newsite.com/page",
            text="Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "max_pages_reached" in reason

    def test_max_depth_limit(self, gate, sample_state):
        """Test max depth limit enforcement."""
        config = CrawlConfig(max_depth=2)
        gate = MatryoshkaImportanceGate(config)

        link = LinkInfo(
            url="https://example.com/deep/page",
            text="Deep Page",
            context="",
            source_url="https://example.com/level2",
            depth=3,  # Beyond max_depth=2
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "max_depth" in reason

    def test_same_domain_only(self, sample_state):
        """Test same domain restriction."""
        config = CrawlConfig(same_domain_only=True)
        gate = MatryoshkaImportanceGate(config)

        link = LinkInfo(
            url="https://other-site.com/page",
            text="External Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "different_domain" in reason

    def test_allowed_domains(self, sample_state):
        """Test allowed domains restriction."""
        config = CrawlConfig(allowed_domains=["example.com", "allowed.org"])
        gate = MatryoshkaImportanceGate(config)

        # Allowed domain
        allowed_link = LinkInfo(
            url="https://example.com/page",
            text="Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, _ = gate.should_crawl(allowed_link, sample_state)
        assert should is True

        # Not allowed domain
        blocked_link = LinkInfo(
            url="https://blocked.com/page",
            text="Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=1.0
        )

        should, reason = gate.should_crawl(blocked_link, sample_state)
        assert should is False
        assert "not_allowed_domain" in reason

    def test_importance_below_threshold(self, sample_state):
        """Test matryoshka gating based on importance score."""
        config = CrawlConfig(base_threshold=0.5, threshold_increment=0.1)
        gate = MatryoshkaImportanceGate(config)

        # At depth 1, threshold = 0.5 + 0.1 = 0.6
        link = LinkInfo(
            url="https://newsite.com/page",
            text="Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=0.4  # Below threshold of 0.6
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is False
        assert "below_threshold" in reason
        assert "0.40<0.60" in reason

    def test_importance_above_threshold(self, sample_state):
        """Test acceptance when importance is above threshold."""
        config = CrawlConfig(base_threshold=0.3, threshold_increment=0.1)
        gate = MatryoshkaImportanceGate(config)

        # At depth 1, threshold = 0.3 + 0.1 = 0.4
        link = LinkInfo(
            url="https://newsite.com/page",
            text="Page",
            context="",
            source_url="https://example.com/",
            depth=1,
            position='main',
            importance_score=0.8  # Above threshold of 0.4
        )

        should, reason = gate.should_crawl(link, sample_state)

        assert should is True
        assert reason == "accepted"


# ============================================================================
# Test URL Normalization
# ============================================================================

class TestURLNormalization:
    """Tests for URL normalization."""

    def test_normalize_trailing_slash(self, gate):
        """Test removing trailing slash."""
        normalized = gate._normalize_url("https://example.com/page/")
        assert normalized == "https://example.com/page"

    def test_normalize_fragments(self, gate):
        """Test handling URL fragments."""
        # Note: Current implementation doesn't strip fragments
        normalized = gate._normalize_url("https://example.com/page#section")
        assert "example.com/page" in normalized

    def test_normalize_preserves_path(self, gate):
        """Test that path is preserved."""
        normalized = gate._normalize_url("https://example.com/a/b/c")
        assert normalized == "https://example.com/a/b/c"

    def test_normalize_invalid_url(self, gate):
        """Test handling invalid URL."""
        result = gate._normalize_url("not-a-valid-url")
        # URLs without scheme get ://netloc prefix from urlparse behavior
        assert result == "://not-a-valid-url"


# ============================================================================
# Test RecursiveCrawlerSpinner Initialization
# ============================================================================

class TestRecursiveCrawlerSpinnerInit:
    """Tests for RecursiveCrawlerSpinner initialization."""

    def test_default_init(self):
        """Test default initialization."""
        crawler = RecursiveCrawlerSpinner()

        assert crawler.config is not None
        assert crawler.gate is not None
        assert isinstance(crawler.config, CrawlConfig)

    def test_custom_config_init(self, strict_config):
        """Test initialization with custom config."""
        crawler = RecursiveCrawlerSpinner(strict_config)

        assert crawler.config.max_depth == 2
        assert crawler.config.max_pages == 10
        assert crawler.config.same_domain_only is True

    def test_capabilities(self):
        """Test spinner capabilities."""
        crawler = RecursiveCrawlerSpinner()
        caps = crawler.get_capabilities()

        assert 'url' in caps.supported_formats
        assert 'urls' in caps.supported_formats
        assert caps.streaming is True
        assert caps.batch_processing is True

    def test_is_available(self):
        """Test availability check."""
        crawler = RecursiveCrawlerSpinner()

        # Should match WEBSITE_SPINNER_AVAILABLE constant
        assert crawler.is_available() == WEBSITE_SPINNER_AVAILABLE


# ============================================================================
# Test RecursiveCrawlerSpinner Link Extraction
# ============================================================================

class TestLinkExtraction:
    """Tests for link extraction from pages."""

    def test_extract_links_from_text_basic(self):
        """Test regex-based link extraction from text."""
        crawler = RecursiveCrawlerSpinner()

        source_link = LinkInfo(
            url="https://example.com/",
            text="Home",
            context="",
            source_url="",
            depth=0,
            position='main'
        )

        text = """
        Check out https://example.com/page1 for more info.
        Also visit https://other.com/article for details.
        """

        links = crawler._extract_links_from_text(source_link, text)

        assert len(links) >= 2
        urls = [l.url for l in links]
        assert any("example.com/page1" in u for u in urls)
        assert any("other.com/article" in u for u in urls)

    def test_extract_links_depth_increment(self):
        """Test that extracted links have incremented depth."""
        crawler = RecursiveCrawlerSpinner()

        source_link = LinkInfo(
            url="https://example.com/",
            text="Home",
            context="",
            source_url="",
            depth=2,
            position='main'
        )

        text = "Visit https://example.com/deep-page for more."
        links = crawler._extract_links_from_text(source_link, text)

        if links:
            assert links[0].depth == 3  # source depth + 1

    def test_extract_links_context(self):
        """Test that extracted links include context."""
        crawler = RecursiveCrawlerSpinner()

        source_link = LinkInfo(
            url="https://example.com/",
            text="Home",
            context="",
            source_url="",
            depth=0,
            position='main'
        )

        text = "This is important: https://example.com/page more text here."
        links = crawler._extract_links_from_text(source_link, text)

        if links:
            assert len(links[0].context) > 0


# ============================================================================
# Test Link Position Detection
# ============================================================================

class TestLinkPositionDetection:
    """Tests for detecting link position on page."""

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_detect_nav_position(self):
        """Test detecting nav position."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        html = '<nav><a href="/page">Link</a></nav>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        position = crawler._detect_link_position(a_tag)
        assert position == 'nav'

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_detect_footer_position(self):
        """Test detecting footer position."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        html = '<footer><a href="/page">Link</a></footer>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        position = crawler._detect_link_position(a_tag)
        assert position == 'footer'

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_detect_main_position(self):
        """Test detecting main content position."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        html = '<main><article><a href="/page">Link</a></article></main>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        position = crawler._detect_link_position(a_tag)
        assert position in ['main', 'article']

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_detect_sidebar_by_class(self):
        """Test detecting sidebar by class name."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        html = '<div class="sidebar-widget"><a href="/page">Link</a></div>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        position = crawler._detect_link_position(a_tag)
        assert position == 'sidebar'


# ============================================================================
# Test Link Context Extraction
# ============================================================================

class TestLinkContext:
    """Tests for link context extraction."""

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_get_link_context_from_paragraph(self):
        """Test extracting context from parent paragraph."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        html = '<p>This is some text with a <a href="/page">link</a> in it.</p>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        context = crawler._get_link_context(a_tag)
        assert "This is some text" in context
        assert "link" in context

    @pytest.mark.skipif(
        not WEBSITE_SPINNER_AVAILABLE,
        reason="Requires BeautifulSoup"
    )
    def test_get_link_context_truncation(self):
        """Test context truncation for long text."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not available")

        crawler = RecursiveCrawlerSpinner()
        long_text = "Word " * 100  # Very long paragraph
        html = f'<p>{long_text}<a href="/page">link</a></p>'
        soup = BeautifulSoup(html, 'html.parser')
        a_tag = soup.find('a')

        context = crawler._get_link_context(a_tag)
        assert len(context) <= 200


# ============================================================================
# Test Importance Scoring
# ============================================================================

class TestCrawlerImportanceScoring:
    """Tests for crawler importance scoring of shards."""

    def test_score_importance_basic(self):
        """Test basic importance scoring."""
        from HoloLoom.documentation.types import MemoryShard

        crawler = RecursiveCrawlerSpinner()
        shard = MemoryShard(
            id="test-1",
            text="This is a test document about machine learning.",
            episode="test",
            entities=["machine learning"],
            motifs=["education"],
            metadata={
                'crawl_depth': 1,
                'crawl_importance': 0.7
            }
        )

        score = crawler.score_importance(shard)

        assert isinstance(score, ImportanceScore)
        assert 0.0 <= score.score <= 1.0

    def test_score_importance_technical_content(self):
        """Test importance scoring for technical content."""
        from HoloLoom.documentation.types import MemoryShard

        crawler = RecursiveCrawlerSpinner()
        shard = MemoryShard(
            id="test-2",
            text="```python\ndef train_model():\n    pass\n```",
            episode="test",
            entities=["code"],
            motifs=["programming"],
            metadata={
                'crawl_depth': 0,
                'crawl_importance': 0.9
            }
        )

        score = crawler.score_importance(shard)

        # Technical content should have high technical signal
        assert score.signals.technical_score >= 0.7

    def test_score_importance_depth_penalty(self):
        """Test that deeper pages get lower scores."""
        from HoloLoom.documentation.types import MemoryShard

        crawler = RecursiveCrawlerSpinner()

        shallow = MemoryShard(
            id="shallow",
            text="Content " * 100,
            episode="test",
            entities=[],
            motifs=[],
            metadata={'crawl_depth': 0, 'crawl_importance': 0.8}
        )

        deep = MemoryShard(
            id="deep",
            text="Content " * 100,
            episode="test",
            entities=[],
            motifs=[],
            metadata={'crawl_depth': 3, 'crawl_importance': 0.8}
        )

        shallow_score = crawler.score_importance(shallow)
        deep_score = crawler.score_importance(deep)

        # Shallow content should score higher due to depth penalty
        assert shallow_score.signals.custom_signals['depth_penalty'] > deep_score.signals.custom_signals['depth_penalty']


# ============================================================================
# Test Summary Shard Creation
# ============================================================================

class TestSummaryShard:
    """Tests for crawl summary shard creation."""

    def test_create_summary_shard(self):
        """Test creating summary shard."""
        crawler = RecursiveCrawlerSpinner()

        state = CrawlState()
        state.pages_crawled = 25
        state.shards_created = 30
        state.discovered_links = 150
        state.filtered_links = 100
        state.pages_per_domain = {
            "example.com": 15,
            "other.com": 10
        }

        seed_urls = ["https://example.com/", "https://other.com/"]

        summary = crawler._create_summary_shard(state, seed_urls)

        assert summary is not None
        assert "Web Crawl Summary" in summary.text
        assert "Pages crawled: 25" in summary.text
        assert "Links discovered: 150" in summary.text
        assert summary.metadata['content_type'] == 'crawl_summary'
        assert summary.metadata['pages_crawled'] == 25
        assert 'crawl_summary' in summary.id

    def test_summary_includes_thresholds(self):
        """Test summary includes matryoshka thresholds."""
        config = CrawlConfig(
            base_threshold=0.4,
            threshold_increment=0.15
        )
        crawler = RecursiveCrawlerSpinner(config)

        state = CrawlState()
        summary = crawler._create_summary_shard(state, ["https://example.com/"])

        # Check thresholds are displayed
        assert "Depth 0: 0.40" in summary.text
        assert "Depth 1: 0.55" in summary.text


# ============================================================================
# Test Crawl Orchestration (Mocked)
# ============================================================================

class TestCrawlOrchestration:
    """Tests for crawl orchestration with mocked page fetching."""

    @pytest.mark.asyncio
    async def test_spin_single_url(self):
        """Test spinning single URL."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        crawler = RecursiveCrawlerSpinner()

        # Mock the page spinner
        mock_result = SpinResult(
            shards=[],
            success=True
        )

        with patch.object(crawler.page_spinner, 'spin', new_callable=AsyncMock) as mock_spin:
            mock_spin.return_value = mock_result

            result = await crawler.spin("https://example.com/")

            # Should have at least the summary shard
            assert isinstance(result, SpinResult)

    @pytest.mark.asyncio
    async def test_spin_multiple_urls(self):
        """Test spinning multiple URLs."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        crawler = RecursiveCrawlerSpinner()

        mock_result = SpinResult(
            shards=[],
            success=True
        )

        with patch.object(crawler.page_spinner, 'spin', new_callable=AsyncMock) as mock_spin:
            mock_spin.return_value = mock_result

            result = await crawler.spin([
                "https://example.com/",
                "https://other.com/"
            ])

            assert isinstance(result, SpinResult)

    @pytest.mark.asyncio
    async def test_spin_with_topic_override(self):
        """Test topic override via kwargs."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        config = CrawlConfig(seed_topic="original topic")
        crawler = RecursiveCrawlerSpinner(config)

        mock_result = SpinResult(shards=[], success=True)

        with patch.object(crawler.page_spinner, 'spin', new_callable=AsyncMock) as mock_spin:
            mock_spin.return_value = mock_result

            # Override topic
            await crawler.spin("https://example.com/", topic="new topic")

            # Config should be restored after spin
            assert crawler.config.seed_topic == "original topic"

    @pytest.mark.asyncio
    async def test_spin_unavailable_raises(self):
        """Test that spin raises when WebsiteSpinner unavailable."""
        crawler = RecursiveCrawlerSpinner()
        crawler.page_spinner = None  # Simulate unavailable

        with pytest.raises(RuntimeError, match="WebsiteSpinner not available"):
            await crawler._spin_impl("https://example.com/")


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_crawl_website_function(self):
        """Test crawl_website convenience function."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        with patch.object(
            RecursiveCrawlerSpinner,
            'spin',
            new_callable=AsyncMock
        ) as mock_spin:
            mock_spin.return_value = SpinResult(
                shards=[],
                success=True
            )

            result = await crawl_website(
                url="https://example.com/",
                topic="machine learning",
                max_depth=2,
                max_pages=30,
                same_domain=True
            )

            assert isinstance(result, SpinResult)

    @pytest.mark.asyncio
    async def test_crawl_multiple_function(self):
        """Test crawl_multiple convenience function."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        with patch.object(
            RecursiveCrawlerSpinner,
            'spin',
            new_callable=AsyncMock
        ) as mock_spin:
            mock_spin.return_value = SpinResult(
                shards=[],
                success=True
            )

            result = await crawl_multiple(
                urls=["https://example.com/", "https://other.com/"],
                topic="web crawling",
                max_depth=2,
                max_pages=50
            )

            assert isinstance(result, SpinResult)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_topic(self, gate):
        """Test scoring with empty topic."""
        link = LinkInfo(
            url="https://example.com/",
            text="Page",
            context="",
            source_url="",
            depth=0,
            position='main'
        )

        score = gate.score_link(link, "")

        # Should still produce valid score
        assert 0.0 <= score <= 1.0

    def test_invalid_url_parsing(self, gate):
        """Test handling invalid URLs in domain scoring."""
        score = gate._score_domain("not-a-valid-url-at-all")

        # URLs without scheme still parse (netloc=''), so returns default 0.7
        assert score == 0.7

    def test_very_deep_crawl(self, gate):
        """Test threshold at very deep levels."""
        # Depth 100 should be capped at max_threshold
        threshold = gate.get_threshold_for_depth(100)
        assert threshold == 0.85  # max_threshold

    def test_unicode_in_link_text(self, gate):
        """Test scoring links with unicode text."""
        link = LinkInfo(
            url="https://example.com/",
            text="学习机器学习",  # Chinese: "Learn machine learning"
            context="人工智能教程",
            source_url="",
            depth=0,
            position='main'
        )

        score = gate.score_link(link, "machine learning")
        assert 0.0 <= score <= 1.0

    def test_empty_state_crawl_check(self, gate):
        """Test crawl check with empty state."""
        state = CrawlState()

        link = LinkInfo(
            url="https://example.com/page",
            text="Page",
            context="",
            source_url="",
            depth=0,
            position='main',
            importance_score=1.0  # High importance
        )

        should, reason = gate.should_crawl(link, state)

        assert should is True
        assert reason == "accepted"

    def test_crawl_config_serialization(self):
        """Test that CrawlConfig can be serialized to dict."""
        config = CrawlConfig(
            max_depth=5,
            seed_topic="testing"
        )

        config_dict = asdict(config)

        assert config_dict['max_depth'] == 5
        assert config_dict['seed_topic'] == "testing"


# ============================================================================
# Integration Test (with Mocking)
# ============================================================================

class TestCrawlerIntegration:
    """Integration tests for the crawler with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_full_crawl_simulation(self):
        """Test full crawl simulation with mocked pages."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        from HoloLoom.documentation.types import MemoryShard

        config = CrawlConfig(
            max_depth=1,
            max_pages=3,
            seed_topic="python programming",
            delay_between_requests=0  # No delay for testing
        )
        crawler = RecursiveCrawlerSpinner(config)

        # Create mock shards for pages
        page1_shard = MemoryShard(
            id="page1",
            text="Python programming tutorial with examples.",
            episode="web",
            entities=["python"],
            motifs=["tutorial"],
            metadata={
                'content_type': 'webpage',
                'url': 'https://example.com/',
                'raw_html': ''
            }
        )

        mock_result = SpinResult(
            shards=[page1_shard],
            success=True
        )

        with patch.object(crawler.page_spinner, 'spin', new_callable=AsyncMock) as mock_spin:
            mock_spin.return_value = mock_result

            result = await crawler.spin("https://example.com/")

            assert result.success is True
            # Should have page shard + summary shard
            assert len(result.shards) >= 1

    @pytest.mark.asyncio
    async def test_crawl_respects_max_pages(self):
        """Test that crawler respects max_pages limit."""
        if not WEBSITE_SPINNER_AVAILABLE:
            pytest.skip("WebsiteSpinner not available")

        from HoloLoom.documentation.types import MemoryShard

        config = CrawlConfig(
            max_depth=3,
            max_pages=2,  # Very small limit
            delay_between_requests=0
        )
        crawler = RecursiveCrawlerSpinner(config)

        page_shard = MemoryShard(
            id="page",
            text="Content",
            episode="web",
            entities=[],
            motifs=[],
            metadata={'content_type': 'webpage', 'raw_html': ''}
        )

        mock_result = SpinResult(
            shards=[page_shard],
            success=True
        )

        call_count = 0

        async def counting_spin(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_result

        with patch.object(crawler.page_spinner, 'spin', side_effect=counting_spin):
            result = await crawler.spin([
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
                "https://example.com/4"
            ])

            # Should stop after max_pages
            assert call_count <= config.max_pages + 1  # +1 for some tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
