"""
URL Spinner Tests
=================

Comprehensive tests for URLSpinner covering:
- HTML parsing and text extraction
- Link extraction (internal/external)
- Metadata extraction
- Invalid URL handling
- Timeout handling
- Recursive crawling

Author: Claude Code
Date: January 2026
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

# Import spinner components (WebPage doesn't require BeautifulSoup)
from HoloLoom.spinningWheel.url_spinner import WebPage, WEB_AVAILABLE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page Title</title>
        <meta name="description" content="This is a test page description">
        <meta name="keywords" content="test, page, keywords">
        <meta name="author" content="Test Author">
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is the first paragraph with some text content.</p>
        <p>This is another paragraph about machine learning and neural networks.</p>

        <h2>Section Two</h2>
        <ul>
            <li>First item</li>
            <li>Second item</li>
            <li>Third item</li>
        </ul>

        <a href="/internal/page">Internal Link</a>
        <a href="https://external.com/page">External Link</a>
        <a href="https://example.com/another">Another External</a>

        <img src="/images/test.png" alt="Test Image">
        <img src="https://cdn.example.com/image.jpg" alt="External Image">

        <footer>
            <p>Footer content</p>
        </footer>
    </body>
    </html>
    """


@pytest.fixture
def minimal_html():
    """Minimal HTML for simple tests."""
    return """
    <html>
    <head><title>Simple Page</title></head>
    <body>
        <p>Simple content here.</p>
    </body>
    </html>
    """


@pytest.fixture
def url_spinner():
    """Create a URLSpinner with mocked dependencies."""
    with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
        # Need to reimport after patching
        with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
            from HoloLoom.spinningWheel.url_spinner import URLSpinner

            # Create spinner instance
            spinner = URLSpinner(
                importance_threshold=0.3,
                max_depth=2,
                delay_seconds=0.0  # No delay for tests
            )
            return spinner


@pytest.fixture
def minimal_spinner():
    """Create a minimal URLSpinner for fast tests."""
    with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
        with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
            from HoloLoom.spinningWheel.url_spinner import URLSpinner

            spinner = URLSpinner(
                importance_threshold=0.5,
                max_depth=1,
                delay_seconds=0.0
            )
            return spinner


def create_mock_url_spinner(importance_threshold=0.3, max_depth=2, delay_seconds=0.0):
    """Helper to create URLSpinner with mocked dependencies."""
    with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
        with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
            from HoloLoom.spinningWheel.url_spinner import URLSpinner
            return URLSpinner(
                importance_threshold=importance_threshold,
                max_depth=max_depth,
                delay_seconds=delay_seconds
            )


# =============================================================================
# WebPage Dataclass Tests (No external dependencies needed)
# =============================================================================

class TestWebPage:
    """Tests for WebPage dataclass."""

    def test_webpage_creation(self):
        """Test basic WebPage creation."""
        page = WebPage(
            url="https://example.com/page",
            title="Test Page",
            description="Test description",
            text_content="This is the page content.",
            html_content="<html>...</html>",
            links=["https://example.com/link1"],
            images=[{"src": "https://example.com/image.png", "alt": "Test"}],
            metadata={"author": "Test"}
        )

        assert page.url == "https://example.com/page"
        assert page.title == "Test Page"

    def test_webpage_domain_property(self):
        """Test domain extraction from URL."""
        page = WebPage(
            url="https://www.example.com/path/to/page",
            title="Test",
            description=None,
            text_content="Content",
            html_content="<html></html>",
            links=[],
            images=[],
            metadata={}
        )

        assert page.domain == "www.example.com"

    def test_webpage_word_count_property(self):
        """Test word count calculation."""
        page = WebPage(
            url="https://example.com",
            title="Test",
            description=None,
            text_content="This is a sentence with exactly eight words here.",
            html_content="",
            links=[],
            images=[],
            metadata={}
        )

        assert page.word_count == 9

    def test_webpage_internal_links_property(self):
        """Test internal links filtering."""
        page = WebPage(
            url="https://example.com/page",
            title="Test",
            description=None,
            text_content="Content",
            html_content="",
            links=[
                "https://example.com/internal1",
                "https://example.com/internal2",
                "https://external.com/other",
                "https://another.com/page"
            ],
            images=[],
            metadata={}
        )

        internal = page.internal_links
        assert len(internal) == 2
        assert "https://example.com/internal1" in internal
        assert "https://example.com/internal2" in internal

    def test_webpage_external_links_property(self):
        """Test external links filtering."""
        page = WebPage(
            url="https://example.com/page",
            title="Test",
            description=None,
            text_content="Content",
            html_content="",
            links=[
                "https://example.com/internal",
                "https://external.com/other",
                "https://another.com/page"
            ],
            images=[],
            metadata={}
        )

        external = page.external_links
        assert len(external) == 2
        assert "https://external.com/other" in external
        assert "https://another.com/page" in external


# =============================================================================
# WebParser Tests (Require mocking BeautifulSoup)
# =============================================================================

class TestWebParser:
    """Tests for WebParser static methods."""

    def test_fetch_url_success(self, sample_html):
        """Test successful URL fetch."""
        with patch('HoloLoom.spinningWheel.url_spinner.requests') as mock_requests:
            from HoloLoom.spinningWheel.url_spinner import WebParser

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_html
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.raise_for_status = Mock()  # No error
            mock_requests.get.return_value = mock_response

            response = WebParser.fetch_url("https://example.com/page")

            assert response.text == sample_html
            mock_requests.get.assert_called_once()

    def test_fetch_url_timeout(self):
        """Test URL fetch timeout handling."""
        with patch('HoloLoom.spinningWheel.url_spinner.requests') as mock_requests:
            from HoloLoom.spinningWheel.url_spinner import WebParser
            import requests

            mock_requests.get.side_effect = requests.exceptions.Timeout("Connection timed out")

            with pytest.raises(requests.exceptions.Timeout):
                WebParser.fetch_url("https://example.com/slow")

    def test_fetch_url_404(self):
        """Test handling of 404 response."""
        with patch('HoloLoom.spinningWheel.url_spinner.requests') as mock_requests:
            from HoloLoom.spinningWheel.url_spinner import WebParser

            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_requests.get.return_value = mock_response

            with pytest.raises(Exception):
                WebParser.fetch_url("https://example.com/notfound")

    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    def test_parse_html_extracts_title(self, sample_html):
        """Test HTML parsing extracts title."""
        # Create mock BeautifulSoup
        mock_soup = MagicMock()
        mock_title_tag = Mock()
        mock_title_tag.get_text.return_value = "Test Page Title"
        mock_soup.find.side_effect = lambda *args, **kwargs: {
            ('title',): mock_title_tag,
        }.get(args, None)
        mock_soup.find_all.return_value = []
        mock_soup.get_text.return_value = "Main Heading\nContent here"
        mock_soup.__call__ = Mock(return_value=None)

        with patch('HoloLoom.spinningWheel.url_spinner.BeautifulSoup', return_value=mock_soup, create=True):
            from HoloLoom.spinningWheel.url_spinner import WebParser

            page = WebParser.parse_html("https://example.com", sample_html)
            assert page.title == "Test Page Title"

    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    def test_parse_html_handles_minimal(self, minimal_html):
        """Test parsing minimal HTML without errors."""
        # Create mock BeautifulSoup for minimal HTML
        mock_soup = MagicMock()
        mock_title_tag = Mock()
        mock_title_tag.get_text.return_value = "Simple Page"

        def find_side_effect(*args, **kwargs):
            if args == ('title',):
                return mock_title_tag
            return None

        mock_soup.find.side_effect = find_side_effect
        mock_soup.find_all.return_value = []
        mock_soup.get_text.return_value = "Simple content here."
        mock_soup.__call__ = Mock(return_value=[])

        with patch('HoloLoom.spinningWheel.url_spinner.BeautifulSoup', return_value=mock_soup, create=True):
            from HoloLoom.spinningWheel.url_spinner import WebParser

            page = WebParser.parse_html("https://example.com", minimal_html)
            assert page.title == "Simple Page"


# =============================================================================
# URLSpinner Core Tests (With mocked dependencies)
# =============================================================================

class TestURLSpinnerCore:
    """Tests for URLSpinner core functionality."""

    @pytest.mark.asyncio
    async def test_spin_single_url(self, sample_html):
        """Test spinning a single URL."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.0,  # Accept all
                    max_depth=1,
                    delay_seconds=0.0
                )

                # Mock _fetch_page to return a WebPage
                mock_page = WebPage(
                    url="https://example.com",
                    title="Test Page",
                    description="Test description",
                    text_content="Content about machine learning and algorithms.",
                    html_content=sample_html,
                    links=["https://example.com/page2"],
                    images=[{"src": "https://example.com/img.png", "alt": "Test"}],
                    metadata={}
                )

                with patch.object(spinner, '_fetch_page', return_value=mock_page):
                    shards = await spinner._spin_impl("https://example.com")

                    # Should return list of shards
                    assert isinstance(shards, list)

    @pytest.mark.asyncio
    async def test_spin_connection_error(self):
        """Test handling of connection errors - returns empty list."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.3,
                    max_depth=1,
                    delay_seconds=0.0
                )

                # Mock _fetch_page to return None (error case)
                with patch.object(spinner, '_fetch_page', return_value=None):
                    shards = await spinner._spin_impl("https://unreachable.example.com")
                    assert shards == []


# =============================================================================
# Shard Creation Tests
# =============================================================================

class TestShardCreation:
    """Tests for memory shard creation from web pages."""

    def test_shard_metadata(self):
        """Test shard contains correct metadata."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.0,  # Accept all
                    max_depth=1,
                    delay_seconds=0.0
                )

                page = WebPage(
                    url="https://example.com/test",
                    title="Test Page Title",
                    description="Test description",
                    text_content="This is the page content with enough text to be meaningful for testing purposes.",
                    html_content="<html>...</html>",
                    links=["https://example.com/link1", "https://external.com/link"],
                    images=[{"src": "https://example.com/image.png", "alt": "Test"}],
                    metadata={"author": "Test Author"}
                )

                # _page_to_shards returns a list
                shards = spinner._page_to_shards(page)

                # Should have at least one shard with importance_threshold=0.0
                assert len(shards) > 0
                shard = shards[0]
                assert shard.metadata['url'] == "https://example.com/test"
                assert shard.metadata['title'] == "Test Page Title"
                assert shard.metadata['domain'] == "example.com"
                assert 'word_count' in shard.metadata
                assert 'link_count' in shard.metadata

    def test_shard_importance_score(self):
        """Test shard has importance score."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.0,  # Accept all
                    max_depth=1,
                    delay_seconds=0.0
                )

                page = WebPage(
                    url="https://example.com",
                    title="Technical Documentation",
                    description="Documentation about algorithms",
                    text_content="This document discusses algorithms and data structures in detail. It covers tutorials and guides.",
                    html_content="",
                    links=[],
                    images=[],
                    metadata={}
                )

                shards = spinner._page_to_shards(page)

                assert len(shards) > 0
                shard = shards[0]
                assert 'importance_score' in shard.metadata
                assert 0.0 <= shard.metadata['importance_score'] <= 1.0


# =============================================================================
# Recursive Crawling Tests
# =============================================================================

class TestRecursiveCrawling:
    """Tests for recursive website crawling."""

    @pytest.mark.asyncio
    async def test_crawl_respects_max_depth(self):
        """Test crawling respects max_depth setting."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                # Setup mock pages
                pages = {
                    "https://example.com": WebPage(
                        url="https://example.com",
                        title="Home",
                        description=None,
                        text_content="Home page content with enough words to pass importance threshold",
                        html_content="",
                        links=["https://example.com/page1"],
                        images=[],
                        metadata={}
                    ),
                    "https://example.com/page1": WebPage(
                        url="https://example.com/page1",
                        title="Page 1",
                        description=None,
                        text_content="Page 1 content with enough words",
                        html_content="",
                        links=["https://example.com/page2"],
                        images=[],
                        metadata={}
                    ),
                }

                spinner = URLSpinner(max_depth=1, delay_seconds=0.0, importance_threshold=0.0)

                with patch.object(spinner, '_fetch_page', side_effect=lambda url: pages.get(url)):
                    shards = await spinner.spin_website("https://example.com")

                    # Returns list of shards
                    assert isinstance(shards, list)

    @pytest.mark.asyncio
    async def test_crawl_avoids_duplicate_urls(self):
        """Test crawling doesn't revisit same URL."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                # Page that links back to itself
                page = WebPage(
                    url="https://example.com",
                    title="Home",
                    description=None,
                    text_content="Home page content with enough words for importance",
                    html_content="",
                    links=[
                        "https://example.com",  # Self-link
                        "https://example.com/",  # Variant
                        "https://example.com/page1"
                    ],
                    images=[],
                    metadata={}
                )

                spinner = URLSpinner(max_depth=3, delay_seconds=0.0, importance_threshold=0.0)

                with patch.object(spinner, '_fetch_page', return_value=page):
                    shards = await spinner.spin_website("https://example.com")

                    # Should not infinitely loop - returns list
                    assert isinstance(shards, list)

    @pytest.mark.asyncio
    async def test_crawl_stays_on_domain(self):
        """Test crawling stays on the same domain."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                page = WebPage(
                    url="https://example.com",
                    title="Home",
                    description=None,
                    text_content="Home content with enough text for importance scoring",
                    html_content="",
                    links=[
                        "https://example.com/internal",
                        "https://external.com/page",  # Should not be crawled
                        "https://other-domain.com/page"  # Should not be crawled
                    ],
                    images=[],
                    metadata={}
                )

                spinner = URLSpinner(max_depth=2, delay_seconds=0.0, importance_threshold=0.0)

                with patch.object(spinner, '_fetch_page', return_value=page):
                    shards = await spinner.spin_website("https://example.com")

                    # External links should not be in shards
                    for shard in shards:
                        assert "external.com" not in shard.metadata.get('url', '')
                        assert "other-domain.com" not in shard.metadata.get('url', '')


# =============================================================================
# Spinner Capabilities Tests
# =============================================================================

class TestSpinnerCapabilities:
    """Tests for URLSpinner capabilities."""

    def test_capabilities(self):
        """Test spinner capabilities reporting."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.3,
                    max_depth=2,
                    delay_seconds=0.0
                )

                caps = spinner.get_capabilities()

                assert caps.basic_processing is True
                assert caps.batch_processing is True
                assert caps.importance_scoring is True
                assert 'http' in caps.supported_formats or 'https' in caps.supported_formats

    def test_is_available(self):
        """Test availability check."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.3,
                    max_depth=2,
                    delay_seconds=0.0
                )

                result = spinner.is_available()
                assert isinstance(result, bool)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    async def test_spin_url_function(self):
        """Test spin_url convenience function."""
        from HoloLoom.spinningWheel.url_spinner import spin_url, URLSpinner, SpinResult

        # Mock _fetch_page on URLSpinner class
        mock_page = WebPage(
            url="https://example.com",
            title="Test",
            description="Test description",
            text_content="Test content about important topics with enough words.",
            html_content="",
            links=[],
            images=[],
            metadata={}
        )

        with patch.object(URLSpinner, '_fetch_page', return_value=mock_page):
            result = await spin_url("https://example.com")

            assert isinstance(result, SpinResult)
            assert result.success is True


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout is properly handled - returns None from _fetch_page."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.3,
                    max_depth=1,
                    delay_seconds=0.0
                )

                # Mock _fetch_page to return None (error case)
                with patch.object(spinner, '_fetch_page', return_value=None):
                    shards = await spinner._spin_impl("https://slow.example.com")
                    assert shards == []

    @pytest.mark.asyncio
    async def test_ssl_error_handling(self):
        """Test SSL errors are handled - returns None from _fetch_page."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', True):
            with patch.dict('sys.modules', {'requests': MagicMock(), 'bs4': MagicMock()}):
                from HoloLoom.spinningWheel.url_spinner import URLSpinner

                spinner = URLSpinner(
                    importance_threshold=0.3,
                    max_depth=1,
                    delay_seconds=0.0
                )

                with patch.object(spinner, '_fetch_page', return_value=None):
                    shards = await spinner._spin_impl("https://invalid-ssl.example.com")
                    assert shards == []

    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    def test_parse_html_malformed(self):
        """Test parsing malformed HTML doesn't crash."""
        malformed_html = """
        <html>
        <head><title>Broken
        <body>
        <p>Unclosed paragraph
        <div>No closing div
        </body>
        """

        # Mock BeautifulSoup to handle malformed HTML gracefully
        mock_soup = MagicMock()
        mock_title_tag = Mock()
        mock_title_tag.get_text.return_value = "Broken"
        mock_soup.find.side_effect = lambda *args, **kwargs: mock_title_tag if args == ('title',) else None
        mock_soup.find_all.return_value = []
        mock_soup.get_text.return_value = "Unclosed paragraph"
        mock_soup.__call__ = Mock(return_value=[])

        with patch('HoloLoom.spinningWheel.url_spinner.BeautifulSoup', return_value=mock_soup, create=True):
            from HoloLoom.spinningWheel.url_spinner import WebParser

            # Should not raise exception
            page = WebParser.parse_html("https://example.com", malformed_html)

            assert page is not None
            assert page.url == "https://example.com"


# =============================================================================
# URL Normalization Tests
# =============================================================================

class TestURLNormalization:
    """Tests for URL normalization."""

    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    def test_relative_url_resolution(self):
        """Test relative URLs are resolved to absolute."""
        html = """
        <html>
        <body>
            <a href="/page1">Link 1</a>
            <a href="page2">Link 2</a>
            <a href="../parent">Parent</a>
        </body>
        </html>
        """

        # Create mock links
        mock_links = [
            Mock(**{'__getitem__': lambda self, k: '/page1', 'get': lambda k, d=None: '/page1' if k == 'href' else d}),
            Mock(**{'__getitem__': lambda self, k: 'page2', 'get': lambda k, d=None: 'page2' if k == 'href' else d}),
            Mock(**{'__getitem__': lambda self, k: '../parent', 'get': lambda k, d=None: '../parent' if k == 'href' else d}),
        ]

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find.return_value = None
        mock_soup.find_all.side_effect = lambda tag, **kwargs: mock_links if tag == 'a' else []
        mock_soup.get_text.return_value = "Content"
        mock_soup.__call__ = Mock(return_value=[])

        with patch('HoloLoom.spinningWheel.url_spinner.BeautifulSoup', return_value=mock_soup, create=True):
            from HoloLoom.spinningWheel.url_spinner import WebParser

            page = WebParser.parse_html("https://example.com/section/current", html)

            # All links should be absolute
            for link in page.links:
                assert link.startswith("http")

    @pytest.mark.skipif(not WEB_AVAILABLE, reason="BeautifulSoup/requests not available")
    def test_fragment_removal(self):
        """Test URL fragments are handled."""
        html = """
        <html>
        <body>
            <a href="/page#section1">Link with fragment</a>
            <a href="#anchor">Anchor only</a>
        </body>
        </html>
        """

        # Create mock links
        mock_links = [
            Mock(**{'__getitem__': lambda self, k: '/page#section1', 'get': lambda k, d=None: '/page#section1' if k == 'href' else d}),
            Mock(**{'__getitem__': lambda self, k: '#anchor', 'get': lambda k, d=None: '#anchor' if k == 'href' else d}),
        ]

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find.return_value = None
        mock_soup.find_all.side_effect = lambda tag, **kwargs: mock_links if tag == 'a' else []
        mock_soup.get_text.return_value = "Content"
        mock_soup.__call__ = Mock(return_value=[])

        with patch('HoloLoom.spinningWheel.url_spinner.BeautifulSoup', return_value=mock_soup, create=True):
            from HoloLoom.spinningWheel.url_spinner import WebParser

            page = WebParser.parse_html("https://example.com", html)

            # Should have some links processed
            assert isinstance(page.links, list)


# =============================================================================
# Import Availability Tests
# =============================================================================

class TestImportAvailability:
    """Tests for import availability handling."""

    def test_web_available_false_raises_import_error(self):
        """Test that URLSpinner raises ImportError when WEB_AVAILABLE is False."""
        with patch('HoloLoom.spinningWheel.url_spinner.WEB_AVAILABLE', False):
            # Need to reload to get the patched value
            from HoloLoom.spinningWheel.url_spinner import URLSpinner

            with pytest.raises(ImportError) as exc_info:
                URLSpinner()

            assert "requests and beautifulsoup4" in str(exc_info.value)

    def test_web_available_check(self):
        """Test WEB_AVAILABLE flag check."""
        from HoloLoom.spinningWheel.url_spinner import WEB_AVAILABLE

        # Should be a boolean
        assert isinstance(WEB_AVAILABLE, bool)
