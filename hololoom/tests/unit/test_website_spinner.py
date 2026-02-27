"""
Tests for WebsiteSpinner - Multimodal Web Scraping
===================================================

Comprehensive tests for the website spinner including:
- ExtractedImage dataclass
- WebPageContent dataclass
- ImageExtractor filtering and context extraction
- WebsiteSpinner initialization and capabilities
- Importance scoring methods
- Mocked HTTP response tests

Author: Claude Code
Date: December 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from dataclasses import asdict

# Import test targets
from hololoom.spinningWheel.website import (
    ExtractedImage,
    WebPageContent,
    ImageExtractor,
    WebsiteSpinner,
    scrape_webpage,
    spin_website,
    REQUESTS_AVAILABLE,
    BS4_AVAILABLE
)
from hololoom.spinningWheel.protocol import (
    SpinnerCapabilities,
    ImportanceSignals,
    ImportanceScore
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample ExtractedImage."""
    return ExtractedImage(
        url="https://example.com/image.jpg",
        alt_text="A beautiful sunset",
        caption="Sunset over the mountains",
        surrounding_text="The view from the peak was breathtaking.",
        width=800,
        height=600,
        context_element="figure"
    )


@pytest.fixture
def sample_image_no_context():
    """Create an image without context."""
    return ExtractedImage(
        url="https://example.com/logo.png",
        alt_text="",
        caption="",
        surrounding_text="",
        width=100,
        height=100
    )


@pytest.fixture
def sample_page_content():
    """Create sample WebPageContent."""
    return WebPageContent(
        url="https://example.com/article",
        title="Test Article Title",
        text_content="This is the main article content. It discusses various topics.",
        images=[],
        links=["https://example.com/link1", "https://example.com/link2"],
        meta_description="A test article for unit testing",
        meta_keywords=["test", "article", "example"],
        headings=["Introduction", "Main Section", "Conclusion"]
    )


@pytest.fixture
def image_extractor():
    """Create an ImageExtractor instance."""
    return ImageExtractor(
        min_width=100,
        min_height=100,
        max_images=20,
        download_images=False
    )


@pytest.fixture
def website_spinner():
    """Create a WebsiteSpinner instance."""
    return WebsiteSpinner(
        importance_threshold=0.3,
        extract_images=True,
        min_image_width=100,
        min_image_height=100,
        max_images_per_page=20,
        timeout=30.0,
        user_agent="TestBot/1.0"
    )


@pytest.fixture
def mock_html_content():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page Title</title>
        <meta name="description" content="Test page description">
        <meta name="keywords" content="test, example, keywords">
    </head>
    <body>
        <header>Navigation here</header>
        <main>
            <h1>Main Heading</h1>
            <h2>Subheading</h2>
            <p>This is the main content of the page.</p>
            <figure>
                <img src="/images/test.jpg" alt="Test Image" width="800" height="600">
                <figcaption>This is a test image caption</figcaption>
            </figure>
            <p>More content follows the image.</p>
            <a href="/link1">Link 1</a>
            <a href="https://external.com/link2">Link 2</a>
        </main>
        <footer>Footer content</footer>
    </body>
    </html>
    """


# =============================================================================
# ExtractedImage Tests
# =============================================================================

class TestExtractedImage:
    """Tests for ExtractedImage dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ExtractedImage with minimal args."""
        img = ExtractedImage(url="https://example.com/image.jpg")

        assert img.url == "https://example.com/image.jpg"
        assert img.alt_text == ""
        assert img.caption == ""
        assert img.surrounding_text == ""
        assert img.width is None
        assert img.height is None
        assert img.file_size is None
        assert img.image_data is None
        assert img.context_element == ""

    def test_creation_with_all_fields(self, sample_image):
        """Test creating ExtractedImage with all fields."""
        assert sample_image.url == "https://example.com/image.jpg"
        assert sample_image.alt_text == "A beautiful sunset"
        assert sample_image.caption == "Sunset over the mountains"
        assert sample_image.width == 800
        assert sample_image.height == 600

    def test_has_context_true(self, sample_image):
        """Test has_context returns True when context exists."""
        assert sample_image.has_context is True

    def test_has_context_false(self, sample_image_no_context):
        """Test has_context returns False when no context."""
        assert sample_image_no_context.has_context is False

    def test_has_context_with_only_alt_text(self):
        """Test has_context with only alt text."""
        img = ExtractedImage(
            url="https://example.com/img.jpg",
            alt_text="Some alt text"
        )
        assert img.has_context is True

    def test_has_context_with_only_caption(self):
        """Test has_context with only caption."""
        img = ExtractedImage(
            url="https://example.com/img.jpg",
            caption="A caption"
        )
        assert img.has_context is True

    def test_context_text_full(self, sample_image):
        """Test context_text with all context fields."""
        context = sample_image.context_text

        assert "Caption: Sunset over the mountains" in context
        assert "Alt: A beautiful sunset" in context
        assert "Context: The view from the peak" in context

    def test_context_text_empty(self, sample_image_no_context):
        """Test context_text with no context."""
        assert sample_image_no_context.context_text == ""

    def test_context_text_truncates_surrounding(self):
        """Test that surrounding_text is truncated in context_text."""
        long_text = "x" * 500
        img = ExtractedImage(
            url="https://example.com/img.jpg",
            surrounding_text=long_text
        )
        context = img.context_text

        # Should be truncated to 200 chars
        assert len(context) < 300  # "Context: " prefix + 200 chars


# =============================================================================
# WebPageContent Tests
# =============================================================================

class TestWebPageContent:
    """Tests for WebPageContent dataclass."""

    def test_creation(self, sample_page_content):
        """Test creating WebPageContent."""
        assert sample_page_content.url == "https://example.com/article"
        assert sample_page_content.title == "Test Article Title"
        assert "main article content" in sample_page_content.text_content
        assert len(sample_page_content.links) == 2
        assert len(sample_page_content.headings) == 3

    def test_default_fetch_time(self):
        """Test that fetch_time defaults to now."""
        before = datetime.now()
        page = WebPageContent(
            url="https://example.com",
            title="Test",
            text_content="Content",
            images=[],
            links=[]
        )
        after = datetime.now()

        assert before <= page.fetch_time <= after

    def test_empty_lists_default(self):
        """Test that list fields default to empty."""
        page = WebPageContent(
            url="https://example.com",
            title="Test",
            text_content="Content",
            images=[],
            links=[]
        )

        assert page.meta_keywords == []
        assert page.headings == []


# =============================================================================
# ImageExtractor Tests
# =============================================================================

class TestImageExtractor:
    """Tests for ImageExtractor class."""

    def test_initialization(self, image_extractor):
        """Test ImageExtractor initialization."""
        assert image_extractor.min_width == 100
        assert image_extractor.min_height == 100
        assert image_extractor.max_images == 20
        assert image_extractor.download_images is False

    def test_skip_regex_compiled(self, image_extractor):
        """Test that skip patterns are compiled."""
        assert image_extractor.skip_regex is not None
        assert image_extractor.skip_regex.search("logo.png")
        assert image_extractor.skip_regex.search("tracking-pixel")
        assert image_extractor.skip_regex.search("ad_banner")

    def test_should_skip_logo_url(self, image_extractor):
        """Test skipping logo images by URL."""
        mock_img = Mock()
        mock_img.get.return_value = None

        result = image_extractor._should_skip(mock_img, "https://example.com/logo.png")
        assert result is True

    def test_should_skip_icon_url(self, image_extractor):
        """Test skipping icon images by URL."""
        mock_img = Mock()
        mock_img.get.return_value = None

        result = image_extractor._should_skip(mock_img, "https://example.com/icon-social.png")
        assert result is True

    def test_should_skip_tracking_pixel_url(self, image_extractor):
        """Test skipping tracking pixel by URL."""
        mock_img = Mock()
        mock_img.get.return_value = None

        result = image_extractor._should_skip(mock_img, "https://example.com/tracking_pixel.gif")
        assert result is True

    def test_should_skip_ad_url(self, image_extractor):
        """Test skipping ad images by URL."""
        mock_img = Mock()
        mock_img.get.return_value = None

        result = image_extractor._should_skip(mock_img, "https://example.com/ad_300x250.jpg")
        assert result is True

    def test_should_skip_by_alt_text(self, image_extractor):
        """Test skipping by alt text pattern."""
        mock_img = Mock()
        mock_img.get.side_effect = lambda key, default=None: {
            'alt': 'company logo',
            'class': None,
            'width': None,
            'height': None
        }.get(key, default)

        result = image_extractor._should_skip(mock_img, "https://example.com/image.png")
        assert result is True

    def test_should_skip_by_class(self, image_extractor):
        """Test skipping by class pattern."""
        mock_img = Mock()
        mock_img.get.side_effect = lambda key, default=None: {
            'alt': '',
            'class': ['site-icon', 'nav-image'],
            'width': None,
            'height': None
        }.get(key, default)

        result = image_extractor._should_skip(mock_img, "https://example.com/image.png")
        assert result is True

    def test_should_skip_small_dimensions(self, image_extractor):
        """Test skipping small images."""
        mock_img = Mock()
        mock_img.get.side_effect = lambda key, default=None: {
            'alt': '',
            'class': None,
            'width': '50',
            'height': '50'
        }.get(key, default)

        result = image_extractor._should_skip(mock_img, "https://example.com/image.png")
        assert result is True

    def test_should_skip_tracking_pixel_dimensions(self, image_extractor):
        """Test skipping 1x1 tracking pixels."""
        mock_img = Mock()
        mock_img.get.side_effect = lambda key, default=None: {
            'alt': '',
            'class': None,
            'width': '1',
            'height': '1'
        }.get(key, default)

        result = image_extractor._should_skip(mock_img, "https://example.com/pixel.gif")
        assert result is True

    def test_should_not_skip_valid_image(self, image_extractor):
        """Test not skipping valid content images."""
        mock_img = Mock()
        mock_img.get.side_effect = lambda key, default=None: {
            'alt': 'Product photo',
            'class': ['product-image'],
            'width': '800',
            'height': '600'
        }.get(key, default)

        result = image_extractor._should_skip(mock_img, "https://example.com/product.jpg")
        assert result is False

    def test_parse_dimension_integer(self, image_extractor):
        """Test parsing integer dimensions."""
        assert image_extractor._parse_dimension("800") == 800
        assert image_extractor._parse_dimension(600) == 600

    def test_parse_dimension_with_px(self, image_extractor):
        """Test parsing dimensions with px suffix."""
        assert image_extractor._parse_dimension("800px") == 800
        assert image_extractor._parse_dimension("600px") == 600

    def test_parse_dimension_with_percent(self, image_extractor):
        """Test parsing dimensions with percent."""
        assert image_extractor._parse_dimension("100%") == 100

    def test_parse_dimension_float(self, image_extractor):
        """Test parsing float dimensions."""
        assert image_extractor._parse_dimension("800.5") == 800

    def test_parse_dimension_none(self, image_extractor):
        """Test parsing None dimension."""
        assert image_extractor._parse_dimension(None) is None

    def test_parse_dimension_invalid(self, image_extractor):
        """Test parsing invalid dimension."""
        assert image_extractor._parse_dimension("invalid") is None
        assert image_extractor._parse_dimension("") is None

    @pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
    def test_extract_from_html(self, image_extractor, mock_html_content):
        """Test extracting images from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(mock_html_content, 'html.parser')
        images = image_extractor.extract(soup, "https://example.com")

        # Should find the test image
        assert len(images) >= 1

        # First image should have context
        if images:
            img = images[0]
            assert "https://example.com" in img.url or "/images/test.jpg" in img.url


# =============================================================================
# WebsiteSpinner Tests
# =============================================================================

class TestWebsiteSpinner:
    """Tests for WebsiteSpinner class."""

    def test_initialization(self, website_spinner):
        """Test WebsiteSpinner initialization."""
        assert website_spinner.name == "website"
        assert website_spinner.importance_threshold == 0.3
        assert website_spinner.extract_images is True
        assert website_spinner.timeout == 30.0
        assert website_spinner.user_agent == "TestBot/1.0"

    def test_initialization_defaults(self):
        """Test WebsiteSpinner with default values."""
        spinner = WebsiteSpinner()

        assert spinner.importance_threshold == 0.3
        assert spinner.extract_images is True
        assert spinner.timeout == 30.0

    def test_get_capabilities(self, website_spinner):
        """Test getting spinner capabilities."""
        caps = website_spinner.get_capabilities()

        assert isinstance(caps, SpinnerCapabilities)
        assert caps.basic_processing is True
        assert caps.streaming is False
        assert caps.batch_processing is True
        assert caps.importance_scoring is True
        assert caps.multimodal is True
        assert 'url' in caps.supported_formats
        assert 'html' in caps.supported_formats

    def test_is_available(self, website_spinner):
        """Test checking availability."""
        # Result depends on whether requests and bs4 are installed
        result = website_spinner.is_available()
        expected = REQUESTS_AVAILABLE and BS4_AVAILABLE
        assert result == expected

    def test_score_text_importance(self, website_spinner, sample_page_content):
        """Test text importance scoring."""
        score = website_spinner._score_text_importance(sample_page_content)

        assert isinstance(score, ImportanceScore)
        assert 0 <= score.score <= 1
        assert score.reason is not None

    def test_score_text_importance_edu_domain(self, website_spinner):
        """Test authority score for .edu domain."""
        page = WebPageContent(
            url="https://stanford.edu/article",
            title="Research Paper",
            text_content="Academic content " * 500,
            images=[],
            links=[],
            headings=["Abstract", "Methods", "Results"]
        )

        score = website_spinner._score_text_importance(page)
        # .edu domains should have higher authority
        assert score.signals.authority_score == 0.8

    def test_score_text_importance_gov_domain(self, website_spinner):
        """Test authority score for .gov domain."""
        page = WebPageContent(
            url="https://nih.gov/research",
            title="Government Report",
            text_content="Official content " * 500,
            images=[],
            links=[],
            headings=["Summary"]
        )

        score = website_spinner._score_text_importance(page)
        assert score.signals.authority_score == 0.8

    def test_score_text_importance_tech_domain(self, website_spinner):
        """Test authority score for tech domains."""
        page = WebPageContent(
            url="https://github.com/project",
            title="Open Source Project",
            text_content="Code documentation " * 500,
            images=[],
            links=[],
            headings=["Installation", "Usage"]
        )

        score = website_spinner._score_text_importance(page)
        assert score.signals.authority_score == 0.75

    def test_score_image_importance(self, website_spinner, sample_image):
        """Test image importance scoring."""
        score = website_spinner._score_image_importance(sample_image)

        assert isinstance(score, ImportanceScore)
        assert 0 <= score.score <= 1
        assert score.reason is not None

    def test_score_image_importance_with_caption(self, website_spinner):
        """Test image scoring with caption (strong signal)."""
        img_with_caption = ExtractedImage(
            url="https://example.com/img.jpg",
            caption="Detailed figure caption",
            width=800,
            height=600
        )

        img_without_caption = ExtractedImage(
            url="https://example.com/img2.jpg",
            width=800,
            height=600
        )

        score_with = website_spinner._score_image_importance(img_with_caption)
        score_without = website_spinner._score_image_importance(img_without_caption)

        # Caption should increase score
        assert score_with.signals.technical_score > score_without.signals.technical_score

    def test_score_image_importance_large_image(self, website_spinner):
        """Test image scoring for large images."""
        large_img = ExtractedImage(
            url="https://example.com/large.jpg",
            width=1200,
            height=800
        )

        small_img = ExtractedImage(
            url="https://example.com/small.jpg",
            width=200,
            height=150
        )

        large_score = website_spinner._score_image_importance(large_img)
        small_score = website_spinner._score_image_importance(small_img)

        # Larger image should have higher authority score
        assert large_score.signals.authority_score > small_score.signals.authority_score


# =============================================================================
# Integration Tests (with mocking)
# =============================================================================

class TestWebsiteSpinnerIntegration:
    """Integration tests for WebsiteSpinner with mocked HTTP."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not (REQUESTS_AVAILABLE and BS4_AVAILABLE),
                        reason="requests or bs4 not installed")
    async def test_spin_impl_success(self, website_spinner, mock_html_content):
        """Test successful page scraping."""
        with patch.object(website_spinner, '_fetch_sync', return_value=mock_html_content):
            shards = await website_spinner._spin_impl("https://example.com")

            # Should have at least text shard
            assert len(shards) >= 1

            # First shard should be text content
            text_shard = shards[0]
            assert text_shard.metadata.get('shard_type') == 'page_text'
            assert 'Test Page Title' in text_shard.text or text_shard.metadata.get('title') == 'Test Page Title'

    @pytest.mark.asyncio
    @pytest.mark.skipif(not (REQUESTS_AVAILABLE and BS4_AVAILABLE),
                        reason="requests or bs4 not installed")
    async def test_spin_impl_with_images(self, mock_html_content):
        """Test scraping with image extraction."""
        spinner = WebsiteSpinner(extract_images=True)

        with patch.object(spinner, '_fetch_sync', return_value=mock_html_content):
            shards = await spinner._spin_impl("https://example.com")

            # Check for image shards
            image_shards = [s for s in shards if s.metadata.get('shard_type') == 'page_image']

            # Should have image shards (if images have context)
            # The test HTML has a figure with figcaption

    @pytest.mark.asyncio
    async def test_spin_impl_fetch_failure(self, website_spinner):
        """Test handling of fetch failure."""
        with patch.object(website_spinner, '_fetch_sync', return_value=None):
            shards = await website_spinner._spin_impl("https://invalid.example.com")

            # Should return empty list on failure
            assert shards == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not (REQUESTS_AVAILABLE and BS4_AVAILABLE),
                        reason="requests or bs4 not installed")
    async def test_spin_impl_metadata(self, website_spinner, mock_html_content):
        """Test that metadata is correctly populated."""
        with patch.object(website_spinner, '_fetch_sync', return_value=mock_html_content):
            shards = await website_spinner._spin_impl("https://example.com/page")

            if shards:
                text_shard = shards[0]
                assert 'url' in text_shard.metadata
                assert 'title' in text_shard.metadata
                assert 'importance_score' in text_shard.metadata
                assert 'fetch_time' in text_shard.metadata


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_spin_website_alias(self):
        """Test that spin_website is alias for scrape_webpage."""
        assert spin_website is scrape_webpage

    @pytest.mark.asyncio
    @pytest.mark.skipif(not (REQUESTS_AVAILABLE and BS4_AVAILABLE),
                        reason="requests or bs4 not installed")
    async def test_scrape_webpage(self, mock_html_content):
        """Test scrape_webpage convenience function."""
        with patch('hololoom.spinningWheel.website.WebsiteSpinner._fetch_sync',
                   return_value=mock_html_content):
            result = await scrape_webpage(
                "https://example.com",
                extract_images=True,
                max_images=10
            )

            # Result should be a SpinResult
            assert hasattr(result, 'shards') or hasattr(result, 'success')


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_image_extractor_empty_soup(self, image_extractor):
        """Test extracting from empty HTML."""
        if not BS4_AVAILABLE:
            pytest.skip("BeautifulSoup not installed")

        from bs4 import BeautifulSoup
        soup = BeautifulSoup("", 'html.parser')
        images = image_extractor.extract(soup, "https://example.com")

        assert images == []

    def test_image_extractor_no_images(self, image_extractor):
        """Test extracting from HTML with no images."""
        if not BS4_AVAILABLE:
            pytest.skip("BeautifulSoup not installed")

        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html><body><p>No images here</p></body></html>", 'html.parser')
        images = image_extractor.extract(soup, "https://example.com")

        assert images == []

    def test_image_extractor_duplicate_urls(self, image_extractor):
        """Test that duplicate image URLs are filtered."""
        if not BS4_AVAILABLE:
            pytest.skip("BeautifulSoup not installed")

        from bs4 import BeautifulSoup
        html = """
        <html><body>
            <img src="/image.jpg" alt="First">
            <img src="/image.jpg" alt="Second">
            <img src="/image.jpg" alt="Third">
        </body></html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = image_extractor.extract(soup, "https://example.com")

        # Should only have one unique URL
        urls = [img.url for img in images]
        assert len(urls) == len(set(urls))

    def test_spinner_without_images(self):
        """Test spinner with image extraction disabled."""
        spinner = WebsiteSpinner(extract_images=False)

        assert spinner.extract_images is False

    def test_extracted_image_empty_context(self):
        """Test ExtractedImage with completely empty context."""
        img = ExtractedImage(url="https://example.com/img.jpg")

        assert img.has_context is False
        assert img.context_text == ""

    @pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
    def test_image_extractor_max_images_limit(self, image_extractor):
        """Test that max_images limit is respected."""
        from bs4 import BeautifulSoup

        # Create HTML with many images
        images_html = "".join([
            f'<img src="/img{i}.jpg" alt="Image {i}" width="800" height="600">'
            for i in range(50)
        ])
        html = f"<html><body>{images_html}</body></html>"
        soup = BeautifulSoup(html, 'html.parser')

        extractor = ImageExtractor(max_images=5)
        images = extractor.extract(soup, "https://example.com")

        assert len(images) <= 5


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
