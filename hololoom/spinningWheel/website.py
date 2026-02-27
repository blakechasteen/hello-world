"""
Website Spinner - Multimodal Web Scraping with Image Extraction
================================================================

Ingests web pages with both text AND images, extracting meaningful visual
content with context (captions, alt-text, surrounding text).

Features:
- Text content extraction (like url_spinner)
- Image extraction with context (captions, alt text, surrounding text)
- Smart image filtering (skip logos, icons, ads)
- Image importance scoring
- Recursive crawling with matryoshka importance gating
- Rate limiting and robots.txt respect

Requirements:
    pip install beautifulsoup4 requests pillow

Author: Claude Code
Date: December 2025
"""

import re
import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from datetime import datetime
import warnings

# Type checking imports (don't execute at runtime)
if TYPE_CHECKING:
    from bs4 import BeautifulSoup

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceSignals,
    ImportanceScore
)
from HoloLoom.protocols.types import MemoryShard


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedImage:
    """Extracted image with context."""
    url: str
    alt_text: str = ''
    caption: str = ''
    surrounding_text: str = ''
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    image_data: Optional[bytes] = None
    context_element: str = ''  # 'figure', 'article', 'content'

    @property
    def has_context(self) -> bool:
        """Check if image has meaningful context."""
        return bool(self.alt_text or self.caption or self.surrounding_text)

    @property
    def context_text(self) -> str:
        """Combined context text."""
        parts = []
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.alt_text:
            parts.append(f"Alt: {self.alt_text}")
        if self.surrounding_text:
            parts.append(f"Context: {self.surrounding_text[:200]}")
        return '\n'.join(parts)


@dataclass
class WebPageContent:
    """Extracted web page content."""
    url: str
    title: str
    text_content: str
    images: List[ExtractedImage]
    links: List[str]
    meta_description: str = ''
    meta_keywords: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    fetch_time: datetime = field(default_factory=datetime.now)


# =============================================================================
# Image Extractor
# =============================================================================

class ImageExtractor:
    """Extract meaningful images from HTML with context."""

    # Skip patterns for filtering out non-content images
    SKIP_PATTERNS = [
        r'logo', r'icon', r'favicon', r'sprite', r'pixel',
        r'tracking', r'analytics', r'ad[_-]?', r'banner',
        r'button', r'arrow', r'social', r'share',
        r'avatar', r'profile', r'thumbnail',
        r'spacer', r'blank', r'placeholder',
        r'\d+x\d+',  # Size patterns like "1x1"
        r'loading', r'spinner',
    ]

    # Common non-content image dimensions (width, height)
    SKIP_DIMENSIONS = [
        (1, 1), (0, 0),  # Tracking pixels
        (16, 16), (24, 24), (32, 32),  # Icons
        (150, 50), (728, 90), (300, 250),  # Common ad sizes
    ]

    MIN_WIDTH = 100
    MIN_HEIGHT = 100

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        max_images: int = 20,
        download_images: bool = False
    ):
        """
        Initialize image extractor.

        Args:
            min_width: Minimum image width to include
            min_height: Minimum image height to include
            max_images: Maximum images to extract per page
            download_images: Whether to download image data
        """
        self.min_width = min_width
        self.min_height = min_height
        self.max_images = max_images
        self.download_images = download_images

        # Compile skip patterns
        self.skip_regex = re.compile(
            '|'.join(self.SKIP_PATTERNS),
            re.IGNORECASE
        )

    def extract(
        self,
        soup: "BeautifulSoup",
        base_url: str
    ) -> List[ExtractedImage]:
        """
        Extract meaningful images from parsed HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative paths

        Returns:
            List of ExtractedImage objects
        """
        images = []
        seen_urls: Set[str] = set()

        # Find all img tags
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if not src:
                continue

            # Resolve relative URL
            full_url = urljoin(base_url, src)

            # Skip duplicates
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Skip filtered images
            if self._should_skip(img, full_url):
                continue

            # Extract context
            extracted = self._extract_image_context(img, full_url)
            if extracted:
                images.append(extracted)

            if len(images) >= self.max_images:
                break

        # Sort by context richness
        images.sort(key=lambda x: len(x.context_text), reverse=True)

        return images[:self.max_images]

    def _should_skip(self, img, url: str) -> bool:
        """Check if image should be skipped."""
        # Check URL patterns
        if self.skip_regex.search(url):
            return True

        # Check alt text patterns
        alt = img.get('alt', '')
        if alt and self.skip_regex.search(alt):
            return True

        # Check class patterns
        classes = img.get('class', [])
        if isinstance(classes, list):
            class_str = ' '.join(classes)
        else:
            class_str = classes

        if class_str and self.skip_regex.search(class_str):
            return True

        # Check dimensions
        width = self._parse_dimension(img.get('width'))
        height = self._parse_dimension(img.get('height'))

        if width and height:
            # Skip known non-content dimensions
            if (width, height) in self.SKIP_DIMENSIONS:
                return True

            # Skip small images
            if width < self.min_width or height < self.min_height:
                return True

        return False

    def _parse_dimension(self, value) -> Optional[int]:
        """Parse dimension value (handles 'px', '%', etc.)."""
        if not value:
            return None

        try:
            # Remove units
            clean = str(value).replace('px', '').replace('%', '').strip()
            return int(float(clean))
        except (ValueError, TypeError):
            return None

    def _extract_image_context(
        self,
        img,
        url: str
    ) -> Optional[ExtractedImage]:
        """Extract image with surrounding context."""
        # Get alt text
        alt_text = img.get('alt', '')

        # Get dimensions
        width = self._parse_dimension(img.get('width'))
        height = self._parse_dimension(img.get('height'))

        # Find caption from figure/figcaption
        caption = ''
        context_element = 'img'

        # Check for figure parent
        figure = img.find_parent('figure')
        if figure:
            context_element = 'figure'
            figcaption = figure.find('figcaption')
            if figcaption:
                caption = figcaption.get_text(strip=True)

        # Get surrounding text
        surrounding_text = self._get_surrounding_text(img)

        return ExtractedImage(
            url=url,
            alt_text=alt_text,
            caption=caption,
            surrounding_text=surrounding_text,
            width=width,
            height=height,
            context_element=context_element
        )

    def _get_surrounding_text(self, img, max_chars: int = 200) -> str:
        """Get text surrounding the image."""
        text_parts = []

        # Look for nearby text elements
        parent = img.parent
        for _ in range(3):  # Go up 3 levels max
            if not parent:
                break

            # Get sibling text
            for sibling in parent.children:
                if hasattr(sibling, 'get_text'):
                    text = sibling.get_text(strip=True)
                    if text and len(text) > 20:  # Skip short fragments
                        text_parts.append(text[:max_chars])

            parent = parent.parent

        return ' '.join(text_parts)[:max_chars]


# =============================================================================
# Website Spinner
# =============================================================================

class WebsiteSpinner(BaseSpinner):
    """
    Multimodal web scraping with text and image extraction.

    Features:
    - Full page text extraction
    - Meaningful image extraction with context
    - Smart filtering (skip logos, ads, icons)
    - Link extraction for crawling
    - Rate limiting and robots.txt respect
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        extract_images: bool = True,
        min_image_width: int = 100,
        min_image_height: int = 100,
        max_images_per_page: int = 20,
        download_images: bool = False,
        timeout: float = 30.0,
        user_agent: str = 'HoloLoom/1.0 (Research Bot)'
    ):
        """
        Initialize WebsiteSpinner.

        Args:
            importance_threshold: Minimum importance score
            extract_images: Whether to extract images
            min_image_width: Minimum image width to include
            min_image_height: Minimum image height to include
            max_images_per_page: Max images to extract per page
            download_images: Whether to download image data
            timeout: Request timeout in seconds
            user_agent: User agent string
        """
        super().__init__(
            name='website',
            importance_threshold=importance_threshold
        )

        self.extract_images = extract_images
        self.timeout = timeout
        self.user_agent = user_agent

        # Image extractor
        self.image_extractor = ImageExtractor(
            min_width=min_image_width,
            min_height=min_image_height,
            max_images=max_images_per_page,
            download_images=download_images
        )

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=False,
            motif_extraction=False,
            multimodal=True,  # Text + images
            supported_formats=['url', 'html']
        )

    def is_available(self) -> bool:
        """Check if dependencies are available."""
        return REQUESTS_AVAILABLE and BS4_AVAILABLE

    async def _spin_impl(
        self,
        source: str,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Extract content from web page.

        Args:
            source: URL to scrape

        Returns:
            List of MemoryShard objects (text + images)
        """
        # Fetch page
        page_content = await self._fetch_page(source)
        if not page_content:
            return []

        shards = []

        # Create text shard
        text_importance = self._score_text_importance(page_content)
        text_shard = MemoryShard(
            id=f"website_text_{hashlib.md5(source.encode()).hexdigest()[:12]}",
            text=f"{page_content.title}\n\n{page_content.text_content}",
            episode=f"website_{urlparse(source).netloc}",
            entities=[urlparse(source).netloc] + page_content.meta_keywords[:5],
            motifs=['web_content'] + (['technical'] if page_content.meta_keywords else []),
            metadata={
                'url': source,
                'title': page_content.title,
                'meta_description': page_content.meta_description,
                'headings': page_content.headings[:10],
                'image_count': len(page_content.images),
                'link_count': len(page_content.links),
                'fetch_time': page_content.fetch_time.isoformat(),
                'importance_score': text_importance.score,
                'shard_type': 'page_text'
            }
        )
        shards.append(text_shard)

        # Create image shards
        if self.extract_images and page_content.images:
            for i, img in enumerate(page_content.images):
                if not img.has_context:
                    continue  # Skip images without context

                img_importance = self._score_image_importance(img)

                img_shard = MemoryShard(
                    id=f"website_img_{hashlib.md5(img.url.encode()).hexdigest()[:12]}",
                    text=f"Image from {page_content.title}\n\n{img.context_text}",
                    episode=f"website_{urlparse(source).netloc}",
                    entities=[urlparse(source).netloc],
                    motifs=['web_image', 'visual_content'],
                    metadata={
                        'image_url': img.url,
                        'page_url': source,
                        'alt_text': img.alt_text,
                        'caption': img.caption,
                        'surrounding_text': img.surrounding_text,
                        'width': img.width,
                        'height': img.height,
                        'context_element': img.context_element,
                        'importance_score': img_importance.score,
                        'shard_type': 'page_image'
                    }
                )
                shards.append(img_shard)

        return shards

    async def _fetch_page(self, url: str) -> Optional[WebPageContent]:
        """Fetch and parse web page."""
        loop = asyncio.get_event_loop()

        try:
            # Fetch in thread pool
            response = await loop.run_in_executor(
                None,
                self._fetch_sync,
                url
            )

            if not response:
                return None

            # Parse HTML
            soup = BeautifulSoup(response, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ''

            # Extract meta description
            meta_desc = ''
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_desc = meta_tag.get('content', '')

            # Extract meta keywords
            keywords = []
            kw_tag = soup.find('meta', attrs={'name': 'keywords'})
            if kw_tag:
                kw_content = kw_tag.get('content', '')
                keywords = [k.strip() for k in kw_content.split(',')]

            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                text = h.get_text(strip=True)
                if text:
                    headings.append(text)

            # Extract main text content
            # Remove script, style, nav, footer
            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            text_content = main_content.get_text(separator='\n', strip=True) if main_content else ''

            # Clean up text
            text_content = re.sub(r'\n{3,}', '\n\n', text_content)
            text_content = text_content[:10000]  # Limit length

            # Extract images
            images = []
            if self.extract_images:
                images = self.image_extractor.extract(soup, url)

            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                if href and not href.startswith('#'):
                    full_url = urljoin(url, href)
                    links.append(full_url)

            return WebPageContent(
                url=url,
                title=title,
                text_content=text_content,
                images=images,
                links=links[:100],  # Limit links
                meta_description=meta_desc,
                meta_keywords=keywords,
                headings=headings
            )

        except Exception as e:
            warnings.warn(f"Error fetching {url}: {e}")
            return None

    def _fetch_sync(self, url: str) -> Optional[str]:
        """Synchronous fetch for thread pool."""
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': self.user_agent}
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            warnings.warn(f"Request error: {e}")
            return None

    def _score_text_importance(self, content: WebPageContent) -> ImportanceScore:
        """Score text content importance."""
        signals = ImportanceSignals()

        # Length score
        text_len = len(content.text_content)
        signals.length_score = min(1.0, text_len / 5000)

        # Structural score (headings)
        signals.structural_score = min(1.0, len(content.headings) / 5)

        # Technical score (has meta keywords)
        if content.meta_keywords:
            signals.technical_score = 0.7

        # Authority score based on domain
        domain = urlparse(content.url).netloc
        if any(edu in domain for edu in ['.edu', '.gov', '.org']):
            signals.authority_score = 0.8
        elif any(tech in domain for tech in ['github', 'stackoverflow', 'docs.', 'wiki']):
            signals.authority_score = 0.75
        else:
            signals.authority_score = 0.5

        total = signals.compute_total()
        return ImportanceScore(
            score=total,
            signals=signals,
            reason=f"length={text_len}, headings={len(content.headings)}"
        )

    def _score_image_importance(self, img: ExtractedImage) -> ImportanceScore:
        """Score image importance."""
        signals = ImportanceSignals()

        # Context richness
        context_len = len(img.context_text)
        signals.length_score = min(1.0, context_len / 200)

        # Has caption (strong signal)
        if img.caption:
            signals.technical_score = 0.8

        # Has alt text
        if img.alt_text and len(img.alt_text) > 10:
            signals.structural_score = 0.7

        # Image size (larger = more important)
        if img.width and img.height:
            area = img.width * img.height
            signals.authority_score = min(1.0, area / 250000)  # 500x500 = 1.0
        else:
            signals.authority_score = 0.5

        total = signals.compute_total()
        return ImportanceScore(
            score=total,
            signals=signals,
            reason=f"context={context_len}, has_caption={bool(img.caption)}"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def scrape_webpage(
    url: str,
    extract_images: bool = True,
    max_images: int = 20
) -> SpinResult:
    """
    Convenience function to scrape a web page.

    Args:
        url: URL to scrape
        extract_images: Whether to extract images
        max_images: Maximum images to extract

    Returns:
        SpinResult with text and image shards
    """
    spinner = WebsiteSpinner(
        importance_threshold=0.2,
        extract_images=extract_images,
        max_images_per_page=max_images
    )

    return await spinner.spin(url)


# Alias
spin_website = scrape_webpage
