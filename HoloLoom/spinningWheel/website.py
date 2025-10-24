"""
WebsiteSpinner - Process web content into memory shards.

This spinner:
1. Accepts webpage data (URL, title, content)
2. Scrapes content if not provided
3. Chunks via TextSpinner
4. Extracts entities and motifs
5. Enriches with web metadata (URL, domain, visit data)

Use cases:
- Browser history auto-ingest
- Bookmark processing
- Active tab capture
- Research article collection
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse
from dataclasses import dataclass

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    requests = None
    BeautifulSoup = None

from .base import BaseSpinner, SpinnerConfig
from .text import spin_text

logger = logging.getLogger(__name__)


@dataclass
class WebsiteSpinnerConfig(SpinnerConfig):
    """Configuration for website spinner."""

    # Scraping options
    timeout: int = 10
    user_agent: str = "HoloLoom WebsiteSpinner/1.0"

    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 100000

    # Chunking options (passed to TextSpinner)
    chunk_by: Optional[str] = "paragraph"  # paragraph, sentence, fixed
    chunk_size: int = 500

    # Auto-tagging
    auto_tag_domain: bool = True
    include_url_in_tags: bool = False

    # Multimodal options (NEW!)
    extract_images: bool = True           # Extract meaningful images
    image_storage_dir: Optional[str] = None  # Where to save images (None = temp)
    max_images: int = 10                  # Max images per page
    min_image_width: int = 200            # Minimum image dimensions
    min_image_height: int = 200
    download_images: bool = True          # Actually download images (vs just metadata)


class WebsiteSpinner(BaseSpinner):
    """
    Spinner for web content and browser history.

    Converts webpages → MemoryShards with:
    - Smart chunking via TextSpinner
    - Entity and motif extraction
    - URL and domain metadata
    - Visit statistics (if provided)
    """

    def __init__(self, config: Optional[WebsiteSpinnerConfig] = None):
        self.config = config or WebsiteSpinnerConfig()

        if not SCRAPING_AVAILABLE:
            logger.warning(
                "Web scraping not available. Install with: "
                "pip install requests beautifulsoup4"
            )

    async def spin(self, raw_data: Dict) -> List:
        """
        Process webpage into MemoryShards.

        Args:
            raw_data: {
                'url': 'https://example.com/article',  # Required
                'title': 'Article Title',               # Optional
                'content': 'Full page text...',         # Optional (will scrape if missing)
                'visited_at': datetime,                 # Optional
                'duration': 30,                         # Optional (seconds on page)
                'visit_count': 5,                       # Optional
                'tags': ['research', 'python']          # Optional
            }

        Returns:
            List of MemoryShards with web metadata
        """
        url = raw_data.get('url')
        if not url:
            logger.error("URL is required for WebsiteSpinner")
            return []

        # Extract title and content
        title = raw_data.get('title', '')
        content = raw_data.get('content')
        html_content = None  # Store full HTML for image extraction

        # Scrape if content not provided
        if not content:
            if not SCRAPING_AVAILABLE:
                logger.error("Cannot scrape - requests/beautifulsoup not installed")
                return []

            content, html_content = await self._scrape_content(url)
            if not content:
                logger.warning(f"No content scraped from {url}")
                return []

            # Try to extract title if not provided
            if not title:
                title = await self._extract_title(url)

        # Filter by length
        if len(content) < self.config.min_content_length:
            logger.info(f"Content too short ({len(content)} chars), skipping {url}")
            return []

        if len(content) > self.config.max_content_length:
            logger.warning(f"Content too long ({len(content)} chars), truncating {url}")
            content = content[:self.config.max_content_length]

        # Parse URL for metadata
        parsed = urlparse(url)
        domain = parsed.netloc

        # Build full text with title
        if title:
            full_text = f"# {title}\n\nSource: {url}\n\n{content}"
        else:
            full_text = f"Source: {url}\n\n{content}"

        # Extract images if enabled (MULTIMODAL!)
        images = []
        if self.config.extract_images and html_content:
            images = await self._extract_images(url, html_content)

        # Process through text spinner for chunking + entity extraction
        shards = await spin_text(
            text=full_text,
            source=f"website:{domain}",
            chunk_by=self.config.chunk_by,
            chunk_size=self.config.chunk_size
        )

        # Enrich each shard with web metadata AND images
        for shard in shards:
            # Add web-specific metadata
            shard.metadata['url'] = url
            shard.metadata['domain'] = domain
            shard.metadata['title'] = title
            shard.metadata['content_type'] = 'webpage'

            # Add visit statistics if provided
            if 'visited_at' in raw_data:
                shard.metadata['visited_at'] = raw_data['visited_at'].isoformat()
            if 'duration' in raw_data:
                shard.metadata['duration_seconds'] = raw_data['duration']
            if 'visit_count' in raw_data:
                shard.metadata['visit_count'] = raw_data['visit_count']

            # Auto-tag with domain
            if 'tags' not in shard.metadata:
                shard.metadata['tags'] = []

            # Add domain tag
            if self.config.auto_tag_domain:
                shard.metadata['tags'].append(f"web:{domain}")

            # Add URL tag if configured
            if self.config.include_url_in_tags:
                shard.metadata['tags'].append(f"url:{url}")

            # Add user-provided tags
            if 'tags' in raw_data:
                shard.metadata['tags'].extend(raw_data['tags'])

        # Attach images to first shard (main content shard)
        if images and shards:
            shards[0].metadata['images'] = [
                {
                    'url': img.url,
                    'alt_text': img.alt_text,
                    'caption': img.caption,
                    'title': img.title,
                    'context': img.context,
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'size_bytes': img.size_bytes,
                    'local_path': img.local_path,
                    'hash': img.hash,
                    'relevance_score': img.relevance_score
                }
                for img in images
            ]
            shards[0].metadata['image_count'] = len(images)
            logger.info(f"✓ Extracted {len(images)} images")

        logger.info(f"✓ Spun {len(shards)} shards from {url}")
        return shards

    async def _extract_images(self, url: str, html_content: str) -> List:
        """
        Extract meaningful images from webpage.

        Returns:
            List of ImageInfo objects
        """
        try:
            from .image_utils import ImageExtractor
            from pathlib import Path
            import tempfile

            # Determine storage directory
            storage_dir = None
            if self.config.download_images:
                if self.config.image_storage_dir:
                    storage_dir = Path(self.config.image_storage_dir)
                else:
                    # Use temp directory
                    storage_dir = Path(tempfile.gettempdir()) / "hololoom_images"
                storage_dir.mkdir(parents=True, exist_ok=True)

            # Create extractor
            extractor = ImageExtractor(
                storage_dir=storage_dir,
                min_width=self.config.min_image_width,
                min_height=self.config.min_image_height,
                max_images=self.config.max_images
            )

            # Parse HTML
            if not BeautifulSoup:
                logger.warning("BeautifulSoup not available for image extraction")
                return []

            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract images
            images = extractor.extract_images(soup, url)

            # Download if configured
            if self.config.download_images and images:
                images = await extractor.download_all(images)

            return images

        except ImportError:
            logger.warning("Image extraction utilities not available")
            return []
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []

    async def _scrape_content(self, url: str) -> Tuple[str, str]:
        """
        Scrape main content from webpage.

        Uses BeautifulSoup to extract clean text, removing:
        - Scripts and styles
        - Navigation, headers, footers
        - Ads and sidebars

        Returns:
            Tuple of (text_content, html_content)
        """
        try:
            headers = {'User-Agent': self.config.user_agent}
            response = requests.get(url, timeout=self.config.timeout, headers=headers)
            response.raise_for_status()

            # Store original HTML for image extraction
            html_content = response.text

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove noise elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            # Try to find main content area
            main = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', {'class': 'content'}) or
                soup.find('div', {'id': 'content'}) or
                soup.body
            )

            if not main:
                logger.warning(f"No main content found in {url}")
                return "", html_content

            # Extract clean text
            text = main.get_text(separator='\n', strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            return text, html_content

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return "", ""
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return "", ""

    async def _extract_title(self, url: str) -> str:
        """Extract page title from HTML."""
        try:
            headers = {'User-Agent': self.config.user_agent}
            response = requests.get(url, timeout=self.config.timeout, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try multiple title sources
            title = None

            # <title> tag
            if soup.title:
                title = soup.title.string

            # Open Graph title
            if not title:
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    title = og_title.get('content')

            # Twitter title
            if not title:
                tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
                if tw_title:
                    title = tw_title.get('content')

            # <h1> as fallback
            if not title:
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)

            return title.strip() if title else ""

        except Exception as e:
            logger.warning(f"Failed to extract title from {url}: {e}")
            return ""


# Convenience function for simple usage
async def spin_webpage(
    url: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    tags: Optional[List[str]] = None,
    visited_at: Optional[datetime] = None,
    duration: Optional[int] = None
):
    """
    Quick function to spin a webpage into memory shards.

    Args:
        url: Webpage URL
        title: Page title (will scrape if not provided)
        content: Page content (will scrape if not provided)
        tags: Optional tags to add
        visited_at: When the page was visited
        duration: How long spent on page (seconds)

    Returns:
        List of MemoryShards

    Example:
        # With URL only (will scrape)
        shards = await spin_webpage('https://example.com/article')

        # With pre-fetched content
        shards = await spin_webpage(
            url='https://example.com',
            title='My Article',
            content='Full text...',
            tags=['research', 'python']
        )
    """
    spinner = WebsiteSpinner()

    raw_data = {'url': url}
    if title:
        raw_data['title'] = title
    if content:
        raw_data['content'] = content
    if tags:
        raw_data['tags'] = tags
    if visited_at:
        raw_data['visited_at'] = visited_at
    if duration:
        raw_data['duration'] = duration

    return await spinner.spin(raw_data)
