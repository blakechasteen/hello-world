"""
Image Utilities for Multimodal Web Scraping
============================================

Extract, download, analyze, and store images from webpages.

Features:
- Download images from URLs
- Filter for meaningful images (skip icons, logos, ads)
- Extract image context (captions, surrounding text)
- Analyze image relevance to content
- Generate embeddings for image similarity search
- Store images with metadata
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import base64

try:
    import requests
    from PIL import Image
    from io import BytesIO
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    requests = None
    Image = None
    BytesIO = None

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about an extracted image."""
    url: str                          # Image URL
    alt_text: str                     # Alt text description
    title: str                        # Title attribute
    caption: str                      # Caption text (from <figcaption>, nearby text)
    context: str                      # Surrounding text context
    width: Optional[int] = None       # Image dimensions
    height: Optional[int] = None
    size_bytes: Optional[int] = None  # File size
    format: Optional[str] = None      # JPEG, PNG, etc.
    relevance_score: float = 0.0      # How relevant to main content (0-1)
    local_path: Optional[str] = None  # Where image is stored locally
    hash: Optional[str] = None        # Content hash for deduplication


class ImageExtractor:
    """Extract meaningful images from webpages."""

    # Minimum dimensions for meaningful images
    MIN_WIDTH = 200
    MIN_HEIGHT = 200

    # Skip images matching these patterns
    SKIP_PATTERNS = [
        r'logo',
        r'icon',
        r'avatar',
        r'badge',
        r'button',
        r'spacer',
        r'pixel',
        r'tracking',
        r'analytics',
        r'ad[s]?',
        r'banner',
        r'sprite',
    ]

    # Preferred image formats
    PREFERRED_FORMATS = ['jpg', 'jpeg', 'png', 'webp']

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
        max_images: int = 10
    ):
        """
        Initialize image extractor.

        Args:
            storage_dir: Where to save downloaded images (None = don't save)
            min_width: Minimum image width to consider
            min_height: Minimum image height to consider
            max_images: Maximum images to extract per page
        """
        self.storage_dir = storage_dir
        self.min_width = min_width
        self.min_height = min_height
        self.max_images = max_images

        if storage_dir:
            storage_dir.mkdir(parents=True, exist_ok=True)

    def extract_images(
        self,
        soup: "BeautifulSoup",
        base_url: str
    ) -> List[ImageInfo]:
        """
        Extract meaningful images from HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative paths

        Returns:
            List of ImageInfo objects
        """
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available")
            return []

        images = []

        # Find all <img> tags
        for img in soup.find_all('img'):
            try:
                info = self._extract_image_info(img, base_url, soup)
                if info and self._is_meaningful(info):
                    images.append(info)

                    if len(images) >= self.max_images:
                        break
            except Exception as e:
                logger.debug(f"Error extracting image: {e}")
                continue

        # Sort by relevance
        images.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Extracted {len(images)} meaningful images")
        return images

    def _extract_image_info(
        self,
        img_tag,
        base_url: str,
        soup: "BeautifulSoup"
    ) -> Optional[ImageInfo]:
        """Extract information from <img> tag."""

        # Get image URL
        src = img_tag.get('src') or img_tag.get('data-src')
        if not src:
            return None

        # Skip data URLs for now
        if src.startswith('data:'):
            return None

        # Resolve relative URLs
        url = urljoin(base_url, src)

        # Get alt text and title
        alt_text = img_tag.get('alt', '').strip()
        title = img_tag.get('title', '').strip()

        # Get dimensions
        width = self._parse_dimension(img_tag.get('width'))
        height = self._parse_dimension(img_tag.get('height'))

        # Find caption (figcaption, nearby text)
        caption = self._find_caption(img_tag)

        # Get surrounding context
        context = self._extract_context(img_tag)

        # Calculate relevance score
        relevance = self._calculate_relevance(
            url, alt_text, title, caption, context, width, height
        )

        return ImageInfo(
            url=url,
            alt_text=alt_text,
            title=title,
            caption=caption,
            context=context,
            width=width,
            height=height,
            relevance_score=relevance
        )

    def _find_caption(self, img_tag) -> str:
        """Find caption text associated with image."""
        # Check for <figcaption> in parent <figure>
        parent = img_tag.parent
        if parent and parent.name == 'figure':
            figcaption = parent.find('figcaption')
            if figcaption:
                return figcaption.get_text(strip=True)

        # Check for nearby <p> with class containing 'caption'
        for sibling in img_tag.find_next_siblings(['p', 'div'], limit=2):
            if 'caption' in str(sibling.get('class', '')).lower():
                return sibling.get_text(strip=True)

        return ""

    def _extract_context(self, img_tag, context_chars: int = 200) -> str:
        """Extract surrounding text context."""
        # Get text from parent container
        parent = img_tag.parent
        if parent:
            text = parent.get_text(separator=' ', strip=True)
            # Remove the image alt text from context
            text = text.replace(img_tag.get('alt', ''), '')
            return text[:context_chars]
        return ""

    def _calculate_relevance(
        self,
        url: str,
        alt_text: str,
        title: str,
        caption: str,
        context: str,
        width: Optional[int],
        height: Optional[int]
    ) -> float:
        """
        Calculate image relevance score (0-1).

        Higher score = more likely to be meaningful content.
        """
        score = 0.5  # Start neutral

        # Check URL patterns (negative signals)
        url_lower = url.lower()
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, url_lower):
                score -= 0.3
                break

        # Alt text quality (positive signal)
        if alt_text:
            if len(alt_text) > 20:
                score += 0.2
            elif len(alt_text) > 5:
                score += 0.1

        # Has caption or title (positive)
        if caption:
            score += 0.15
        if title:
            score += 0.05

        # Good dimensions (positive)
        if width and height:
            if width >= 400 and height >= 300:
                score += 0.2
            elif width >= self.min_width and height >= self.min_height:
                score += 0.1

            # Penalize very tall/wide images (likely banners)
            aspect = width / height if height > 0 else 1
            if aspect > 5 or aspect < 0.2:
                score -= 0.2

        # Has surrounding context (positive)
        if context and len(context) > 50:
            score += 0.1

        # Format preference
        if any(fmt in url_lower for fmt in self.PREFERRED_FORMATS):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _is_meaningful(self, info: ImageInfo) -> bool:
        """Check if image meets criteria for meaningful content."""
        # Check relevance threshold
        if info.relevance_score < 0.3:
            return False

        # Check dimensions if available
        if info.width and info.height:
            if info.width < self.min_width or info.height < self.min_height:
                return False

        # Must have some descriptive text
        if not (info.alt_text or info.caption or info.title):
            return False

        return True

    def _parse_dimension(self, value) -> Optional[int]:
        """Parse width/height attribute."""
        if not value:
            return None
        try:
            # Remove 'px' suffix if present
            value_str = str(value).lower().replace('px', '').strip()
            return int(value_str)
        except (ValueError, TypeError):
            return None

    async def download_image(self, info: ImageInfo) -> bool:
        """
        Download image and populate metadata.

        Updates info with: size_bytes, format, width, height, local_path, hash
        """
        if not IMAGING_AVAILABLE:
            logger.warning("PIL not available for image download")
            return False

        try:
            # Download
            response = requests.get(info.url, timeout=10, stream=True)
            response.raise_for_status()

            # Read image data
            image_data = response.content

            # Calculate hash for deduplication
            info.hash = hashlib.sha256(image_data).hexdigest()[:16]

            # Get size
            info.size_bytes = len(image_data)

            # Load with PIL to get format and dimensions
            img = Image.open(BytesIO(image_data))
            info.format = img.format
            info.width = img.width
            info.height = img.height

            # Save if storage directory configured
            if self.storage_dir:
                ext = info.format.lower() if info.format else 'jpg'
                filename = f"{info.hash}.{ext}"
                local_path = self.storage_dir / filename

                if not local_path.exists():
                    with open(local_path, 'wb') as f:
                        f.write(image_data)

                info.local_path = str(local_path)
                logger.debug(f"Saved image: {filename}")

            return True

        except Exception as e:
            logger.warning(f"Failed to download {info.url}: {e}")
            return False

    async def download_all(self, images: List[ImageInfo]) -> List[ImageInfo]:
        """Download all images and update metadata."""
        downloaded = []

        for info in images:
            success = await self.download_image(info)
            if success:
                # Re-check if image is meaningful after getting real dimensions
                if self._is_meaningful(info):
                    downloaded.append(info)

        logger.info(f"Downloaded {len(downloaded)}/{len(images)} images")
        return downloaded


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for embedding in memory."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_and_download_images(
    html_content: str,
    base_url: str,
    storage_dir: Optional[Path] = None,
    max_images: int = 10
) -> List[ImageInfo]:
    """
    Convenience function to extract and download images from HTML.

    Args:
        html_content: Raw HTML string
        base_url: Base URL for resolving relative paths
        storage_dir: Where to save images (None = don't save)
        max_images: Maximum images to extract

    Returns:
        List of ImageInfo objects with downloaded metadata
    """
    if not BS4_AVAILABLE:
        logger.error("BeautifulSoup not available")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    extractor = ImageExtractor(storage_dir, max_images=max_images)

    images = extractor.extract_images(soup, base_url)

    # Download images (synchronously for now)
    import asyncio
    downloaded = asyncio.run(extractor.download_all(images))

    return downloaded
