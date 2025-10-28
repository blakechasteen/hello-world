"""
SpinningWheel
=============
Lightweight input adapters for HoloLoom.

Converts raw modality data -> MemoryShards -> Orchestrator

Philosophy:
- Keep spinners simple and focused
- Standardize output (MemoryShards)
- Optional pre-enrichment with context
- Let the Orchestrator do heavy lifting

Available Spinners:
- AudioSpinner: Converts audio transcripts + metadata -> MemoryShards
- YouTubeSpinner: Extracts YouTube video transcripts -> MemoryShards
- TextSpinner: Plain text/markdown -> MemoryShards
- CodeSpinner: Code/git diffs -> MemoryShards
- WebsiteSpinner: Web content and browser history -> MemoryShards
- RecursiveCrawler: Recursive web crawling with importance gating -> MemoryShards
"""

from .base import BaseSpinner, SpinnerConfig
from .audio import AudioSpinner
from .youtube import YouTubeSpinner, YouTubeSpinnerConfig, transcribe_youtube
from .text import TextSpinner, TextSpinnerConfig, spin_text
from .code import CodeSpinner, CodeSpinnerConfig, spin_code_file, spin_git_diff, spin_repository
from .website import WebsiteSpinner, WebsiteSpinnerConfig, spin_webpage
from .browser_history import BrowserHistoryReader, BrowserVisit, get_recent_history
from .recursive_crawler import RecursiveCrawler, CrawlConfig, LinkInfo, crawl_recursive
from .image_utils import ImageExtractor, ImageInfo
from .batch_utils import batch_ingest_urls, batch_ingest_files, batch_ingest_from_list_file, BatchConfig, BatchResult

__all__ = [
    # Base
    "BaseSpinner",
    "SpinnerConfig",
    # Audio
    "AudioSpinner",
    # YouTube
    "YouTubeSpinner",
    "YouTubeSpinnerConfig",
    "transcribe_youtube",
    # Text
    "TextSpinner",
    "TextSpinnerConfig",
    "spin_text",
    # Code
    "CodeSpinner",
    "CodeSpinnerConfig",
    "spin_code_file",
    "spin_git_diff",
    "spin_repository",
    # Website
    "WebsiteSpinner",
    "WebsiteSpinnerConfig",
    "spin_webpage",
    # Browser History
    "BrowserHistoryReader",
    "BrowserVisit",
    "get_recent_history",
    # Recursive Crawler
    "RecursiveCrawler",
    "CrawlConfig",
    "LinkInfo",
    "crawl_recursive",
    # Image Utils
    "ImageExtractor",
    "ImageInfo",
    # Batch Utils
    "batch_ingest_urls",
    "batch_ingest_files",
    "batch_ingest_from_list_file",
    "BatchConfig",
    "BatchResult",
    # Factory
    "create_spinner"
]

__version__ = "0.1.0"


def create_spinner(modality: str, config: SpinnerConfig = None):
    """
    Factory function to create spinners.

    Args:
        modality: 'audio', 'youtube', 'text', 'code', 'website'
        config: SpinnerConfig (or subclass like WebsiteSpinnerConfig)

    Returns:
        Spinner instance

    Example:
        # Simple usage
        spinner = create_spinner('audio', SpinnerConfig(enable_enrichment=True))
        shards = await spinner.spin(raw_data)

        # Website with custom config
        from HoloLoom.spinning_wheel import WebsiteSpinnerConfig
        config = WebsiteSpinnerConfig(chunk_by='paragraph', extract_images=True)
        spinner = create_spinner('website', config)
        shards = await spinner.spin({'url': 'https://example.com'})
    """
    if config is None:
        config = SpinnerConfig()

    spinners = {
        'audio': AudioSpinner,
        'youtube': YouTubeSpinner,
        'text': TextSpinner,
        'code': CodeSpinner,
        'website': WebsiteSpinner,
        # Future spinners:
        # 'video': VideoSpinner,
    }

    spinner_class = spinners.get(modality)
    if not spinner_class:
        available = ', '.join(spinners.keys())
        raise ValueError(f"Unknown modality: {modality}. Available: {available}")

    return spinner_class(config)