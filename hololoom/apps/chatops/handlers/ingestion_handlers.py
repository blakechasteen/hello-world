from __future__ import annotations
"""
Ingestion Handlers for Matrix ChatOps
======================================

Matrix command handlers for HoloLoom SpinningWheel data ingestion.
Exposes 47+ adapters to ChatOps via the !ingest command family.

Commands:
- !ingest <source> - Auto-detect and ingest any source
- !ingest youtube <url> - YouTube video transcript
- !ingest pdf <path/url> - PDF document
- !ingest url <url> - Web page content
- !ingest git <path> - Git repository analysis
- !ingest image <path> - Image with CLIP embeddings
- !ingest status - List available adapters
- !ingest help - Ingestion help

Usage:
    from hololoom.apps.chatops.handlers.ingestion_handlers import register_ingestion_handlers

    # In run_chatops.py:
    register_ingestion_handlers(bot, memory_backend)

Created: 2025-12-09
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Handler registry
try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerCategory,
        HandlerRegistry,
        chatops_handler,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# SpinningWheel imports (graceful degradation)
try:
    from hololoom.spinningWheel.auto import spin, spin_batch, spin_url
    SPIN_AVAILABLE = True
except ImportError:
    SPIN_AVAILABLE = False

try:
    from hololoom.spinningWheel.protocol import SpinnerCapabilities, SpinResult
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False

# Specific spinners (optional)
try:
    from hololoom.spinningWheel.youtube import YouTubeSpinner
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

try:
    from hololoom.spinningWheel.pdf_extractor import PDFSpinner
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from hololoom.spinningWheel.url import URLSpinner
    URL_AVAILABLE = True
except ImportError:
    URL_AVAILABLE = False

try:
    from hololoom.spinningWheel.git_parser import GitSpinner
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    from hololoom.spinningWheel.image import ImageSpinner
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Source Type Detection
# ============================================================================

@dataclass
class SourceDetection:
    """Result of source type detection."""
    source_type: str  # youtube, pdf, url, git, image, text, unknown
    confidence: float  # 0.0-1.0
    adapter_name: str  # Display name
    adapter_available: bool


def detect_source_type(source: str) -> SourceDetection:
    """
    Auto-detect the type of source for routing to appropriate adapter.

    Args:
        source: Raw source string (URL, path, or text)

    Returns:
        SourceDetection with type, confidence, and availability
    """
    source_lower = source.lower().strip()

    # YouTube detection
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'^[a-zA-Z0-9_-]{11}$'  # Video ID only
    ]
    for pattern in youtube_patterns:
        if re.search(pattern, source_lower):
            return SourceDetection(
                source_type='youtube',
                confidence=0.95,
                adapter_name='YouTubeSpinner',
                adapter_available=YOUTUBE_AVAILABLE
            )

    # PDF detection
    if source_lower.endswith('.pdf') or '/pdf/' in source_lower:
        return SourceDetection(
            source_type='pdf',
            confidence=0.95,
            adapter_name='PDFSpinner',
            adapter_available=PDF_AVAILABLE
        )

    # Git repository detection
    git_patterns = [
        r'github\.com/',
        r'gitlab\.com/',
        r'bitbucket\.org/',
        r'\.git$',
        r'^git@'
    ]
    for pattern in git_patterns:
        if re.search(pattern, source_lower):
            return SourceDetection(
                source_type='git',
                confidence=0.90,
                adapter_name='GitSpinner',
                adapter_available=GIT_AVAILABLE
            )

    # Local path detection
    if Path(source).exists():
        path = Path(source)
        suffix = path.suffix.lower()

        # Image files
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return SourceDetection(
                source_type='image',
                confidence=0.95,
                adapter_name='ImageSpinner',
                adapter_available=IMAGE_AVAILABLE
            )

        # PDF files
        if suffix == '.pdf':
            return SourceDetection(
                source_type='pdf',
                confidence=0.95,
                adapter_name='PDFSpinner',
                adapter_available=PDF_AVAILABLE
            )

        # Git directory
        if (path / '.git').exists() or path.name == '.git':
            return SourceDetection(
                source_type='git',
                confidence=0.90,
                adapter_name='GitSpinner',
                adapter_available=GIT_AVAILABLE
            )

    # URL detection (general)
    if source_lower.startswith(('http://', 'https://', 'www.')):
        return SourceDetection(
            source_type='url',
            confidence=0.85,
            adapter_name='URLSpinner',
            adapter_available=URL_AVAILABLE
        )

    # Plain text fallback
    if len(source) > 10:  # Reasonable text length
        return SourceDetection(
            source_type='text',
            confidence=0.70,
            adapter_name='TextSpinner',
            adapter_available=SPIN_AVAILABLE  # spin() handles text
        )

    return SourceDetection(
        source_type='unknown',
        confidence=0.0,
        adapter_name='Unknown',
        adapter_available=False
    )


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_ingest(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest command - Auto-detect and ingest any source.

    Args:
        room: Matrix room
        event: Message event
        args: Source to ingest (URL, path, or text)
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest <source>\nExample: !ingest https://youtube.com/watch?v=abc123"

    if not SPIN_AVAILABLE:
        return "SpinningWheel not available. Install with: pip install HoloLoom[spinners]"

    source = args.strip()

    # Detect source type
    detection = detect_source_type(source)

    if not detection.adapter_available:
        return f"Adapter '{detection.adapter_name}' not available for {detection.source_type} sources."

    try:
        start_time = time.time()

        # Use universal spin() function
        result = await spin(source, memory=memory, return_shards=True)

        elapsed_ms = (time.time() - start_time) * 1000

        # Format response
        return _format_ingestion_result(
            source=source,
            detection=detection,
            shards=result,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in ingest: {e}")
        return f"Ingestion failed: {str(e)}"


async def handle_ingest_youtube(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest youtube command - YouTube video transcript.

    Args:
        room: Matrix room
        event: Message event
        args: YouTube URL or video ID
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest youtube <url>\nExample: !ingest youtube https://youtube.com/watch?v=abc123"

    if not YOUTUBE_AVAILABLE:
        return "YouTubeSpinner not available. Install: pip install youtube-transcript-api"

    url = args.strip()

    try:
        start_time = time.time()

        spinner = YouTubeSpinner()
        result = await spinner.spin({'url': url})

        elapsed_ms = (time.time() - start_time) * 1000

        detection = SourceDetection(
            source_type='youtube',
            confidence=1.0,
            adapter_name='YouTubeSpinner',
            adapter_available=True
        )

        shards = result.shards if hasattr(result, 'shards') else result

        return _format_ingestion_result(
            source=url,
            detection=detection,
            shards=shards,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in youtube ingest: {e}")
        return f"YouTube ingestion failed: {str(e)}"


async def handle_ingest_pdf(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest pdf command - PDF document extraction.

    Args:
        room: Matrix room
        event: Message event
        args: PDF path or URL
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest pdf <path/url>\nExample: !ingest pdf /path/to/document.pdf"

    if not PDF_AVAILABLE:
        return "PDFSpinner not available. Install: pip install pypdf2 pdfplumber"

    source = args.strip()

    try:
        start_time = time.time()

        spinner = PDFSpinner()
        result = await spinner.spin({'path': source})

        elapsed_ms = (time.time() - start_time) * 1000

        detection = SourceDetection(
            source_type='pdf',
            confidence=1.0,
            adapter_name='PDFSpinner',
            adapter_available=True
        )

        shards = result.shards if hasattr(result, 'shards') else result

        return _format_ingestion_result(
            source=source,
            detection=detection,
            shards=shards,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in pdf ingest: {e}")
        return f"PDF ingestion failed: {str(e)}"


async def handle_ingest_url(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest url command - Web page content extraction.

    Args:
        room: Matrix room
        event: Message event
        args: Web URL
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest url <url>\nExample: !ingest url https://example.com/article"

    if not URL_AVAILABLE:
        return "URLSpinner not available. Install: pip install beautifulsoup4 requests"

    url = args.strip()

    # Parse optional depth flag
    max_depth = 1
    follow_links = False

    if url.startswith('--depth='):
        parts = url.split(' ', 1)
        try:
            max_depth = int(parts[0].replace('--depth=', ''))
            follow_links = max_depth > 1
        except ValueError:
            pass
        url = parts[1] if len(parts) > 1 else ''

    if not url:
        return "Usage: !ingest url [--depth=N] <url>"

    try:
        start_time = time.time()

        if SPIN_AVAILABLE and follow_links:
            # Use recursive crawler
            result = await spin_url(url, memory=memory, max_depth=max_depth, follow_links=True)
            shards = result if isinstance(result, list) else []
        else:
            spinner = URLSpinner()
            result = await spinner.spin({'url': url})
            shards = result.shards if hasattr(result, 'shards') else result

        elapsed_ms = (time.time() - start_time) * 1000

        detection = SourceDetection(
            source_type='url',
            confidence=1.0,
            adapter_name='URLSpinner',
            adapter_available=True
        )

        return _format_ingestion_result(
            source=url,
            detection=detection,
            shards=shards,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in url ingest: {e}")
        return f"URL ingestion failed: {str(e)}"


async def handle_ingest_git(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest git command - Git repository analysis.

    Args:
        room: Matrix room
        event: Message event
        args: Git repository path or URL
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest git <path/url>\nExample: !ingest git https://github.com/user/repo"

    if not GIT_AVAILABLE:
        return "GitSpinner not available. Install: pip install gitpython"

    source = args.strip()

    try:
        start_time = time.time()

        spinner = GitSpinner()
        result = await spinner.spin({'repo': source})

        elapsed_ms = (time.time() - start_time) * 1000

        detection = SourceDetection(
            source_type='git',
            confidence=1.0,
            adapter_name='GitSpinner',
            adapter_available=True
        )

        shards = result.shards if hasattr(result, 'shards') else result

        return _format_ingestion_result(
            source=source,
            detection=detection,
            shards=shards,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in git ingest: {e}")
        return f"Git ingestion failed: {str(e)}"


async def handle_ingest_image(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: Any | None = None
) -> str:
    """
    Handle !ingest image command - Image with CLIP embeddings.

    Args:
        room: Matrix room
        event: Message event
        args: Image path
        memory: Optional memory backend

    Returns:
        Response message with ingestion results
    """
    if not args.strip():
        return "Usage: !ingest image <path>\nExample: !ingest image /path/to/photo.jpg"

    if not IMAGE_AVAILABLE:
        return "ImageSpinner not available. Install: pip install pillow torch torchvision"

    path = args.strip()

    try:
        start_time = time.time()

        spinner = ImageSpinner()
        result = await spinner.spin({'path': path})

        elapsed_ms = (time.time() - start_time) * 1000

        detection = SourceDetection(
            source_type='image',
            confidence=1.0,
            adapter_name='ImageSpinner',
            adapter_available=True
        )

        shards = result.shards if hasattr(result, 'shards') else result

        return _format_ingestion_result(
            source=path,
            detection=detection,
            shards=shards,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Error in image ingest: {e}")
        return f"Image ingestion failed: {str(e)}"


async def handle_ingest_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !ingest status command - List available adapters.

    Returns:
        Status message showing adapter availability
    """
    adapters = [
        ('SpinningWheel Core', SPIN_AVAILABLE, 'Auto-detection routing'),
        ('YouTubeSpinner', YOUTUBE_AVAILABLE, 'YouTube transcripts'),
        ('PDFSpinner', PDF_AVAILABLE, 'PDF documents'),
        ('URLSpinner', URL_AVAILABLE, 'Web pages'),
        ('GitSpinner', GIT_AVAILABLE, 'Git repositories'),
        ('ImageSpinner', IMAGE_AVAILABLE, 'Images with CLIP'),
    ]

    response = "## SpinningWheel Adapter Status\n\n"

    available_count = 0
    for name, available, description in adapters:
        status = "OK" if available else "Missing"
        icon = "+" if available else "-"
        response += f"{icon} **{name}**: [{status}] {description}\n"
        if available:
            available_count += 1

    response += f"\n**Total**: {available_count}/{len(adapters)} adapters available\n"

    if available_count < len(adapters):
        response += "\n_Install missing adapters with: `pip install HoloLoom[spinners]`_"

    return response


async def handle_ingest_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !ingest help command.

    Returns:
        Help message
    """
    return """## Ingestion Commands

**Universal Ingestion:**
- `!ingest <source>` - Auto-detect and ingest any source

**Specific Adapters:**
- `!ingest youtube <url>` - YouTube video transcript
- `!ingest pdf <path/url>` - PDF document
- `!ingest url <url>` - Web page content
- `!ingest url --depth=2 <url>` - Crawl with depth
- `!ingest git <path/url>` - Git repository
- `!ingest image <path>` - Image with CLIP

**System:**
- `!ingest status` - Show adapter availability
- `!ingest help` - This help message

**Auto-Detection:**
The universal `!ingest <source>` command automatically detects:
- YouTube URLs/video IDs
- PDF files (by extension or URL pattern)
- Git repositories (GitHub, GitLab, etc.)
- Image files (jpg, png, gif, webp)
- Web URLs (http/https)
- Plain text (fallback)

**Examples:**
```
!ingest https://youtube.com/watch?v=dQw4w9WgXcQ
!ingest /path/to/research.pdf
!ingest https://github.com/anthropics/claude-code
!ingest https://example.com/article
!ingest /photos/diagram.png
!ingest "Thompson Sampling balances exploration and exploitation"
```
"""


# ============================================================================
# Helpers
# ============================================================================

def _format_ingestion_result(
    source: str,
    detection: SourceDetection,
    shards: Any,
    elapsed_ms: float
) -> str:
    """Format ingestion result for Matrix response."""

    # Handle different shard formats
    if isinstance(shards, list):
        shard_list = shards
    elif hasattr(shards, 'shards'):
        shard_list = shards.shards
    else:
        shard_list = [shards] if shards else []

    shard_count = len(shard_list)

    # Extract statistics
    entity_count = 0
    motif_count = 0
    total_importance = 0.0
    sample_text = ""

    for shard in shard_list[:10]:  # Sample first 10
        if hasattr(shard, 'entities'):
            entity_count += len(shard.entities)
        if hasattr(shard, 'motifs'):
            motif_count += len(shard.motifs)
        if hasattr(shard, 'importance'):
            total_importance += shard.importance
        if not sample_text and hasattr(shard, 'text'):
            sample_text = shard.text[:150]

    avg_importance = total_importance / max(shard_count, 1)

    # Truncate source for display
    source_display = source[:60] + "..." if len(source) > 60 else source

    response = f"""## Ingestion Complete

**Source**: {source_display}
**Adapter**: {detection.adapter_name}
**Status**: [OK] Success

### Results
- Shards created: {shard_count}
- Entities found: {entity_count}
- Topics/motifs: {motif_count}
- Avg importance: {avg_importance:.2f}
- Processing time: {elapsed_ms:.1f}ms
"""

    if sample_text:
        # Escape for Matrix markdown
        sample_text = sample_text.replace('\n', ' ').strip()
        response += f"\n### Sample Content\n> \"{sample_text}...\"\n"

    return response


# ============================================================================
# Registration
# ============================================================================

def register_ingestion_handlers(
    bot,
    memory: Any | None = None
) -> None:
    """
    Register all ingestion command handlers.

    Args:
        bot: Matrix bot instance
        memory: Optional memory backend for storing shards
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, ingestion handlers not registered")
        return

    logger.info("Registering ingestion command handlers")

    # Define command map
    command_map = {
        "youtube": lambda room, event, args: handle_ingest_youtube(room, event, args, memory),
        "pdf": lambda room, event, args: handle_ingest_pdf(room, event, args, memory),
        "url": lambda room, event, args: handle_ingest_url(room, event, args, memory),
        "git": lambda room, event, args: handle_ingest_git(room, event, args, memory),
        "image": lambda room, event, args: handle_ingest_image(room, event, args, memory),
        "status": handle_ingest_status,
        "help": handle_ingest_help
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(f"ingest {cmd}", handler)

    # Also register universal !ingest
    bot.register_command("ingest", lambda room, event, args: handle_ingest(room, event, args, memory))

    logger.info(f"Registered {len(command_map) + 1} ingestion commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class IngestionHandlers:
    """
    Decorator-based ChatOps handlers for data ingestion.

    Usage:
        from hololoom.apps.chatops.handlers.ingestion_handlers import IngestionHandlers

        handlers = IngestionHandlers(memory=memory_backend)
        registry.register_instance(handlers)
    """

    def __init__(self, memory: Any | None = None):
        """
        Initialize with optional memory backend.

        Args:
            memory: Memory backend for storing ingested shards
        """
        self.memory = memory

    @HandlerRegistry.register(
        command="ingest",
        description="Auto-detect and ingest any source",
        usage="!ingest <source>",
        category=HandlerCategory.LEARNING,
        aliases=["in"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_universal(self, room, event, args: str) -> str:
        """Handle !ingest command - universal ingestion."""
        return await handle_ingest(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest youtube",
        description="Ingest YouTube video transcript",
        usage="!ingest youtube <url>",
        category=HandlerCategory.LEARNING,
        aliases=["iy"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_youtube(self, room, event, args: str) -> str:
        """Handle !ingest youtube command."""
        return await handle_ingest_youtube(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest pdf",
        description="Ingest PDF document",
        usage="!ingest pdf <path/url>",
        category=HandlerCategory.LEARNING,
        aliases=["ip"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_pdf(self, room, event, args: str) -> str:
        """Handle !ingest pdf command."""
        return await handle_ingest_pdf(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest url",
        description="Ingest web page content",
        usage="!ingest url [--depth=N] <url>",
        category=HandlerCategory.LEARNING,
        aliases=["iu"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_url(self, room, event, args: str) -> str:
        """Handle !ingest url command."""
        return await handle_ingest_url(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest git",
        description="Ingest Git repository",
        usage="!ingest git <path/url>",
        category=HandlerCategory.LEARNING,
        aliases=["ig"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_git(self, room, event, args: str) -> str:
        """Handle !ingest git command."""
        return await handle_ingest_git(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest image",
        description="Ingest image with CLIP embeddings",
        usage="!ingest image <path>",
        category=HandlerCategory.LEARNING,
        aliases=["ii"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_image(self, room, event, args: str) -> str:
        """Handle !ingest image command."""
        return await handle_ingest_image(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="ingest status",
        description="Show adapter availability",
        usage="!ingest status",
        category=HandlerCategory.SYSTEM,
        aliases=["is"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_status(self, room, event, args: str) -> str:
        """Handle !ingest status command."""
        return await handle_ingest_status(room, event, args)

    @HandlerRegistry.register(
        command="ingest help",
        description="Show ingestion help",
        usage="!ingest help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !ingest help command."""
        return await handle_ingest_help(room, event, args)
