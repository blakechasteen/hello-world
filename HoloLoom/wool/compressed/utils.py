"""
Compression Utilities
=====================

Compression algorithms and adaptive selection logic.

This module provides:
- LZ4 compression (fast, 2-4x ratio)
- Zstd compression (slower, 3-7x ratio)
- Adaptive algorithm selection
- Graceful fallback if libraries unavailable

Compression Selection Rules:
- Files <1KB: Skip compression (overhead > benefit)
- Text/JSON/HTML: Prefer Zstd (high compression ratio)
- Binary/Media: Skip compression (already compressed)
- Default: LZ4 (fast, universal)

Author: Claude Code
Date: November 17, 2025 (Phase 7.2)
"""

import logging
from enum import Enum
from typing import Tuple

# Optional compression libraries
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Available compression algorithms."""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"


# MIME types that benefit from compression
COMPRESSIBLE_TYPES = {
    'text/plain',
    'text/html',
    'text/css',
    'text/javascript',
    'text/xml',
    'application/json',
    'application/xml',
    'application/javascript',
    'application/x-yaml'
}

# MIME types already compressed (skip compression)
PRECOMPRESSED_TYPES = {
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
    'video/mp4',
    'video/webm',
    'audio/mp3',
    'audio/ogg',
    'application/zip',
    'application/gzip',
    'application/x-bzip2',
    'application/x-7z-compressed'
}


def select_compression(
    data: bytes,
    content_type: str = 'application/octet-stream',
    adaptive: bool = True
) -> str:
    """
    Auto-select compression algorithm based on data characteristics.

    Args:
        data: Data to compress
        content_type: MIME type
        adaptive: Whether to use adaptive selection

    Returns:
        Algorithm string ('none', 'lz4', 'zstd', or 'zstd:N')
    """
    size = len(data)

    # Skip small files (overhead > benefit)
    if size < 1024:
        return 'none'

    # Skip pre-compressed formats
    if content_type in PRECOMPRESSED_TYPES:
        return 'none'

    # If not adaptive, check library availability
    if not adaptive:
        if LZ4_AVAILABLE:
            return 'lz4'
        elif ZSTD_AVAILABLE:
            return 'zstd:3'
        else:
            return 'none'

    # Adaptive selection
    if content_type in COMPRESSIBLE_TYPES:
        # Text-heavy: high compression ratio
        if size > 100_000:
            # Large files: use Zstd for best ratio
            if ZSTD_AVAILABLE:
                return 'zstd:3'
            elif LZ4_AVAILABLE:
                return 'lz4'
            else:
                return 'none'
        else:
            # Medium files: use LZ4 for speed
            if LZ4_AVAILABLE:
                return 'lz4'
            elif ZSTD_AVAILABLE:
                return 'zstd:3'
            else:
                return 'none'

    # Binary data: try LZ4 (fast, might help)
    if LZ4_AVAILABLE:
        return 'lz4'
    elif ZSTD_AVAILABLE:
        return 'zstd:1'  # Low level for speed
    else:
        return 'none'


def compress_data(data: bytes, algorithm: str = 'lz4') -> Tuple[bytes, str]:
    """
    Compress data using specified algorithm.

    Args:
        data: Data to compress
        algorithm: Algorithm to use ('none', 'lz4', 'zstd', 'zstd:N')

    Returns:
        Tuple of (compressed_data, actual_algorithm_used)

    Raises:
        ValueError: If algorithm not supported
    """
    if algorithm == 'none':
        return data, 'none'

    # LZ4 compression
    if algorithm == 'lz4':
        if not LZ4_AVAILABLE:
            logger.warning("LZ4 not available, storing uncompressed")
            return data, 'none'

        try:
            compressed = lz4.frame.compress(
                data,
                compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC  # Fast
            )

            # Check if compression helped (sometimes it makes it worse!)
            if len(compressed) >= len(data):
                logger.debug(f"LZ4 increased size, storing uncompressed")
                return data, 'none'

            return compressed, 'lz4'

        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            return data, 'none'

    # Zstd compression
    if algorithm.startswith('zstd'):
        if not ZSTD_AVAILABLE:
            logger.warning("Zstd not available, falling back to LZ4")
            return compress_data(data, 'lz4')

        # Parse compression level
        if ':' in algorithm:
            level = int(algorithm.split(':')[1])
        else:
            level = 3  # Default

        try:
            cctx = zstd.ZstdCompressor(level=level)
            compressed = cctx.compress(data)

            # Check if compression helped
            if len(compressed) >= len(data):
                logger.debug(f"Zstd increased size, storing uncompressed")
                return data, 'none'

            return compressed, f'zstd:{level}'

        except Exception as e:
            logger.error(f"Zstd compression failed: {e}")
            # Fallback to LZ4
            return compress_data(data, 'lz4')

    # Unknown algorithm
    raise ValueError(f"Unknown compression algorithm: {algorithm}")


def decompress_data(data: bytes, algorithm: str) -> bytes:
    """
    Decompress data using specified algorithm.

    Args:
        data: Compressed data
        algorithm: Algorithm used ('none', 'lz4', 'zstd', 'zstd:N')

    Returns:
        Decompressed data

    Raises:
        ValueError: If algorithm not supported
        RuntimeError: If decompression fails
    """
    if algorithm == 'none':
        return data

    # LZ4 decompression
    if algorithm == 'lz4':
        if not LZ4_AVAILABLE:
            raise RuntimeError("LZ4 not available for decompression")

        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            raise RuntimeError(f"LZ4 decompression failed: {e}")

    # Zstd decompression
    if algorithm.startswith('zstd'):
        if not ZSTD_AVAILABLE:
            raise RuntimeError("Zstd not available for decompression")

        try:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except Exception as e:
            raise RuntimeError(f"Zstd decompression failed: {e}")

    # Unknown algorithm
    raise ValueError(f"Unknown compression algorithm: {algorithm}")


def get_compression_info() -> dict:
    """
    Get compression library availability information.

    Returns:
        Dictionary with availability and capabilities
    """
    info = {
        'lz4': {
            'available': LZ4_AVAILABLE,
            'speed': 'very_fast',
            'ratio': '2-4x'
        },
        'zstd': {
            'available': ZSTD_AVAILABLE,
            'speed': 'fast',
            'ratio': '3-7x'
        }
    }

    # Add version info if available
    if LZ4_AVAILABLE:
        try:
            info['lz4']['version'] = lz4.frame.VERSION
        except:
            pass

    if ZSTD_AVAILABLE:
        try:
            info['zstd']['version'] = zstd.ZSTD_VERSION_STRING
        except:
            pass

    return info


# Log availability at module load
if not LZ4_AVAILABLE:
    logger.warning("lz4 not available - compression disabled. Install: pip install lz4")

if not ZSTD_AVAILABLE:
    logger.warning("zstandard not available - high-ratio compression disabled. Install: pip install zstandard")

if not (LZ4_AVAILABLE or ZSTD_AVAILABLE):
    logger.error("No compression libraries available! Wool storage will not compress data.")
