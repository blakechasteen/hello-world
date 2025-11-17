"""
Transparent Compression for Wool Storage
=========================================

Compression layer providing 3-10x storage savings with minimal overhead.

Key Features:
- LZ4 compression for speed (2-4x ratio, 500 MB/s)
- Zstd compression for ratio (3-7x ratio, 200 MB/s)
- Adaptive algorithm selection based on content type
- Transparent compression/decompression
- Backward compatible with uncompressed files

Usage:
    from HoloLoom.wool.compressed import CompressedWoolStorage

    wool = CompressedWoolStorage(
        base_path="./data/wool",
        enable_compression=True,
        default_algorithm="lz4",
        adaptive=True
    )

    # Automatic compression
    ref = wool.store(data, content_type="text/plain")

    # Automatic decompression
    view = wool.read(ref)

Exports:
- CompressedWoolReference
- CompressedWoolStorage
- CompressionAlgorithm
- select_compression

Author: Claude Code
Date: November 17, 2025 (Phase 7)
"""

from HoloLoom.wool.compressed.reference import CompressedWoolReference
from HoloLoom.wool.compressed.utils import (
    CompressionAlgorithm,
    select_compression,
    compress_data,
    decompress_data
)
from HoloLoom.wool.compressed.storage import CompressedWoolStorage

__all__ = [
    'CompressedWoolReference',
    'CompressedWoolStorage',
    'CompressionAlgorithm',
    'select_compression',
    'compress_data',
    'decompress_data'
]
