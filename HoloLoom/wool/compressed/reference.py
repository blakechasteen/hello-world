"""
Compressed Wool Reference
=========================

Extended WoolReference with compression metadata.

This module provides references that track compression state:
- Compression algorithm used (none, lz4, zstd)
- Compression level
- Compressed vs uncompressed sizes
- Compression ratio

The reference allows transparent decompression on read.

Author: Claude Code
Date: November 17, 2025 (Phase 7.1)
"""

from dataclasses import dataclass
from typing import Optional

from HoloLoom.wool import WoolReference


@dataclass
class CompressedWoolReference(WoolReference):
    """
    Wool reference with compression metadata.

    Extends WoolReference with compression tracking:
    - Algorithm used (none, lz4, zstd:N)
    - Original and compressed sizes
    - Compression ratio

    Properties:
        savings_bytes: Bytes saved by compression
        savings_percentage: Percentage saved (0-100)
        is_compressed: Whether file is compressed
    """

    # Compression metadata
    compression_algorithm: str = 'none'  # none, lz4, zstd, zstd:N
    compression_level: int = 0  # Algorithm-specific level
    compressed_size: int = 0  # Bytes on disk (compressed)
    uncompressed_size: int = 0  # Original size

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio (uncompressed / compressed).

        Returns:
            Ratio >= 1.0 (higher is better)
        """
        if self.compressed_size == 0:
            return 1.0
        return self.uncompressed_size / self.compressed_size

    @property
    def savings_bytes(self) -> int:
        """
        Bytes saved by compression.

        Returns:
            Bytes saved (may be negative if compression increased size)
        """
        return self.uncompressed_size - self.compressed_size

    @property
    def savings_percentage(self) -> float:
        """
        Percentage saved by compression (0-100).

        Returns:
            Percentage saved
        """
        if self.uncompressed_size == 0:
            return 0.0
        return (self.savings_bytes / self.uncompressed_size) * 100

    @property
    def is_compressed(self) -> bool:
        """
        Whether file is compressed.

        Returns:
            True if compressed
        """
        return self.compression_algorithm != 'none'

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'file_id': self.file_id,
            'offset': self.offset,
            'length': self.length,
            'content_type': self.content_type,
            'compression_algorithm': self.compression_algorithm,
            'compression_level': self.compression_level,
            'compressed_size': self.compressed_size,
            'uncompressed_size': self.uncompressed_size
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CompressedWoolReference':
        """Create from dictionary."""
        return cls(
            file_id=d['file_id'],
            offset=d.get('offset', 0),
            length=d.get('length', 0),
            content_type=d.get('content_type', 'application/octet-stream'),
            compression_algorithm=d.get('compression_algorithm', 'none'),
            compression_level=d.get('compression_level', 0),
            compressed_size=d.get('compressed_size', 0),
            uncompressed_size=d.get('uncompressed_size', 0)
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.is_compressed:
            return (
                f"CompressedWoolReference("
                f"file_id={self.file_id[:12]}..., "
                f"algo={self.compression_algorithm}, "
                f"ratio={self.compression_ratio:.1f}x, "
                f"saved={self.savings_percentage:.0f}%)"
            )
        else:
            return f"CompressedWoolReference(file_id={self.file_id[:12]}..., uncompressed)"
