"""
Wool Storage - Content-Addressable Storage Layer
=================================================

Memory-mapped, content-addressable storage for zero-copy data access.

Key Features:
- Files stored by SHA-256 hash (automatic deduplication)
- All files memory-mapped for zero-copy access
- Thread-safe mmap cache
- Immutable, append-only storage
- Directory sharding for filesystem performance

Architecture:
    ./data/wool/
    ├── e3b/0c4/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    ├── abc/123/abc123def456...
    └── ...

Author: Claude Code
Date: November 17, 2025
"""

import os
import mmap
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WoolReference:
    """
    Reference to data stored in wool storage.

    Immutable reference to content-addressable data.

    Attributes:
        file_id: SHA-256 hash of content (unique identifier)
        offset: Byte offset into file (for sub-regions)
        length: Number of bytes
        path: Physical path to mmap'd file (resolved lazily)
        content_type: MIME type (e.g., 'application/pdf')

    Example:
        ref = WoolReference(
            file_id='e3b0c44298fc1c...',
            offset=0,
            length=10485760,
            content_type='application/pdf'
        )
    """
    file_id: str        # SHA-256 hash (64 hex chars)
    offset: int = 0     # Byte offset
    length: int = 0     # Byte length
    path: Optional[Path] = None   # Physical path (resolved lazily)
    content_type: str = "application/octet-stream"

    def __post_init__(self):
        """Validate file_id format."""
        if len(self.file_id) != 64:
            raise ValueError(f"Invalid file_id length: {len(self.file_id)} (expected 64)")

        # Validate hex
        try:
            int(self.file_id, 16)
        except ValueError:
            raise ValueError(f"Invalid file_id: not hexadecimal")

    def __hash__(self) -> int:
        """Hash based on file_id + offset + length."""
        return hash((self.file_id, self.offset, self.length))

    def to_dict(self) -> dict:
        """Serialize for storage in graph."""
        return {
            'file_id': self.file_id,
            'offset': self.offset,
            'length': self.length,
            'content_type': self.content_type
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WoolReference':
        """Deserialize from graph storage."""
        return cls(
            file_id=data['file_id'],
            offset=data.get('offset', 0),
            length=data.get('length', 0),
            content_type=data.get('content_type', 'application/octet-stream')
        )


# ============================================================================
# Wool Storage Implementation
# ============================================================================

class WoolStorage:
    """
    Content-addressable storage with memory-mapped access.

    Features:
    - Files stored by SHA-256 hash (deduplication)
    - Directory structure: ./data/wool/[first 3]/[next 3]/[full hash]
    - All files memory-mapped for zero-copy access
    - Thread-safe mmap cache
    - Automatic cleanup on close

    Example:
        storage = WoolStorage(base_path='./data/wool')

        # Store data (content-addressable)
        ref = storage.store(pdf_bytes, content_type='application/pdf')
        # → WoolReference(file_id='e3b0c44...', offset=0, length=10485760)

        # Read data (zero-copy)
        data_view = storage.read(ref)
        # → memoryview to mmap'd data (no copy!)

        # Read substring (zero-copy slice)
        substring_view = storage.read_range(ref.file_id, offset=1000, length=500)

        # Cleanup
        storage.close()
    """

    def __init__(
        self,
        base_path: Union[str, Path] = Path('./data/wool'),
        enable_cache: bool = True
    ):
        """
        Initialize wool storage.

        Args:
            base_path: Base directory for wool storage
            enable_cache: Enable mmap caching (recommended)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

        # Cache of open mmap handles (thread-safe)
        self._mmap_cache: Dict[str, mmap.mmap] = {}
        self._file_handles: Dict[str, object] = {}  # Keep files open for mmap
        self._cache_lock = threading.Lock()

        # Statistics
        self._stats = {
            'files_stored': 0,
            'bytes_stored': 0,
            'deduplications': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"Initialized WoolStorage at {self.base_path}")

    def _compute_hash(self, data: bytes) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Raw bytes

        Returns:
            64-character hex string
        """
        return hashlib.sha256(data).hexdigest()

    def _get_file_path(self, file_id: str) -> Path:
        """
        Get file path from hash.

        Directory structure for filesystem performance:
        - First 3 chars: top-level directory
        - Next 3 chars: second-level directory
        - Full hash: filename

        Example:
            e3b0c44298fc1c... → ./data/wool/e3b/0c4/e3b0c44298fc1c...

        Args:
            file_id: SHA-256 hash (64 hex chars)

        Returns:
            Path to file
        """
        if len(file_id) < 6:
            raise ValueError(f"Invalid file_id: too short ({len(file_id)} chars)")

        prefix1 = file_id[:3]
        prefix2 = file_id[3:6]
        directory = self.base_path / prefix1 / prefix2
        return directory / file_id

    def store(
        self,
        data: bytes,
        content_type: str = "application/octet-stream"
    ) -> WoolReference:
        """
        Store data in wool storage (content-addressable).

        If data already exists (same hash), returns existing reference.
        This provides automatic deduplication.

        Args:
            data: Raw bytes to store
            content_type: MIME type

        Returns:
            WoolReference pointing to stored data

        Raises:
            IOError: If file write fails
        """
        # Compute hash
        file_id = self._compute_hash(data)
        path = self._get_file_path(file_id)

        # Check if already exists (deduplication)
        if path.exists():
            # Already stored, return existing reference
            self._stats['deduplications'] += 1
            logger.debug(f"Deduplication: {file_id[:8]}... already exists")
            return WoolReference(
                file_id=file_id,
                offset=0,
                length=len(data),
                path=path,
                content_type=content_type
            )

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to disk
        try:
            with open(path, 'wb') as f:
                f.write(data)

            self._stats['files_stored'] += 1
            self._stats['bytes_stored'] += len(data)

            logger.info(f"Stored {len(data)} bytes as {file_id[:8]}...")

        except IOError as e:
            logger.error(f"Failed to store file {file_id[:8]}...: {e}")
            raise

        # Return reference
        return WoolReference(
            file_id=file_id,
            offset=0,
            length=len(data),
            path=path,
            content_type=content_type
        )

    def _get_mmap(self, file_id: str) -> mmap.mmap:
        """
        Get memory-mapped file (cached).

        Thread-safe caching of mmap handles for performance.

        Args:
            file_id: SHA-256 hash

        Returns:
            mmap object

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self.enable_cache:
            # No caching, open fresh mmap
            path = self._get_file_path(file_id)
            if not path.exists():
                raise FileNotFoundError(f"Wool file not found: {file_id[:8]}...")

            f = open(path, 'rb')
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Check cache
        with self._cache_lock:
            if file_id in self._mmap_cache:
                self._stats['cache_hits'] += 1
                return self._mmap_cache[file_id]

            # Cache miss
            self._stats['cache_misses'] += 1

            # Open and mmap file
            path = self._get_file_path(file_id)
            if not path.exists():
                raise FileNotFoundError(f"Wool file not found: {file_id[:8]}...")

            f = open(path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Cache both file handle and mmap
            self._file_handles[file_id] = f
            self._mmap_cache[file_id] = mm

            logger.debug(f"Cached mmap for {file_id[:8]}...")
            return mm

    def read(self, ref: WoolReference) -> memoryview:
        """
        Read data from wool storage (zero-copy).

        Returns memoryview into mmap'd file.

        Args:
            ref: WoolReference to read

        Returns:
            memoryview (zero-copy view of data)

        Raises:
            FileNotFoundError: If referenced file doesn't exist
        """
        mm = self._get_mmap(ref.file_id)

        # Validate range
        if ref.offset + ref.length > len(mm):
            raise ValueError(
                f"Invalid range: offset={ref.offset}, length={ref.length}, "
                f"file_size={len(mm)}"
            )

        # Return memoryview slice (zero-copy!)
        return memoryview(mm)[ref.offset:ref.offset + ref.length]

    def read_range(
        self,
        file_id: str,
        offset: int,
        length: int
    ) -> memoryview:
        """
        Read byte range from file (zero-copy).

        Args:
            file_id: SHA-256 hash of file
            offset: Byte offset
            length: Number of bytes

        Returns:
            memoryview (zero-copy view)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If range is invalid
        """
        mm = self._get_mmap(file_id)

        # Validate range
        if offset + length > len(mm):
            raise ValueError(
                f"Invalid range: offset={offset}, length={length}, "
                f"file_size={len(mm)}"
            )

        return memoryview(mm)[offset:offset + length]

    def read_text(
        self,
        ref: WoolReference,
        encoding: str = 'utf-8'
    ) -> str:
        """
        Read text from wool storage (decodes on-demand).

        Args:
            ref: WoolReference to text data
            encoding: Text encoding (default: utf-8)

        Returns:
            Decoded string

        Raises:
            UnicodeDecodeError: If data is not valid text
        """
        data_view = self.read(ref)
        return data_view.tobytes().decode(encoding)

    def get_path(self, file_id: str) -> Path:
        """
        Get physical path for file_id.

        Args:
            file_id: SHA-256 hash

        Returns:
            Path to file (may not exist)
        """
        return self._get_file_path(file_id)

    def exists(self, file_id: str) -> bool:
        """
        Check if file exists in storage.

        Args:
            file_id: SHA-256 hash

        Returns:
            True if file exists
        """
        return self._get_file_path(file_id).exists()

    def get_size(self, file_id: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_id: SHA-256 hash

        Returns:
            Size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = self._get_file_path(file_id)
        if not path.exists():
            raise FileNotFoundError(f"Wool file not found: {file_id[:8]}...")

        return path.stat().st_size

    def get_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        total_files = self._stats['files_stored'] + self._stats['deduplications']
        cache_total = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = (
            self._stats['cache_hits'] / cache_total
            if cache_total > 0 else 0.0
        )

        return {
            'files_stored': self._stats['files_stored'],
            'bytes_stored': self._stats['bytes_stored'],
            'deduplications': self._stats['deduplications'],
            'total_requests': total_files,
            'dedup_rate': (
                self._stats['deduplications'] / total_files
                if total_files > 0 else 0.0
            ),
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'cached_files': len(self._mmap_cache)
        }

    def clear_cache(self):
        """
        Clear mmap cache (closes all mmap handles).

        Use this to free memory if needed.
        """
        with self._cache_lock:
            # Close all mmaps
            for mm in self._mmap_cache.values():
                try:
                    mm.close()
                except Exception as e:
                    logger.warning(f"Error closing mmap: {e}")

            # Close all file handles
            for f in self._file_handles.values():
                try:
                    f.close()
                except Exception as e:
                    logger.warning(f"Error closing file: {e}")

            self._mmap_cache.clear()
            self._file_handles.clear()

            logger.info("Cleared mmap cache")

    def close(self):
        """
        Close wool storage (closes all mmap handles).

        Always call this when done to release resources.
        """
        self.clear_cache()
        logger.info(f"Closed WoolStorage at {self.base_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (automatic cleanup)."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"WoolStorage(path={self.base_path}, "
            f"files={stats['files_stored']}, "
            f"bytes={stats['bytes_stored']}, "
            f"dedup_rate={stats['dedup_rate']:.1%})"
        )
