"""
HoloLoom Wool Storage
=====================

Content-addressable storage layer for zero-copy data ingestion.

Philosophy:
    "Before spinning wool into yarn, we store the raw wool."

The Wool layer provides:
- Content-addressable storage (CAS) by SHA-256 hash
- Memory-mapped file access for zero-copy reads
- Automatic deduplication (same hash = same file)
- Immutable, append-only storage
- Thread-safe mmap caching

Architecture:
    Raw Data → Wool Storage (hash, mmap)
        ↓ WoolReference
    Spinner (memoryview)
        ↓ TextReference
    Graph (reference storage)
        ↓ Query
    Lazy Resolution (mmap read)

Author: Claude Code
Date: November 17, 2025
"""

from .storage import WoolStorage, WoolReference
from .text_reference import TextReference
from .zerocopy_shard import ZeroCopyMemoryShard

__all__ = [
    'WoolStorage',
    'WoolReference',
    'TextReference',
    'ZeroCopyMemoryShard',
]
