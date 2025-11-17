"""
Time-Travel Versioned Storage
==============================

Complete version history and temporal queries for Wool Storage.

Key Features:
- Immutable append-only log (every write creates new version)
- Version chains with parent pointers
- Temporal indexing for efficient time-range queries
- Point-in-time queries (as_of timestamp)
- Delta encoding for storage efficiency (5-20x savings)
- Branch/merge support (like git)

Usage:
    from HoloLoom.wool.versioned import VersionedWoolStorage

    wool = VersionedWoolStorage(base_path="./data/wool")

    # v1: Initial write
    ref_v1 = wool.store("Hello")

    # v2: Update (creates new version, v1 preserved!)
    ref_v2 = wool.store("Hello, World!", parent=ref_v1)

    # Time-travel query
    ref_past = wool.as_of(timestamp=yesterday)
    data = wool.read(ref_past)

Exports:
- VersionedWoolReference
- TemporalIndex
- VersionedWoolStorage

Author: Claude Code
Date: November 17, 2025 (Phase 8)
"""

from HoloLoom.wool.versioned.reference import VersionedWoolReference
from HoloLoom.wool.versioned.index import TemporalIndex
from HoloLoom.wool.versioned.storage import VersionedWoolStorage

__all__ = [
    'VersionedWoolReference',
    'TemporalIndex',
    'VersionedWoolStorage'
]
