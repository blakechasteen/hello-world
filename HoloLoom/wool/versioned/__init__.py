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

    wool = VersionedWoolStorage(
        base_path="./data/wool",
        enable_delta_encoding=True
    )

    # v1: Initial write (snapshot)
    ref_v1 = wool.store_versioned(b"Hello", message="Initial")

    # v2: Update (stored as delta!)
    ref_v2 = wool.store_versioned(
        b"Hello, World!",
        parent=ref_v1,
        message="Add greeting"
    )

    # Time-travel query
    ref_past = wool.as_of(file_id, timestamp=yesterday)
    data = wool.reconstruct_version(ref_past)

Exports:
- VersionedWoolReference
- TemporalIndex
- VersionedWoolStorage
- DeltaAlgorithm
- compute_delta, apply_delta

Author: Claude Code
Date: November 17, 2025 (Phase 8 + Delta Encoding)
"""

from HoloLoom.wool.versioned.reference import VersionedWoolReference
from HoloLoom.wool.versioned.index import TemporalIndex
from HoloLoom.wool.versioned.storage import VersionedWoolStorage
from HoloLoom.wool.versioned.delta import (
    DeltaAlgorithm,
    compute_delta,
    apply_delta
)
from HoloLoom.wool.versioned.branch import (
    BranchManager,
    Branch,
    MergeStrategy,
    MergeResult,
    MergeConflict
)

__all__ = [
    'VersionedWoolReference',
    'TemporalIndex',
    'VersionedWoolStorage',
    'DeltaAlgorithm',
    'compute_delta',
    'apply_delta',
    'BranchManager',
    'Branch',
    'MergeStrategy',
    'MergeResult',
    'MergeConflict'
]
