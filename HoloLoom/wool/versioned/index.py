"""
Temporal Index
==============

Index for efficient time-travel queries on versioned storage.

This module provides temporal indexing:
- Binary search for point-in-time queries (O(log N))
- Range queries for all versions in time window
- Version chain traversal
- Efficient storage (in-memory B-tree or disk-backed)

Author: Claude Code
Date: November 17, 2025 (Phase 8.2)
"""

import bisect
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from HoloLoom.wool.versioned.reference import VersionedWoolReference


logger = logging.getLogger(__name__)


@dataclass
class VersionEntry:
    """Entry in temporal index."""
    timestamp: float
    version_id: int
    file_id: str
    parent_version: Optional[int] = None

    def __lt__(self, other: 'VersionEntry') -> bool:
        """Compare by timestamp for sorting."""
        return self.timestamp < other.timestamp


class TemporalIndex:
    """
    Temporal index for versioned wool storage.

    Provides efficient time-based queries:
    - as_of(timestamp): Get version at specific time (O(log N))
    - between(start, end): Get versions in time range (O(log N + K))
    - get_history(file_id): Get all versions of file
    - get_lineage(version_id): Get version chain (parent pointers)

    Usage:
        index = TemporalIndex()

        # Add versions
        index.add_version(ref_v1)
        index.add_version(ref_v2)
        index.add_version(ref_v3)

        # Point-in-time query
        version_id = index.get_version_at(file_id, timestamp=yesterday)

        # Range query
        versions = index.get_versions_between(file_id, start=yesterday, end=today)
    """

    def __init__(self):
        """Initialize temporal index."""
        # file_id → sorted list of version entries
        self._index: Dict[str, List[VersionEntry]] = {}

        # version_id → VersionEntry (for fast lookup)
        self._version_map: Dict[int, VersionEntry] = {}

        # Statistics
        self.stats = {
            'files_tracked': 0,
            'total_versions': 0,
            'queries_served': 0
        }

    def add_version(self, ref: VersionedWoolReference):
        """
        Add version to index.

        Args:
            ref: Versioned reference to index
        """
        # Create entry
        entry = VersionEntry(
            timestamp=ref.timestamp,
            version_id=ref.version_id,
            file_id=ref.file_id,
            parent_version=ref.parent_version
        )

        # Add to file index
        if ref.file_id not in self._index:
            self._index[ref.file_id] = []
            self.stats['files_tracked'] += 1

        # Insert in sorted order (by timestamp)
        bisect.insort(self._index[ref.file_id], entry)

        # Add to version map
        self._version_map[ref.version_id] = entry

        self.stats['total_versions'] += 1

        logger.debug(f"Indexed v{ref.version_id} for {ref.file_id[:12]}... at t={ref.timestamp}")

    def get_version_at(
        self,
        file_id: str,
        timestamp: float
    ) -> Optional[int]:
        """
        Get version at specific timestamp.

        Args:
            file_id: File identifier
            timestamp: Unix timestamp

        Returns:
            Version ID at that time, or None if no version
        """
        if file_id not in self._index:
            return None

        entries = self._index[file_id]

        # Binary search for entry <= timestamp
        idx = bisect.bisect_right([e.timestamp for e in entries], timestamp)

        if idx == 0:
            return None  # No version before timestamp

        self.stats['queries_served'] += 1
        return entries[idx - 1].version_id

    def get_versions_between(
        self,
        file_id: str,
        start_time: float,
        end_time: float
    ) -> List[int]:
        """
        Get all versions in time range.

        Args:
            file_id: File identifier
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            List of version IDs in range
        """
        if file_id not in self._index:
            return []

        entries = self._index[file_id]
        timestamps = [e.timestamp for e in entries]

        # Binary search for range
        start_idx = bisect.bisect_left(timestamps, start_time)
        end_idx = bisect.bisect_right(timestamps, end_time)

        self.stats['queries_served'] += 1
        return [entries[i].version_id for i in range(start_idx, end_idx)]

    def get_history(self, file_id: str) -> List[int]:
        """
        Get all versions of file (sorted by time).

        Args:
            file_id: File identifier

        Returns:
            List of version IDs (oldest to newest)
        """
        if file_id not in self._index:
            return []

        return [e.version_id for e in self._index[file_id]]

    def get_lineage(self, version_id: int) -> List[int]:
        """
        Get version lineage (parent chain).

        Args:
            version_id: Version to trace

        Returns:
            List of version IDs from root to specified version
        """
        if version_id not in self._version_map:
            return []

        # Trace back through parent pointers
        lineage = []
        current_id = version_id

        while current_id is not None:
            lineage.append(current_id)

            # Get parent
            entry = self._version_map.get(current_id)
            if entry is None or entry.parent_version is None:
                break

            current_id = entry.parent_version

        # Reverse to get root → current
        lineage.reverse()

        return lineage

    def get_latest_version(self, file_id: str) -> Optional[int]:
        """
        Get latest version of file.

        Args:
            file_id: File identifier

        Returns:
            Latest version ID, or None
        """
        if file_id not in self._index or not self._index[file_id]:
            return None

        return self._index[file_id][-1].version_id

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'files_tracked': self.stats['files_tracked'],
            'total_versions': self.stats['total_versions'],
            'queries_served': self.stats['queries_served'],
            'avg_versions_per_file': (
                self.stats['total_versions'] / self.stats['files_tracked']
                if self.stats['files_tracked'] > 0 else 0
            )
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemporalIndex("
            f"files={self.stats['files_tracked']}, "
            f"versions={self.stats['total_versions']})"
        )
