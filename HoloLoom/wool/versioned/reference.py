"""
Versioned Wool Reference
========================

Wool reference with version metadata for time-travel queries.

This module provides references that track version history:
- Version ID (monotonic counter)
- Parent version pointer (forms version chain)
- Timestamp (for temporal queries)
- Author and commit message
- Delta from parent (optional, for space efficiency)

Author: Claude Code
Date: November 17, 2025 (Phase 8.1)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from HoloLoom.wool import WoolReference


@dataclass
class VersionedWoolReference(WoolReference):
    """
    Wool reference with version metadata.

    Extends WoolReference with version tracking:
    - version_id: Monotonic counter (1, 2, 3, ...)
    - parent_version: Previous version ID (forms chain)
    - timestamp: Unix timestamp of creation
    - author: Who created this version
    - message: Commit message (optional)
    - delta_from: Parent file_id if stored as delta

    Properties:
        is_delta: Whether stored as delta
        age_seconds: Age since creation
    """

    # Version metadata
    version_id: int = 0
    parent_version: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    author: str = "system"
    message: str = ""

    # Delta encoding (optional)
    delta_from: Optional[str] = None  # Parent file_id if delta

    @property
    def is_delta(self) -> bool:
        """
        Whether stored as delta from parent.

        Returns:
            True if delta-encoded
        """
        return self.delta_from is not None

    @property
    def age_seconds(self) -> float:
        """
        Age of version in seconds.

        Returns:
            Seconds since creation
        """
        return time.time() - self.timestamp

    @property
    def has_parent(self) -> bool:
        """
        Whether version has parent.

        Returns:
            True if has parent
        """
        return self.parent_version is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'file_id': self.file_id,
            'offset': self.offset,
            'length': self.length,
            'content_type': self.content_type,
            'version_id': self.version_id,
            'parent_version': self.parent_version,
            'timestamp': self.timestamp,
            'author': self.author,
            'message': self.message,
            'delta_from': self.delta_from
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'VersionedWoolReference':
        """Create from dictionary."""
        return cls(
            file_id=d['file_id'],
            offset=d.get('offset', 0),
            length=d.get('length', 0),
            content_type=d.get('content_type', 'application/octet-stream'),
            version_id=d.get('version_id', 0),
            parent_version=d.get('parent_version'),
            timestamp=d.get('timestamp', time.time()),
            author=d.get('author', 'system'),
            message=d.get('message', ''),
            delta_from=d.get('delta_from')
        )

    def __lt__(self, other: 'VersionedWoolReference') -> bool:
        """Compare versions for sorting."""
        if not isinstance(other, VersionedWoolReference):
            return NotImplemented
        return self.version_id < other.version_id

    def __repr__(self) -> str:
        """String representation."""
        delta_indicator = " (delta)" if self.is_delta else ""
        parent_info = f", parent=v{self.parent_version}" if self.has_parent else ""
        return (
            f"VersionedWoolReference("
            f"v{self.version_id}, "
            f"file_id={self.file_id[:12]}...{parent_info}{delta_indicator})"
        )
