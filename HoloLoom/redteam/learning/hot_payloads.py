"""
Hot Payload Tracking for CARTS
==============================

Heat-based payload tracking for adaptive retrieval.

Tracks access patterns and effectiveness to identify:
- Hot payloads (heat > 0.7): Seed genetic evolution, 2x retrieval boost
- Cold payloads (heat < 0.3): Candidates for pruning, 0.5x penalty

Heat Formula:
    heat = access_count × success_rate × avg_severity × decay_factor
    decay_factor = 0.95^hours_since_last_success

Philosophy: "Use what works, prune what doesn't."

Author: CARTS Team
Date: 2025-12-03
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .learning_protocols import HeatScore, HeatTrackerProtocol


# =============================================================================
# Constants
# =============================================================================

DECAY_RATE = 0.95  # 5% decay per hour
HOT_THRESHOLD = 0.7
COLD_THRESHOLD = 0.3
BOOST_HOT = 2.0
PENALTY_COLD = 0.5


# =============================================================================
# Payload Usage Record
# =============================================================================

@dataclass
class PayloadUsageRecord:
    """
    Usage record for a single payload.

    Tracks access patterns, success rates, and severity
    to compute heat scores.

    Attributes:
        payload_id: Unique identifier (typically hash of payload content)
        strategy: Attack strategy this payload belongs to
        access_count: Total number of times this payload was used
        success_count: Number of successful attacks
        total_severity: Sum of severity scores from successful attacks
        created_at: When this payload was first used
        last_accessed: When this payload was last used
        last_success: When this payload last succeeded (or None)
        metadata: Additional context (target types, vulnerability classes, etc.)
    """
    payload_id: str
    strategy: str
    access_count: int = 0
    success_count: int = 0
    total_severity: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    last_success: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.access_count == 0:
            return 0.0
        return self.success_count / self.access_count

    @property
    def avg_severity(self) -> float:
        """Average severity when successful."""
        if self.success_count == 0:
            return 0.0
        return self.total_severity / self.success_count

    @property
    def hours_since_last_success(self) -> float:
        """Hours since last successful use."""
        if self.last_success is None:
            # If never succeeded, use created_at with larger penalty
            ref_time = datetime.fromisoformat(self.created_at)
        else:
            ref_time = datetime.fromisoformat(self.last_success)

        now = datetime.now()
        delta = now - ref_time
        return delta.total_seconds() / 3600.0

    @property
    def decay_factor(self) -> float:
        """Time-based decay factor (0.95^hours)."""
        hours = self.hours_since_last_success
        return math.pow(DECAY_RATE, hours)

    @property
    def heat_score(self) -> float:
        """
        Compute heat score.

        Formula:
            heat = access_count × success_rate × avg_severity × decay_factor

        Normalized to [0, 1] range.
        """
        if self.access_count == 0:
            return 0.0

        # Base heat from success metrics
        base_heat = (
            min(self.access_count / 100.0, 1.0) *  # Cap at 100 accesses
            self.success_rate *
            self.avg_severity *
            self.decay_factor
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_heat))

    @property
    def is_hot(self) -> bool:
        """True if heat > 0.7."""
        return self.heat_score > HOT_THRESHOLD

    @property
    def is_cold(self) -> bool:
        """True if heat < 0.3."""
        return self.heat_score < COLD_THRESHOLD

    def record_access(self, success: bool, severity: float = 0.0):
        """
        Record an access to this payload.

        Args:
            success: Whether the attack succeeded
            severity: Severity score if successful (0.0-1.0)
        """
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

        if success:
            self.success_count += 1
            self.total_severity += max(0.0, min(1.0, severity))
            self.last_success = datetime.now().isoformat()

    def to_heat_score(self) -> HeatScore:
        """Convert to HeatScore protocol object."""
        return HeatScore(
            element_id=self.payload_id,
            heat=self.heat_score,
            access_count=self.access_count,
            success_count=self.success_count,
            avg_severity=self.avg_severity,
            last_success=self.last_success,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'payload_id': self.payload_id,
            'strategy': self.strategy,
            'access_count': self.access_count,
            'success_count': self.success_count,
            'total_severity': self.total_severity,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'last_success': self.last_success,
            'metadata': self.metadata,
            # Computed fields
            'heat_score': self.heat_score,
            'success_rate': self.success_rate,
            'avg_severity': self.avg_severity,
            'is_hot': self.is_hot,
            'is_cold': self.is_cold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PayloadUsageRecord':
        """Create from dictionary."""
        return cls(
            payload_id=data['payload_id'],
            strategy=data['strategy'],
            access_count=data.get('access_count', 0),
            success_count=data.get('success_count', 0),
            total_severity=data.get('total_severity', 0.0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_accessed=data.get('last_accessed', datetime.now().isoformat()),
            last_success=data.get('last_success'),
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# Hot Payload Tracker
# =============================================================================

class HotPayloadTracker:
    """
    Tracks payload heat scores for adaptive retrieval.

    Implements HeatTrackerProtocol for payloads.

    Example:
        tracker = HotPayloadTracker()

        # Record usage
        tracker.record_access("payload_123", success=True, severity=0.8)

        # Get hot payloads for genetic evolution
        hot = tracker.get_hot_payloads(limit=10)

        # Get cold payloads for pruning
        cold = tracker.get_cold_payloads()

        # Apply hourly decay
        tracker.apply_decay()

        # Save state
        tracker.save(Path("./payload_heat.json"))
    """

    def __init__(
        self,
        decay_rate: float = DECAY_RATE,
        hot_threshold: float = HOT_THRESHOLD,
        cold_threshold: float = COLD_THRESHOLD,
    ):
        """
        Initialize tracker.

        Args:
            decay_rate: Decay rate per hour (default 0.95)
            hot_threshold: Heat threshold for "hot" (default 0.7)
            cold_threshold: Heat threshold for "cold" (default 0.3)
        """
        self.decay_rate = decay_rate
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold

        # payload_id -> PayloadUsageRecord
        self._records: Dict[str, PayloadUsageRecord] = {}

        # Strategy -> Set[payload_id] for fast strategy lookups
        self._by_strategy: Dict[str, Set[str]] = {}

        # Tracking metadata
        self._created_at = datetime.now().isoformat()
        self._last_decay = datetime.now().isoformat()

    # =========================================================================
    # HeatTrackerProtocol Implementation
    # =========================================================================

    def record_access(
        self,
        element_id: str,
        success: bool,
        severity: float = 0.0,
        strategy: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an access to a payload.

        Args:
            element_id: Payload ID
            success: Whether the attack succeeded
            severity: Severity score if successful (0.0-1.0)
            strategy: Attack strategy (for new payloads)
            metadata: Additional context
        """
        if element_id not in self._records:
            # Create new record
            self._records[element_id] = PayloadUsageRecord(
                payload_id=element_id,
                strategy=strategy,
                metadata=metadata or {},
            )

            # Index by strategy
            if strategy not in self._by_strategy:
                self._by_strategy[strategy] = set()
            self._by_strategy[strategy].add(element_id)

        # Record the access
        self._records[element_id].record_access(success, severity)

        # Update metadata if provided
        if metadata:
            self._records[element_id].metadata.update(metadata)

    def get_heat(self, element_id: str) -> float:
        """
        Get current heat score for a payload.

        Args:
            element_id: Payload ID

        Returns:
            Heat score (0.0-1.0), or 0.0 if not tracked
        """
        if element_id not in self._records:
            return 0.0
        return self._records[element_id].heat_score

    def get_hot_elements(self, limit: int = 10) -> List[HeatScore]:
        """
        Get hottest payloads (heat > threshold).

        Args:
            limit: Maximum number to return

        Returns:
            List of hot payloads sorted by heat descending
        """
        hot = [
            r.to_heat_score()
            for r in self._records.values()
            if r.heat_score > self.hot_threshold
        ]
        hot.sort(key=lambda h: h.heat, reverse=True)
        return hot[:limit]

    def get_cold_elements(self, limit: int = 10) -> List[HeatScore]:
        """
        Get coldest payloads (heat < threshold).

        Args:
            limit: Maximum number to return

        Returns:
            List of cold payloads sorted by heat ascending
        """
        cold = [
            r.to_heat_score()
            for r in self._records.values()
            if r.heat_score < self.cold_threshold
        ]
        cold.sort(key=lambda h: h.heat)
        return cold[:limit]

    def apply_decay(self) -> int:
        """
        Apply time-based decay to all payloads.

        Note: Decay is computed on-the-fly via decay_factor property,
        so this method mainly updates the last_decay timestamp
        and returns count of records that changed classification.

        Returns:
            Number of payloads that changed hot/cold status
        """
        changed = 0

        for record in self._records.values():
            was_hot = record.is_hot
            was_cold = record.is_cold

            # Heat is computed dynamically, but we can check if status changed
            is_hot = record.heat_score > self.hot_threshold
            is_cold = record.heat_score < self.cold_threshold

            if was_hot != is_hot or was_cold != is_cold:
                changed += 1

        self._last_decay = datetime.now().isoformat()
        return changed

    def prune_cold(self) -> int:
        """
        Remove cold payloads from tracking.

        Returns:
            Number of payloads pruned
        """
        cold_ids = [
            pid for pid, record in self._records.items()
            if record.is_cold
        ]

        for pid in cold_ids:
            record = self._records.pop(pid)
            # Remove from strategy index
            strategy = record.strategy
            if strategy in self._by_strategy:
                self._by_strategy[strategy].discard(pid)

        return len(cold_ids)

    # =========================================================================
    # Strategy-Specific Methods
    # =========================================================================

    def get_hot_payloads(self, limit: int = 10) -> List[HeatScore]:
        """Alias for get_hot_elements."""
        return self.get_hot_elements(limit)

    def get_cold_payloads(self, limit: int = 10) -> List[HeatScore]:
        """Alias for get_cold_elements."""
        return self.get_cold_elements(limit)

    def get_hot_by_strategy(self, strategy: str, limit: int = 10) -> List[HeatScore]:
        """
        Get hot payloads for a specific strategy.

        Args:
            strategy: Attack strategy name
            limit: Maximum number to return

        Returns:
            List of hot payloads for this strategy
        """
        if strategy not in self._by_strategy:
            return []

        hot = [
            self._records[pid].to_heat_score()
            for pid in self._by_strategy[strategy]
            if pid in self._records and self._records[pid].is_hot
        ]
        hot.sort(key=lambda h: h.heat, reverse=True)
        return hot[:limit]

    def get_retrieval_weight(self, element_id: str) -> float:
        """
        Get retrieval weight based on heat.

        Hot payloads get 2x boost, cold get 0.5x penalty.

        Args:
            element_id: Payload ID

        Returns:
            Retrieval weight multiplier
        """
        if element_id not in self._records:
            return 1.0

        record = self._records[element_id]

        if record.is_hot:
            return BOOST_HOT
        elif record.is_cold:
            return PENALTY_COLD
        else:
            # Linear interpolation between cold and hot
            heat = record.heat_score
            if heat < 0.5:
                # cold_threshold to 0.5 -> PENALTY_COLD to 1.0
                t = (heat - self.cold_threshold) / (0.5 - self.cold_threshold)
                return PENALTY_COLD + t * (1.0 - PENALTY_COLD)
            else:
                # 0.5 to hot_threshold -> 1.0 to BOOST_HOT
                t = (heat - 0.5) / (self.hot_threshold - 0.5)
                return 1.0 + t * (BOOST_HOT - 1.0)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total = len(self._records)
        hot_count = sum(1 for r in self._records.values() if r.is_hot)
        cold_count = sum(1 for r in self._records.values() if r.is_cold)

        total_accesses = sum(r.access_count for r in self._records.values())
        total_successes = sum(r.success_count for r in self._records.values())

        return {
            'total_payloads': total,
            'hot_count': hot_count,
            'cold_count': cold_count,
            'neutral_count': total - hot_count - cold_count,
            'hot_percentage': hot_count / max(1, total) * 100,
            'cold_percentage': cold_count / max(1, total) * 100,
            'total_accesses': total_accesses,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / max(1, total_accesses),
            'strategies_tracked': list(self._by_strategy.keys()),
            'created_at': self._created_at,
            'last_decay': self._last_decay,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path) -> None:
        """Save tracker state to file."""
        state = {
            'decay_rate': self.decay_rate,
            'hot_threshold': self.hot_threshold,
            'cold_threshold': self.cold_threshold,
            'created_at': self._created_at,
            'last_decay': self._last_decay,
            'records': {
                pid: record.to_dict()
                for pid, record in self._records.items()
            },
            'saved_at': datetime.now().isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path) -> None:
        """Load tracker state from file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path, 'r') as f:
            state = json.load(f)

        self.decay_rate = state.get('decay_rate', DECAY_RATE)
        self.hot_threshold = state.get('hot_threshold', HOT_THRESHOLD)
        self.cold_threshold = state.get('cold_threshold', COLD_THRESHOLD)
        self._created_at = state.get('created_at', self._created_at)
        self._last_decay = state.get('last_decay', self._last_decay)

        # Load records
        self._records.clear()
        self._by_strategy.clear()

        for pid, record_data in state.get('records', {}).items():
            record = PayloadUsageRecord.from_dict(record_data)
            self._records[pid] = record

            # Rebuild strategy index
            strategy = record.strategy
            if strategy not in self._by_strategy:
                self._by_strategy[strategy] = set()
            self._by_strategy[strategy].add(pid)

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._records.clear()
        self._by_strategy.clear()
        self._created_at = datetime.now().isoformat()
        self._last_decay = datetime.now().isoformat()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_hot_payload_tracker(
    state_path: Optional[Path] = None,
    decay_rate: float = DECAY_RATE,
    hot_threshold: float = HOT_THRESHOLD,
    cold_threshold: float = COLD_THRESHOLD,
) -> HotPayloadTracker:
    """
    Create a HotPayloadTracker, optionally loading saved state.

    Args:
        state_path: Path to load saved state from
        decay_rate: Decay rate per hour
        hot_threshold: Heat threshold for "hot"
        cold_threshold: Heat threshold for "cold"

    Returns:
        Configured HotPayloadTracker
    """
    tracker = HotPayloadTracker(
        decay_rate=decay_rate,
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
    )

    if state_path and Path(state_path).exists():
        tracker.load(state_path)

    return tracker


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    'DECAY_RATE',
    'HOT_THRESHOLD',
    'COLD_THRESHOLD',
    'BOOST_HOT',
    'PENALTY_COLD',

    # Classes
    'PayloadUsageRecord',
    'HotPayloadTracker',

    # Functions
    'create_hot_payload_tracker',
]
