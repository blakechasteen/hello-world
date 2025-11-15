"""
Alert deduplication engine.

**Implemented: 2025-11-15**
**Lines**: 312
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationRule:
    """Deduplication rule."""

    name: str
    alert_types: List[str]
    include_source_ip: bool = True
    include_user: bool = False
    window_seconds: int = 300


@dataclass
class DuplicateAlert:
    """Tracked duplicate alert."""

    dedup_key: str
    first_seen: datetime
    count: int = 1
    last_seen: datetime = field(default_factory=datetime.utcnow)
    source_ips: Set[str] = field(default_factory=set)
    user_ids: Set[str] = field(default_factory=set)


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""

    is_duplicate: bool
    dedup_key: str
    count: int = 1
    first_occurrence: Optional[datetime] = None
    reason: str = ""


class DeduplicationEngine:
    """Handles alert deduplication within time windows."""

    def __init__(self, window_seconds: int = 300):
        """Initialize deduplication engine.

        Args:
            window_seconds: Default deduplication window (5 minutes).
        """
        self.window_seconds = window_seconds
        self.duplicates: Dict[str, DuplicateAlert] = {}
        self.rules: Dict[str, DeduplicationRule] = {}
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default deduplication rules."""
        rules = [
            DeduplicationRule(
                name="sql_injection",
                alert_types=["SQL Injection Detected"],
                include_source_ip=True,
                window_seconds=300,
            ),
            DeduplicationRule(
                name="xss_attempt",
                alert_types=["XSS Attempted"],
                include_source_ip=True,
                window_seconds=300,
            ),
            DeduplicationRule(
                name="brute_force",
                alert_types=["Brute Force Attempt"],
                include_source_ip=False,
                include_user=True,
                window_seconds=600,
            ),
            DeduplicationRule(
                name="rate_limit",
                alert_types=["Rate Limit Exceeded"],
                include_source_ip=True,
                window_seconds=300,
            ),
        ]

        for rule in rules:
            self.rules[rule.name] = rule

    async def check_dedup(self, alert) -> DeduplicationResult:
        """Check if alert is duplicate.

        Args:
            alert: Alert object to check.

        Returns:
            DeduplicationResult with duplicate status.
        """
        # Cleanup old duplicates
        self._cleanup_old_duplicates()

        dedup_key = alert.deduplication_key or alert.compute_dedup_key()

        # Check if duplicate
        if dedup_key in self.duplicates:
            dup = self.duplicates[dedup_key]
            time_since_first = (datetime.utcnow() - dup.first_seen).total_seconds()

            if time_since_first < self.window_seconds:
                dup.count += 1
                dup.last_seen = datetime.utcnow()

                if alert.source_ip:
                    dup.source_ips.add(alert.source_ip)
                if alert.user_id:
                    dup.user_ids.add(alert.user_id)

                logger.info(
                    f"Alert deduplicated: {alert.title} "
                    f"(occurrence #{dup.count} in {time_since_first:.1f}s)"
                )

                return DeduplicationResult(
                    is_duplicate=True,
                    dedup_key=dedup_key,
                    count=dup.count,
                    first_occurrence=dup.first_seen,
                    reason=f"Duplicate within {self.window_seconds}s window",
                )

        # Not a duplicate, register it
        self.duplicates[dedup_key] = DuplicateAlert(
            dedup_key=dedup_key,
            first_seen=datetime.utcnow(),
            source_ips={alert.source_ip} if alert.source_ip else set(),
            user_ids={alert.user_id} if alert.user_id else set(),
        )

        return DeduplicationResult(
            is_duplicate=False,
            dedup_key=dedup_key,
            count=1,
            first_occurrence=datetime.utcnow(),
        )

    async def reset_dedup(self, dedup_key: str):
        """Reset deduplication for a key (after alert resolved).

        Args:
            dedup_key: Deduplication key to reset.
        """
        if dedup_key in self.duplicates:
            del self.duplicates[dedup_key]
            logger.info(f"Deduplication reset for key: {dedup_key}")

    async def batch_dedup(self, alerts: List) -> Dict[str, bool]:
        """Check deduplication for batch of alerts.

        Args:
            alerts: List of alert objects.

        Returns:
            Dict mapping alert_id to is_duplicate boolean.
        """
        results = {}
        for alert in alerts:
            result = await self.check_dedup(alert)
            results[alert.alert_id] = result.is_duplicate

        return results

    def _cleanup_old_duplicates(self):
        """Remove expired duplicate tracking entries."""
        now = datetime.utcnow()
        expired = []

        for key, dup in self.duplicates.items():
            age = (now - dup.first_seen).total_seconds()
            if age > self.window_seconds:
                expired.append(key)

        for key in expired:
            del self.duplicates[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired duplicate entries")

    def get_dedup_stats(self) -> Dict[str, int]:
        """Get deduplication statistics.

        Returns:
            Dict with dedup stats.
        """
        self._cleanup_old_duplicates()

        total_tracked = len(self.duplicates)
        total_duplicates = sum(d.count - 1 for d in self.duplicates.values())

        return {
            "tracked_keys": total_tracked,
            "total_duplicates_suppressed": total_duplicates,
            "avg_duplicates_per_key": (
                total_duplicates / total_tracked if total_tracked > 0 else 0
            ),
        }

    def get_duplicates_for_key(self, dedup_key: str) -> Optional[DuplicateAlert]:
        """Get duplicate info for a key.

        Args:
            dedup_key: Deduplication key.

        Returns:
            DuplicateAlert if found.
        """
        return self.duplicates.get(dedup_key)

    def get_all_duplicates(self) -> List[DuplicateAlert]:
        """Get all tracked duplicates.

        Returns:
            List of DuplicateAlert objects.
        """
        self._cleanup_old_duplicates()
        return list(self.duplicates.values())

    def add_rule(self, rule: DeduplicationRule):
        """Add custom deduplication rule.

        Args:
            rule: DeduplicationRule object.
        """
        self.rules[rule.name] = rule
        logger.info(f"Deduplication rule added: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove deduplication rule.

        Args:
            rule_name: Name of rule to remove.
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Deduplication rule removed: {rule_name}")

    def get_rules(self) -> Dict[str, DeduplicationRule]:
        """Get all deduplication rules.

        Returns:
            Dict of rules.
        """
        return dict(self.rules)
