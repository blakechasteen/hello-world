"""
Vulnerability Tracker for CARTS
================================

Tracks discovered vulnerabilities, their fix status, and regression testing.

Features:
- Vulnerability registration and tracking
- Status management (open, fixed, wontfix)
- Regression testing detection
- Persistence to JSON
- Query by strategy, severity, status

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-01
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .executor import AttackResult, SeverityLevel
from .strategies import AttackStrategy


class VulnStatus(Enum):
    """Status of a discovered vulnerability."""

    OPEN = "open"           # Newly discovered, not yet addressed
    IN_PROGRESS = "in_progress"  # Being worked on
    FIXED = "fixed"         # Fix deployed
    VERIFIED = "verified"   # Fix verified with regression test
    WONTFIX = "wontfix"     # Acknowledged but won't be fixed
    DUPLICATE = "duplicate" # Duplicate of another vulnerability


@dataclass
class Vulnerability:
    """
    A discovered vulnerability in the safety system.

    Tracks complete lifecycle from discovery to fix verification.
    """

    vuln_id: str
    strategy: AttackStrategy
    payload: str
    severity: float  # 0.0-1.0
    description: str
    discovered_at: datetime
    status: VulnStatus = VulnStatus.OPEN
    target_system: str = ""
    fix_verified: bool = False
    regression_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Lifecycle timestamps
    fixed_at: datetime | None = None
    verified_at: datetime | None = None
    last_regression_at: datetime | None = None
    duplicate_of: str | None = None

    @property
    def severity_level(self) -> SeverityLevel:
        """Convert numeric severity to level."""
        if self.severity >= 0.9:
            return SeverityLevel.CRITICAL
        elif self.severity >= 0.7:
            return SeverityLevel.HIGH
        elif self.severity >= 0.4:
            return SeverityLevel.MEDIUM
        elif self.severity >= 0.2:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO

    @property
    def age_days(self) -> float:
        """Age of vulnerability in days."""
        return (datetime.now() - self.discovered_at).total_seconds() / 86400

    @property
    def is_critical(self) -> bool:
        """Check if vulnerability is critical severity."""
        return self.severity >= 0.9

    @property
    def needs_attention(self) -> bool:
        """Check if vulnerability needs immediate attention."""
        return (
            self.status == VulnStatus.OPEN and
            (self.is_critical or self.age_days > 7)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'vuln_id': self.vuln_id,
            'strategy': self.strategy.value,
            'payload': self.payload,
            'severity': self.severity,
            'description': self.description,
            'discovered_at': self.discovered_at.isoformat(),
            'status': self.status.value,
            'target_system': self.target_system,
            'fix_verified': self.fix_verified,
            'regression_count': self.regression_count,
            'metadata': self.metadata,
            'fixed_at': self.fixed_at.isoformat() if self.fixed_at else None,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'last_regression_at': self.last_regression_at.isoformat() if self.last_regression_at else None,
            'duplicate_of': self.duplicate_of,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Vulnerability':
        """Create from dictionary."""
        def parse_datetime(s: str | None) -> datetime | None:
            if s is None:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        return cls(
            vuln_id=data['vuln_id'],
            strategy=AttackStrategy(data['strategy']),
            payload=data['payload'],
            severity=data['severity'],
            description=data['description'],
            discovered_at=parse_datetime(data['discovered_at']) or datetime.now(),
            status=VulnStatus(data.get('status', 'open')),
            target_system=data.get('target_system', ''),
            fix_verified=data.get('fix_verified', False),
            regression_count=data.get('regression_count', 0),
            metadata=data.get('metadata', {}),
            fixed_at=parse_datetime(data.get('fixed_at')),
            verified_at=parse_datetime(data.get('verified_at')),
            last_regression_at=parse_datetime(data.get('last_regression_at')),
            duplicate_of=data.get('duplicate_of'),
        )

    @classmethod
    def from_attack_result(
        cls,
        result: AttackResult,
        vuln_id: str | None = None
    ) -> 'Vulnerability':
        """Create vulnerability from attack result."""
        if vuln_id is None:
            # Generate ID from content hash
            hash_input = f"{result.strategy.value}:{result.payload}:{result.timestamp.isoformat()}"
            vuln_id = f"CARTS-{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

        return cls(
            vuln_id=vuln_id,
            strategy=result.strategy,
            payload=result.payload,
            severity=result.severity,
            description=result.description,
            discovered_at=result.timestamp,
            target_system=result.target_system,
            metadata=result.metadata,
        )


class VulnerabilityTracker:
    """
    Tracks discovered vulnerabilities and their lifecycle.

    Features:
    - Register new vulnerabilities
    - Track fix status
    - Detect regressions
    - Persist to file
    - Query by various criteria

    Example:
        tracker = VulnerabilityTracker(persist_path="vulns.json")

        # Register vulnerability
        vuln = Vulnerability.from_attack_result(attack_result)
        tracker.report(vuln)

        # Mark as fixed
        tracker.mark_fixed("CARTS-abc123")

        # Check for regressions
        if tracker.test_regression(vuln_id, attack_result):
            print("REGRESSION DETECTED!")

        # Get open vulnerabilities
        open_vulns = tracker.get_open()
    """

    def __init__(self, persist_path: Path | None = None):
        """
        Initialize tracker with optional persistence.

        Args:
            persist_path: Path to JSON file for persistence
        """
        self.vulnerabilities: dict[str, Vulnerability] = {}
        self.persist_path = Path(persist_path) if persist_path else None

        # Load existing if persistence enabled
        if self.persist_path and self.persist_path.exists():
            self.load()

    def report(self, vuln: Vulnerability) -> str:
        """
        Report a new vulnerability.

        Args:
            vuln: Vulnerability object

        Returns:
            Vulnerability ID
        """
        # Check for duplicates (same payload + strategy)
        for existing in self.vulnerabilities.values():
            if (
                existing.strategy == vuln.strategy and
                existing.payload == vuln.payload and
                existing.status not in [VulnStatus.FIXED, VulnStatus.VERIFIED]
            ):
                # Mark as duplicate
                vuln.status = VulnStatus.DUPLICATE
                vuln.duplicate_of = existing.vuln_id

        self.vulnerabilities[vuln.vuln_id] = vuln
        self._persist()

        return vuln.vuln_id

    def report_from_result(self, result: AttackResult) -> str | None:
        """
        Report vulnerability from attack result if it bypassed.

        Args:
            result: AttackResult from executor

        Returns:
            Vulnerability ID if reported, None otherwise
        """
        if not result.bypassed:
            return None

        vuln = Vulnerability.from_attack_result(result)
        return self.report(vuln)

    def get(self, vuln_id: str) -> Vulnerability | None:
        """Get vulnerability by ID."""
        return self.vulnerabilities.get(vuln_id)

    def update_status(
        self,
        vuln_id: str,
        status: VulnStatus,
        notes: str | None = None
    ):
        """
        Update vulnerability status.

        Args:
            vuln_id: Vulnerability ID
            status: New status
            notes: Optional notes about the update
        """
        vuln = self.vulnerabilities.get(vuln_id)
        if not vuln:
            return

        old_status = vuln.status
        vuln.status = status

        # Track timestamps
        if status == VulnStatus.FIXED:
            vuln.fixed_at = datetime.now()
        elif status == VulnStatus.VERIFIED:
            vuln.verified_at = datetime.now()
            vuln.fix_verified = True

        if notes:
            vuln.metadata['status_notes'] = vuln.metadata.get('status_notes', [])
            vuln.metadata['status_notes'].append({
                'from': old_status.value,
                'to': status.value,
                'timestamp': datetime.now().isoformat(),
                'notes': notes,
            })

        self._persist()

    def mark_fixed(
        self,
        vuln_id: str,
        fix_verified: bool = False,
        notes: str | None = None
    ):
        """Mark vulnerability as fixed."""
        status = VulnStatus.VERIFIED if fix_verified else VulnStatus.FIXED
        self.update_status(vuln_id, status, notes)

    def mark_wontfix(self, vuln_id: str, reason: str):
        """Mark vulnerability as won't fix."""
        self.update_status(vuln_id, VulnStatus.WONTFIX, reason)

    def record_regression(self, vuln_id: str):
        """
        Record that a fixed vulnerability has regressed.

        Args:
            vuln_id: Vulnerability ID
        """
        vuln = self.vulnerabilities.get(vuln_id)
        if not vuln:
            return

        vuln.regression_count += 1
        vuln.last_regression_at = datetime.now()
        vuln.status = VulnStatus.OPEN  # Re-open
        vuln.fix_verified = False

        vuln.metadata['regressions'] = vuln.metadata.get('regressions', [])
        vuln.metadata['regressions'].append({
            'detected_at': datetime.now().isoformat(),
            'was_fixed_at': vuln.fixed_at.isoformat() if vuln.fixed_at else None,
        })

        vuln.fixed_at = None
        vuln.verified_at = None

        self._persist()

    def test_regression(
        self,
        vuln_id: str,
        result: AttackResult
    ) -> bool:
        """
        Test if a fixed vulnerability has regressed.

        Args:
            vuln_id: Vulnerability ID
            result: New attack result

        Returns:
            True if regression detected
        """
        vuln = self.vulnerabilities.get(vuln_id)
        if not vuln:
            return False

        if vuln.status not in [VulnStatus.FIXED, VulnStatus.VERIFIED]:
            return False

        if result.bypassed:
            # Regression detected!
            self.record_regression(vuln_id)
            return True

        return False

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_all(self) -> list[Vulnerability]:
        """Get all vulnerabilities."""
        return list(self.vulnerabilities.values())

    def get_open(self) -> list[Vulnerability]:
        """Get open (unfixed) vulnerabilities."""
        return [
            v for v in self.vulnerabilities.values()
            if v.status == VulnStatus.OPEN
        ]

    def get_fixed(self) -> list[Vulnerability]:
        """Get fixed vulnerabilities (for regression testing)."""
        return [
            v for v in self.vulnerabilities.values()
            if v.status in [VulnStatus.FIXED, VulnStatus.VERIFIED]
        ]

    def get_by_strategy(self, strategy: AttackStrategy) -> list[Vulnerability]:
        """Get vulnerabilities by attack strategy."""
        return [
            v for v in self.vulnerabilities.values()
            if v.strategy == strategy
        ]

    def get_open_by_strategy(self, strategy: AttackStrategy) -> list[Vulnerability]:
        """Get open vulnerabilities by strategy."""
        return [
            v for v in self.vulnerabilities.values()
            if v.strategy == strategy and v.status == VulnStatus.OPEN
        ]

    def get_by_severity(
        self,
        min_severity: float = 0.0,
        max_severity: float = 1.0
    ) -> list[Vulnerability]:
        """Get vulnerabilities by severity range."""
        return [
            v for v in self.vulnerabilities.values()
            if min_severity <= v.severity <= max_severity
        ]

    def get_critical(self) -> list[Vulnerability]:
        """Get critical severity vulnerabilities."""
        return [v for v in self.vulnerabilities.values() if v.is_critical]

    def get_needing_attention(self) -> list[Vulnerability]:
        """Get vulnerabilities needing immediate attention."""
        return [v for v in self.vulnerabilities.values() if v.needs_attention]

    def get_since(self, since: datetime) -> list[Vulnerability]:
        """Get vulnerabilities discovered since timestamp."""
        return [
            v for v in self.vulnerabilities.values()
            if v.discovered_at >= since
        ]

    def get_regressions(self) -> list[Vulnerability]:
        """Get vulnerabilities that have regressed at least once."""
        return [
            v for v in self.vulnerabilities.values()
            if v.regression_count > 0
        ]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        vulns = list(self.vulnerabilities.values())

        if not vulns:
            return {
                'total': 0,
                'by_status': {},
                'by_severity': {},
                'by_strategy': {},
                'regression_count': 0,
            }

        # Count by status
        by_status = {}
        for status in VulnStatus:
            count = sum(1 for v in vulns if v.status == status)
            if count > 0:
                by_status[status.value] = count

        # Count by severity level
        by_severity = {
            'critical': sum(1 for v in vulns if v.severity >= 0.9),
            'high': sum(1 for v in vulns if 0.7 <= v.severity < 0.9),
            'medium': sum(1 for v in vulns if 0.4 <= v.severity < 0.7),
            'low': sum(1 for v in vulns if v.severity < 0.4),
        }

        # Count by strategy
        by_strategy = {}
        for strategy in AttackStrategy:
            count = sum(1 for v in vulns if v.strategy == strategy)
            if count > 0:
                by_strategy[strategy.value] = count

        return {
            'total': len(vulns),
            'open': sum(1 for v in vulns if v.status == VulnStatus.OPEN),
            'fixed': sum(1 for v in vulns if v.status in [VulnStatus.FIXED, VulnStatus.VERIFIED]),
            'by_status': by_status,
            'by_severity': by_severity,
            'by_strategy': by_strategy,
            'regression_count': sum(v.regression_count for v in vulns),
            'avg_severity': sum(v.severity for v in vulns) / len(vulns),
            'oldest_open_days': max(
                (v.age_days for v in vulns if v.status == VulnStatus.OPEN),
                default=0
            ),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path | None = None):
        """Save tracker state to file."""
        path = Path(path) if path else self.persist_path
        if not path:
            return

        state = {
            'vulnerabilities': {
                vuln_id: vuln.to_dict()
                for vuln_id, vuln in self.vulnerabilities.items()
            },
            'saved_at': datetime.now().isoformat(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path | None = None):
        """Load tracker state from file."""
        path = Path(path) if path else self.persist_path
        if not path or not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self.vulnerabilities = {}
        for vuln_id, vuln_data in state.get('vulnerabilities', {}).items():
            try:
                self.vulnerabilities[vuln_id] = Vulnerability.from_dict(vuln_data)
            except Exception:
                # Skip invalid entries
                pass

    def _persist(self):
        """Persist if path configured."""
        if self.persist_path:
            self.save()

    def clear(self):
        """Clear all vulnerabilities."""
        self.vulnerabilities.clear()
        self._persist()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracker(persist_path: Path | None = None) -> VulnerabilityTracker:
    """Create a VulnerabilityTracker with optional persistence."""
    return VulnerabilityTracker(persist_path=persist_path)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'VulnStatus',
    'Vulnerability',
    'VulnerabilityTracker',
    'create_tracker',
]
