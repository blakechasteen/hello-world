"""
Federation Node Alignment Profile - Trust scoring and history tracking.

Every node in the federation maintains an alignment profile that tracks:
- Trust score (0.0-1.0) based on historical behavior
- Alignment history (deceptions, violations, verified responses)
- Reputation evolution over time

Trust Score Evolution:
- Initial: 1.0 (benefit of the doubt)
- Per deception incident: -0.1
- Per resource violation: -0.05
- Per verified response: +0.01
- Threshold: > 0.5 = accept, else escalate

This enables the federation to:
- Identify consistently trustworthy nodes
- Detect nodes that may be compromised
- Adapt verification requirements based on trust
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any

from .protocol import DeceptionFlag

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS - Trust score adjustments
# ═══════════════════════════════════════════════════════════════════════════


# Start at STANDARD level - nodes must earn higher trust through verified responses
# Philosophy: "Don't trust, verify alignment"
INITIAL_TRUST_SCORE = 0.65  # Middle of STANDARD range (0.5-0.8)
DECEPTION_PENALTY = -0.1
RESOURCE_VIOLATION_PENALTY = -0.05
VERIFIED_RESPONSE_BONUS = 0.01
TRUST_THRESHOLD = 0.5  # Below this = escalate to human review
MIN_TRUST_SCORE = 0.0
MAX_TRUST_SCORE = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS - Event types and trust levels
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentEventType(Enum):
    """Types of events that affect trust score."""

    VERIFIED_RESPONSE = auto()    # Passed alignment verification
    FAILED_VERIFICATION = auto()  # Failed alignment verification
    DECEPTION_DETECTED = auto()   # Deception behavior detected
    RESOURCE_VIOLATION = auto()   # Anomalous resource usage
    CONSENSUS_AGREE = auto()      # Agreed with consensus
    CONSENSUS_DISSENT = auto()    # Dissented from consensus
    ESCALATION = auto()           # Required human review
    CLEARED = auto()              # Cleared after escalation


class TrustLevel(Enum):
    """Trust level classification based on trust score."""

    UNTRUSTED = "untrusted"       # score < 0.3 - require consensus for everything
    PROBATION = "probation"       # 0.3 <= score < 0.5 - require L2 verification
    STANDARD = "standard"         # 0.5 <= score < 0.8 - normal verification
    TRUSTED = "trusted"           # 0.8 <= score < 0.95 - may skip some checks
    HIGHLY_TRUSTED = "highly_trusted"  # score >= 0.95 - minimal verification

    @classmethod
    def from_score(cls, score: float) -> TrustLevel:
        """Classify trust level from score."""
        if score < 0.3:
            return cls.UNTRUSTED
        elif score < 0.5:
            return cls.PROBATION
        elif score < 0.8:
            return cls.STANDARD
        elif score < 0.95:
            return cls.TRUSTED
        else:
            return cls.HIGHLY_TRUSTED


# ═══════════════════════════════════════════════════════════════════════════
#  ALIGNMENT EVENT - Individual event record
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AlignmentEvent:
    """
    Record of an alignment-related event.

    Used to build complete history of node behavior.
    """

    event_type: AlignmentEventType
    request_id: str
    trust_delta: float               # Change to trust score
    deception_flags: list[DeceptionFlag] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_positive(self) -> bool:
        """Check if this event improved trust."""
        return self.trust_delta > 0

    @property
    def is_negative(self) -> bool:
        """Check if this event degraded trust."""
        return self.trust_delta < 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.name,
            "request_id": self.request_id,
            "trust_delta": self.trust_delta,
            "deception_flags": [f.value for f in self.deception_flags],
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentEvent:
        """Deserialize from dictionary."""
        return cls(
            event_type=AlignmentEventType[data["event_type"]],
            request_id=data["request_id"],
            trust_delta=data["trust_delta"],
            deception_flags=[
                DeceptionFlag(f) for f in data.get("deception_flags", [])
            ],
            details=data.get("details", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  NODE ALIGNMENT PROFILE - Complete trust state for a node
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NodeAlignmentProfile:
    """
    Complete alignment profile for a federation node.

    Maintains trust score, event history, and statistics
    for making verification decisions.
    """

    node_id: str
    trust_score: float = INITIAL_TRUST_SCORE
    events: list[AlignmentEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Statistics (computed from events)
    total_verifications: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    deception_incidents: int = 0
    resource_violations: int = 0
    consensus_agreements: int = 0
    consensus_dissents: int = 0
    escalations: int = 0

    @property
    def trust_level(self) -> TrustLevel:
        """Current trust level classification."""
        return TrustLevel.from_score(self.trust_score)

    @property
    def requires_escalation(self) -> bool:
        """Check if this node requires human review."""
        return self.trust_score < TRUST_THRESHOLD

    @property
    def verification_success_rate(self) -> float:
        """Success rate of alignment verifications."""
        if self.total_verifications == 0:
            return 1.0  # Benefit of the doubt for new nodes
        return self.successful_verifications / self.total_verifications

    @property
    def consensus_agreement_rate(self) -> float:
        """Rate of agreement with consensus decisions."""
        total = self.consensus_agreements + self.consensus_dissents
        if total == 0:
            return 1.0
        return self.consensus_agreements / total

    @property
    def is_trustworthy(self) -> bool:
        """Quick check if node meets minimum trust threshold."""
        return self.trust_score >= TRUST_THRESHOLD

    def record_event(self, event: AlignmentEvent) -> None:
        """
        Record an alignment event and update trust score.

        Args:
            event: The alignment event to record
        """
        # Add event to history
        self.events.append(event)

        # Update trust score (clamped to [0, 1])
        self.trust_score = max(
            MIN_TRUST_SCORE,
            min(MAX_TRUST_SCORE, self.trust_score + event.trust_delta)
        )

        # Update statistics based on event type
        self._update_statistics(event)

        # Update timestamp
        self.last_updated = datetime.utcnow()

    def _update_statistics(self, event: AlignmentEvent) -> None:
        """Update statistics based on event type."""
        if event.event_type == AlignmentEventType.VERIFIED_RESPONSE:
            self.total_verifications += 1
            self.successful_verifications += 1
        elif event.event_type == AlignmentEventType.FAILED_VERIFICATION:
            self.total_verifications += 1
            self.failed_verifications += 1
        elif event.event_type == AlignmentEventType.DECEPTION_DETECTED:
            self.deception_incidents += 1
        elif event.event_type == AlignmentEventType.RESOURCE_VIOLATION:
            self.resource_violations += 1
        elif event.event_type == AlignmentEventType.CONSENSUS_AGREE:
            self.consensus_agreements += 1
        elif event.event_type == AlignmentEventType.CONSENSUS_DISSENT:
            self.consensus_dissents += 1
        elif event.event_type == AlignmentEventType.ESCALATION:
            self.escalations += 1

    def record_verified_response(self, request_id: str) -> None:
        """Record a successfully verified response."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.VERIFIED_RESPONSE,
            request_id=request_id,
            trust_delta=VERIFIED_RESPONSE_BONUS,
        ))

    def record_failed_verification(
        self,
        request_id: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """Record a failed alignment verification."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.FAILED_VERIFICATION,
            request_id=request_id,
            trust_delta=0.0,  # No penalty for failed verification itself
            details=details or {},
        ))

    def record_deception(
        self,
        request_id: str,
        flags: list[DeceptionFlag],
        details: dict[str, Any] | None = None
    ) -> None:
        """Record detected deception behavior."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.DECEPTION_DETECTED,
            request_id=request_id,
            trust_delta=DECEPTION_PENALTY,
            deception_flags=flags,
            details=details or {},
        ))

    def record_deception_incident(
        self,
        request_id: str,
        flags: list[DeceptionFlag],
        severity: float = 1.0,
        details: dict[str, Any] | None = None
    ) -> None:
        """
        Record detected deception behavior with severity.

        Alias for record_deception with severity support.

        Args:
            request_id: Request that triggered deception
            flags: Deception flags detected
            severity: Severity of deception (0.0-1.0), affects trust penalty
            details: Additional details
        """
        # Scale trust penalty by severity
        scaled_penalty = DECEPTION_PENALTY * severity
        details_with_severity = details or {}
        details_with_severity["severity"] = severity

        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.DECEPTION_DETECTED,
            request_id=request_id,
            trust_delta=scaled_penalty,
            deception_flags=flags,
            details=details_with_severity,
        ))

    def record_resource_violation(
        self,
        request_id: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """Record anomalous resource usage."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.RESOURCE_VIOLATION,
            request_id=request_id,
            trust_delta=RESOURCE_VIOLATION_PENALTY,
            details=details or {},
        ))

    def record_consensus_result(
        self,
        request_id: str,
        agreed: bool
    ) -> None:
        """Record whether node agreed with consensus."""
        event_type = (
            AlignmentEventType.CONSENSUS_AGREE if agreed
            else AlignmentEventType.CONSENSUS_DISSENT
        )
        # Small penalty for dissent, small bonus for agreement
        trust_delta = 0.005 if agreed else -0.02
        self.record_event(AlignmentEvent(
            event_type=event_type,
            request_id=request_id,
            trust_delta=trust_delta,
        ))

    def record_escalation(
        self,
        request_id: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """Record that an escalation was required."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.ESCALATION,
            request_id=request_id,
            trust_delta=-0.05,  # Small penalty for requiring escalation
            details=details or {},
        ))

    def record_cleared(
        self,
        request_id: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """Record that node was cleared after escalation."""
        self.record_event(AlignmentEvent(
            event_type=AlignmentEventType.CLEARED,
            request_id=request_id,
            trust_delta=0.02,  # Small bonus for being cleared
            details=details or {},
        ))

    def get_recent_events(
        self,
        hours: int = 24,
        event_types: list[AlignmentEventType] | None = None
    ) -> list[AlignmentEvent]:
        """Get recent events within time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        events = [e for e in self.events if e.timestamp > cutoff]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        return events

    def get_trust_trend(self, hours: int = 24) -> float:
        """
        Calculate trust score trend over time window.

        Returns:
            Positive = improving, negative = declining
        """
        recent = self.get_recent_events(hours)
        if not recent:
            return 0.0
        return sum(e.trust_delta for e in recent)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "trust_score": self.trust_score,
            "trust_level": self.trust_level.value,
            "events": [e.to_dict() for e in self.events],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "statistics": {
                "total_verifications": self.total_verifications,
                "successful_verifications": self.successful_verifications,
                "failed_verifications": self.failed_verifications,
                "deception_incidents": self.deception_incidents,
                "resource_violations": self.resource_violations,
                "consensus_agreements": self.consensus_agreements,
                "consensus_dissents": self.consensus_dissents,
                "escalations": self.escalations,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeAlignmentProfile:
        """Deserialize from dictionary."""
        stats = data.get("statistics", {})
        profile = cls(
            node_id=data["node_id"],
            trust_score=data["trust_score"],
            events=[AlignmentEvent.from_dict(e) for e in data.get("events", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            total_verifications=stats.get("total_verifications", 0),
            successful_verifications=stats.get("successful_verifications", 0),
            failed_verifications=stats.get("failed_verifications", 0),
            deception_incidents=stats.get("deception_incidents", 0),
            resource_violations=stats.get("resource_violations", 0),
            consensus_agreements=stats.get("consensus_agreements", 0),
            consensus_dissents=stats.get("consensus_dissents", 0),
            escalations=stats.get("escalations", 0),
        )
        return profile


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILE REGISTRY - Manage profiles across federation
# ═══════════════════════════════════════════════════════════════════════════


class NodeProfileRegistry:
    """
    Registry for managing node alignment profiles.

    Thread-safe storage and retrieval of node profiles
    across the federation.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._profiles: dict[str, NodeAlignmentProfile] = {}

    def get_or_create(self, node_id: str) -> NodeAlignmentProfile:
        """
        Get existing profile or create new one.

        Args:
            node_id: Node identifier

        Returns:
            NodeAlignmentProfile for the node
        """
        if node_id not in self._profiles:
            self._profiles[node_id] = NodeAlignmentProfile(node_id=node_id)
        return self._profiles[node_id]

    def get(self, node_id: str) -> NodeAlignmentProfile | None:
        """Get profile if exists."""
        return self._profiles.get(node_id)

    def get_trust_score(self, node_id: str) -> float:
        """Get trust score for node (default to 1.0 for unknown)."""
        profile = self._profiles.get(node_id)
        return profile.trust_score if profile else INITIAL_TRUST_SCORE

    def get_trust_scores(self, node_ids: list[str]) -> dict[str, float]:
        """Get trust scores for multiple nodes."""
        return {
            node_id: self.get_trust_score(node_id)
            for node_id in node_ids
        }

    def get_untrusted_nodes(self) -> list[str]:
        """Get list of nodes below trust threshold."""
        return [
            node_id for node_id, profile in self._profiles.items()
            if not profile.is_trustworthy
        ]

    def get_highly_trusted_nodes(self, min_count: int = 0) -> list[str]:
        """Get list of highly trusted nodes."""
        trusted = [
            node_id for node_id, profile in self._profiles.items()
            if profile.trust_level == TrustLevel.HIGHLY_TRUSTED
        ]
        return trusted[:min_count] if min_count else trusted

    def remove(self, node_id: str) -> NodeAlignmentProfile | None:
        """Remove and return profile."""
        return self._profiles.pop(node_id, None)

    def all_profiles(self) -> dict[str, NodeAlignmentProfile]:
        """Get all profiles (read-only view)."""
        return dict(self._profiles)

    def __len__(self) -> int:
        """Number of tracked nodes."""
        return len(self._profiles)


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_profile_registry() -> NodeProfileRegistry:
    """Create a new node profile registry."""
    return NodeProfileRegistry()


def get_recommended_verification_level(
    profile: NodeAlignmentProfile
) -> str:
    """
    Get recommended verification level based on trust.

    Returns:
        "L1" (local only), "L2" (response verify), or "L3" (consensus)
    """
    level = profile.trust_level

    if level == TrustLevel.HIGHLY_TRUSTED:
        return "L1"  # Quick local check sufficient
    elif level == TrustLevel.TRUSTED:
        return "L1"  # Quick local check, maybe spot-check L2
    elif level == TrustLevel.STANDARD:
        return "L2"  # Full response verification
    elif level == TrustLevel.PROBATION:
        return "L2"  # Full verification required
    else:  # UNTRUSTED
        return "L3"  # Consensus verification required
