"""
The Court of Threads - Conflict Resolution System

When conflicting regressions occur or multiple agents tug strands
in incompatible directions, the loom convenes The Court of Threads.

This is symbolic arbitration:
- Agents present arguments derived from commit messages, tension, and motions
- The Court evaluates conflicts
- Verdicts are issued with symbolic weight

The Court transforms debugging into narrative.
Instead of "merge conflict in F002", you get:
"Two threads cross and resist the loom. The Court must arbitrate."

Usage:
    from hololoom.domain_harness.strands.loom_court import (
        CourtOfThreads, Conflict, Verdict
    )
    
    court = CourtOfThreads()
    conflicts = court.detect_conflicts(events)
    for conflict in conflicts:
        verdict = court.arbitrate(conflict)
        print(verdict.proclamation)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from hololoom.domain_harness.strands.tension_model import WeaveMotion

# ============================================================================
# Conflict Types
# ============================================================================

class ConflictType(str, Enum):
    """Types of conflicts the Court can arbitrate."""
    REGRESSION_CLASH = "regression_clash"       # Multiple regressions on same feature
    TENSION_DEADLOCK = "tension_deadlock"       # Tension oscillating without resolution
    DEPENDENCY_CYCLE = "dependency_cycle"       # Circular dependencies
    MOTION_CONTRADICTION = "motion_contradiction"  # Conflicting weave motions
    WORKER_DISPUTE = "worker_dispute"           # Multiple workers claim same feature


class VerdictType(str, Enum):
    """Types of Court verdicts."""
    ARBITRATION = "arbitration"     # Court makes a decision
    DEFERRAL = "deferral"           # More information needed
    DISSOLUTION = "dissolution"     # Feature should be removed
    SPLIT = "split"                 # Feature should be split
    MERGE = "merge"                 # Features should be merged
    RITUAL = "ritual"               # Symbolic action required


# ============================================================================
# Court Data Structures
# ============================================================================

@dataclass
class CourtEvent:
    """An event submitted to the Court."""
    feature_id: str
    timestamp: datetime
    motion: WeaveMotion | None = None
    worker_id: str | None = None
    tension_delta: float = 0.0
    notes: str = ""
    commit_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "timestamp": self.timestamp.isoformat(),
            "motion": self.motion.value if self.motion else None,
            "worker_id": self.worker_id,
            "tension_delta": self.tension_delta,
            "notes": self.notes,
            "commit_hash": self.commit_hash
        }


@dataclass
class Conflict:
    """A detected conflict requiring arbitration."""
    conflict_id: str
    conflict_type: ConflictType
    feature_ids: list[str]
    events: list[CourtEvent]
    severity: int  # 1-10
    symbolic_description: str
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "feature_ids": self.feature_ids,
            "events": [e.to_dict() for e in self.events],
            "severity": self.severity,
            "symbolic_description": self.symbolic_description,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class Verdict:
    """Court verdict on a conflict."""
    conflict_id: str
    verdict_type: VerdictType
    ruling: str  # Concrete action to take
    proclamation: str  # Symbolic/narrative version
    affected_features: list[str]
    recommended_actions: list[str]
    issued_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "verdict_type": self.verdict_type.value,
            "ruling": self.ruling,
            "proclamation": self.proclamation,
            "affected_features": self.affected_features,
            "recommended_actions": self.recommended_actions,
            "issued_at": self.issued_at.isoformat()
        }


# ============================================================================
# Symbolic Proclamations
# ============================================================================

CONFLICT_SYMBOLS = {
    ConflictType.REGRESSION_CLASH: "⚔️ Two threads cross and resist the loom.",
    ConflictType.TENSION_DEADLOCK: "🔄 The strand oscillates endlessly, finding no rest.",
    ConflictType.DEPENDENCY_CYCLE: "🌀 A serpent eating its tail. The threads form a closed loop.",
    ConflictType.MOTION_CONTRADICTION: "↔️ Forces pull in opposite directions. The weave strains.",
    ConflictType.WORKER_DISPUTE: "👥 Multiple weavers claim the same thread."
}

VERDICT_PROCLAMATIONS = {
    VerdictType.ARBITRATION: [
        "The Court has spoken. The thread bends to the verdict.",
        "By the authority of the loom, this conflict is resolved.",
        "The weave accepts the ruling. Order is restored."
    ],
    VerdictType.DEFERRAL: [
        "The Court withholds judgment. More must be revealed.",
        "The threads remain tangled. Further evidence required.",
        "Patience. The pattern is not yet clear."
    ],
    VerdictType.DISSOLUTION: [
        "The thread is cut. What cannot be woven, must be released.",
        "The Court decrees dissolution. The strand fades from the weave.",
        "Some threads serve by their absence. This one is dissolved."
    ],
    VerdictType.SPLIT: [
        "One thread becomes two. The burden is divided.",
        "The Court orders bifurcation. Complexity demands separation.",
        "Split the strand. Each part may then find its place."
    ],
    VerdictType.MERGE: [
        "Two threads become one. Redundancy eliminated.",
        "The Court binds these strands together. They are now unified.",
        "Merge decreed. The weave simplifies."
    ],
    VerdictType.RITUAL: [
        "A ritual is required. The loom demands symbolic action.",
        "Beyond mere code, a ceremony must occur.",
        "The Court prescribes ritual. The weavers must attend."
    ]
}


# ============================================================================
# Conflict Detection
# ============================================================================

class ConflictDetector:
    """Detects conflicts from events and loom state."""

    def __init__(self):
        self._conflict_counter = 0

    def _generate_id(self) -> str:
        """Generate unique conflict ID."""
        self._conflict_counter += 1
        return f"COURT-{self._conflict_counter:04d}"

    def detect_regression_clashes(
        self,
        events: list[CourtEvent]
    ) -> list[Conflict]:
        """Detect multiple regressions on same feature."""
        conflicts = []

        # Group by feature
        by_feature = defaultdict(list)
        for e in events:
            if e.motion == WeaveMotion.REGRESSION_SPIKE:
                by_feature[e.feature_id].append(e)

        # Check for clashes
        for fid, regressions in by_feature.items():
            if len(regressions) >= 2:
                conflicts.append(Conflict(
                    conflict_id=self._generate_id(),
                    conflict_type=ConflictType.REGRESSION_CLASH,
                    feature_ids=[fid],
                    events=regressions,
                    severity=min(10, len(regressions) * 3),
                    symbolic_description=CONFLICT_SYMBOLS[ConflictType.REGRESSION_CLASH]
                ))

        return conflicts

    def detect_tension_deadlocks(
        self,
        events: list[CourtEvent],
        threshold: int = 5
    ) -> list[Conflict]:
        """Detect features stuck in tension oscillation."""
        conflicts = []

        # Group by feature and analyze tension changes
        by_feature = defaultdict(list)
        for e in events:
            by_feature[e.feature_id].append(e)

        for fid, feature_events in by_feature.items():
            if len(feature_events) < threshold:
                continue

            # Check for oscillation (alternating +/- deltas)
            deltas = [e.tension_delta for e in feature_events[-threshold:]]
            sign_changes = sum(
                1 for i in range(1, len(deltas))
                if (deltas[i] > 0) != (deltas[i-1] > 0)
            )

            if sign_changes >= threshold - 2:
                conflicts.append(Conflict(
                    conflict_id=self._generate_id(),
                    conflict_type=ConflictType.TENSION_DEADLOCK,
                    feature_ids=[fid],
                    events=feature_events[-threshold:],
                    severity=7,
                    symbolic_description=CONFLICT_SYMBOLS[ConflictType.TENSION_DEADLOCK]
                ))

        return conflicts

    def detect_motion_contradictions(
        self,
        events: list[CourtEvent],
        window_seconds: int = 300
    ) -> list[Conflict]:
        """Detect contradicting motions in short time window."""
        conflicts = []

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Sliding window check
        by_feature = defaultdict(list)
        for e in sorted_events:
            by_feature[e.feature_id].append(e)

        contradicting_pairs = [
            (WeaveMotion.WARP_TUG, WeaveMotion.WEFT_WANDER),
            (WeaveMotion.TENSION_RELEASE, WeaveMotion.REGRESSION_SPIKE),
        ]

        for fid, feature_events in by_feature.items():
            for i, e1 in enumerate(feature_events):
                for e2 in feature_events[i+1:]:
                    # Check time window
                    delta = (e2.timestamp - e1.timestamp).total_seconds()
                    if delta > window_seconds:
                        break

                    # Check for contradiction
                    if e1.motion and e2.motion:
                        for m1, m2 in contradicting_pairs:
                            if (e1.motion == m1 and e2.motion == m2) or \
                               (e1.motion == m2 and e2.motion == m1):
                                conflicts.append(Conflict(
                                    conflict_id=self._generate_id(),
                                    conflict_type=ConflictType.MOTION_CONTRADICTION,
                                    feature_ids=[fid],
                                    events=[e1, e2],
                                    severity=5,
                                    symbolic_description=CONFLICT_SYMBOLS[ConflictType.MOTION_CONTRADICTION]
                                ))

        return conflicts

    def detect_worker_disputes(
        self,
        events: list[CourtEvent]
    ) -> list[Conflict]:
        """Detect multiple workers on same feature simultaneously."""
        conflicts = []

        # Group by feature and time window
        by_feature = defaultdict(list)
        for e in events:
            if e.worker_id:
                by_feature[e.feature_id].append(e)

        for fid, feature_events in by_feature.items():
            workers = set(e.worker_id for e in feature_events if e.worker_id)
            if len(workers) >= 2:
                conflicts.append(Conflict(
                    conflict_id=self._generate_id(),
                    conflict_type=ConflictType.WORKER_DISPUTE,
                    feature_ids=[fid],
                    events=feature_events,
                    severity=6,
                    symbolic_description=CONFLICT_SYMBOLS[ConflictType.WORKER_DISPUTE]
                ))

        return conflicts

    def detect_all(self, events: list[CourtEvent]) -> list[Conflict]:
        """Run all conflict detection."""
        conflicts = []
        conflicts.extend(self.detect_regression_clashes(events))
        conflicts.extend(self.detect_tension_deadlocks(events))
        conflicts.extend(self.detect_motion_contradictions(events))
        conflicts.extend(self.detect_worker_disputes(events))
        return conflicts


# ============================================================================
# The Court of Threads
# ============================================================================

class CourtOfThreads:
    """
    The symbolic arbitration system for the loom.
    
    When threads disagree, the Court convenes.
    """

    def __init__(self, state_bus_path: Path | None = None):
        self.state_bus_path = state_bus_path
        self.detector = ConflictDetector()
        self.verdicts: list[Verdict] = []
        self._proclamation_index = defaultdict(int)

    def _get_proclamation(self, verdict_type: VerdictType) -> str:
        """Get rotating proclamation for verdict type."""
        proclamations = VERDICT_PROCLAMATIONS.get(verdict_type, ["The Court has ruled."])
        idx = self._proclamation_index[verdict_type]
        self._proclamation_index[verdict_type] = idx + 1
        return proclamations[idx % len(proclamations)]

    def load_events(self) -> list[CourtEvent]:
        """Load events from state bus."""
        if not self.state_bus_path or not self.state_bus_path.exists():
            return []

        with open(self.state_bus_path) as f:
            data = json.load(f)

        events = []
        for e in data.get("events", []):
            events.append(CourtEvent(
                feature_id=e["feature_id"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                motion=WeaveMotion(e["motion"]) if e.get("motion") else None,
                worker_id=e.get("worker_id"),
                tension_delta=e.get("tension_delta", 0),
                notes=e.get("notes", ""),
                commit_hash=e.get("commit_hash")
            ))

        return events

    def detect_conflicts(
        self,
        events: list[CourtEvent] | None = None
    ) -> list[Conflict]:
        """Detect all conflicts in events."""
        if events is None:
            events = self.load_events()
        return self.detector.detect_all(events)

    def arbitrate(self, conflict: Conflict) -> Verdict:
        """
        Issue a verdict on a conflict.
        
        The Court considers:
        - Conflict type
        - Severity
        - Historical patterns
        - Tension levels
        """
        # Decision logic based on conflict type
        if conflict.conflict_type == ConflictType.REGRESSION_CLASH:
            if conflict.severity >= 8:
                verdict_type = VerdictType.SPLIT
                ruling = f"Split {conflict.feature_ids[0]} into smaller units"
                actions = [
                    "Break feature into 2-3 smaller, testable units",
                    "Add explicit dependencies between new units",
                    "Reset tension on all new units"
                ]
            else:
                verdict_type = VerdictType.ARBITRATION
                ruling = f"Revert to last stable state for {conflict.feature_ids[0]}"
                actions = [
                    "Reset feature to 'failing' status",
                    "Clear regression count",
                    "Add note about regression pattern"
                ]

        elif conflict.conflict_type == ConflictType.TENSION_DEADLOCK:
            verdict_type = VerdictType.RITUAL
            ruling = f"Manual review required for {conflict.feature_ids[0]}"
            actions = [
                "Pause automated workers on this feature",
                "Human review of approach",
                "Consider fundamental redesign"
            ]

        elif conflict.conflict_type == ConflictType.MOTION_CONTRADICTION:
            verdict_type = VerdictType.DEFERRAL
            ruling = "Wait for additional evidence"
            actions = [
                "Log contradiction for pattern analysis",
                "Continue with most recent motion",
                "Flag for review if pattern repeats"
            ]

        elif conflict.conflict_type == ConflictType.WORKER_DISPUTE:
            verdict_type = VerdictType.ARBITRATION
            # Pick worker with most recent success
            set(e.worker_id for e in conflict.events if e.worker_id)
            ruling = "Assign feature to single worker"
            actions = [
                "Lock feature to first claiming worker",
                "Other workers must wait or pick different feature",
                "Add coordination note to progress log"
            ]

        elif conflict.conflict_type == ConflictType.DEPENDENCY_CYCLE:
            verdict_type = VerdictType.DISSOLUTION
            ruling = "Break dependency cycle"
            actions = [
                "Remove one dependency edge",
                "Consider feature merge if tightly coupled",
                "Document the cycle for future reference"
            ]

        else:
            verdict_type = VerdictType.DEFERRAL
            ruling = "Unknown conflict type"
            actions = ["Manual investigation required"]

        verdict = Verdict(
            conflict_id=conflict.conflict_id,
            verdict_type=verdict_type,
            ruling=ruling,
            proclamation=self._get_proclamation(verdict_type),
            affected_features=conflict.feature_ids,
            recommended_actions=actions
        )

        self.verdicts.append(verdict)
        return verdict

    def convene(self, events: list[CourtEvent] | None = None) -> list[Verdict]:
        """
        Full Court session: detect conflicts and issue verdicts.
        
        Returns list of all verdicts issued.
        """
        conflicts = self.detect_conflicts(events)
        verdicts = [self.arbitrate(c) for c in conflicts]
        return verdicts

    def render_session(self, verdicts: list[Verdict]) -> str:
        """Render Court session as narrative."""
        if not verdicts:
            return "🏛️ The Court of Threads convenes. No conflicts detected. The weave is harmonious."

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║            THE COURT OF THREADS CONVENES                     ║",
            "╠══════════════════════════════════════════════════════════════╣"
        ]

        for v in verdicts:
            lines.append("║                                                              ║")
            lines.append(f"║  Case: {v.conflict_id:52} ║")
            lines.append(f"║  {v.proclamation[:58]:58} ║")
            lines.append("║                                                              ║")
            lines.append(f"║  Ruling: {v.ruling[:50]:50}   ║")
            lines.append("║                                                              ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Demo with synthetic events
    events = [
        CourtEvent(
            feature_id="F001",
            timestamp=datetime.now(),
            motion=WeaveMotion.REGRESSION_SPIKE,
            worker_id="worker_1",
            tension_delta=3.0,
            notes="First regression"
        ),
        CourtEvent(
            feature_id="F001",
            timestamp=datetime.now(),
            motion=WeaveMotion.REGRESSION_SPIKE,
            worker_id="worker_2",
            tension_delta=2.5,
            notes="Second regression"
        ),
        CourtEvent(
            feature_id="F002",
            timestamp=datetime.now(),
            motion=WeaveMotion.WARP_TUG,
            worker_id="worker_1",
            tension_delta=-1.0
        ),
    ]

    court = CourtOfThreads()
    verdicts = court.convene(events)

    print(court.render_session(verdicts))

    for v in verdicts:
        print(f"\nConflict {v.conflict_id}:")
        print(f"  Verdict: {v.verdict_type.value}")
        print(f"  Ruling: {v.ruling}")
        print(f"  Actions: {v.recommended_actions}")
