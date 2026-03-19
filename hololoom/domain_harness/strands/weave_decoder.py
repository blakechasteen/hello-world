"""
Weave Motion Decoder
====================

Decodes worker actions into symbolic weave motions.

Actions are translated into:
- warp_tug: Tightening structural thread (success)
- weft_shift: Horizontal progress motion
- weft_wander: Introducing slack (failure)
- tension_spike: Sudden increase (regression)
- relaxation: Tension release (completion)

Weaving Metaphor:
Every worker action is a motion of the shuttle.
The decoder interprets what that motion means symbolically.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from hololoom.domain_harness.schema import (
    ActionResult,
    Feature,
    FeatureStatus,
    ProgressAction,
    ProgressEntry,
)
from hololoom.domain_harness.strands.tension_model import StrandState

# ============================================================================
# Weave Motion Types
# ============================================================================

class WeaveMotion(str, Enum):
    """Symbolic motions in the loom."""

    # Success motions
    WARP_TUG = "warp_tug"           # Tightening alignment (success)
    WEFT_ADVANCE = "weft_advance"   # Forward progress
    RELAXATION = "relaxation"       # Completion, tension release

    # Failure motions
    WEFT_WANDER = "weft_wander"     # Introducing slack (failure)
    TENSION_SPIKE = "tension_spike" # Sudden increase (regression)
    SNAG = "snag"                   # Blocked, stuck

    # Neutral motions
    SHUTTLE_PASS = "shuttle_pass"   # Worker ran, no change
    IDLE = "idle"                   # No action taken


# Symbolic effects for each motion
MOTION_EFFECTS = {
    WeaveMotion.WARP_TUG: {
        "description": "Structural thread tightened",
        "tension_delta": -0.5,
        "weft_delta": +0.1,
        "symbol": "⟨⟩"
    },
    WeaveMotion.WEFT_ADVANCE: {
        "description": "Progress through the weave",
        "tension_delta": -0.3,
        "weft_delta": +0.2,
        "symbol": "→"
    },
    WeaveMotion.RELAXATION: {
        "description": "Tension released, thread at rest",
        "tension_delta": -2.0,
        "weft_delta": 0,
        "symbol": "∿"
    },
    WeaveMotion.WEFT_WANDER: {
        "description": "Slack introduced, alignment lost",
        "tension_delta": +0.6,
        "weft_delta": -0.1,
        "symbol": "~"
    },
    WeaveMotion.TENSION_SPIKE: {
        "description": "Sudden strain on the weave",
        "tension_delta": +2.0,
        "weft_delta": 0,
        "symbol": "⚡"
    },
    WeaveMotion.SNAG: {
        "description": "Thread caught, cannot proceed",
        "tension_delta": +1.0,
        "weft_delta": 0,
        "symbol": "✕"
    },
    WeaveMotion.SHUTTLE_PASS: {
        "description": "Shuttle passed without effect",
        "tension_delta": 0,
        "weft_delta": 0,
        "symbol": "○"
    },
    WeaveMotion.IDLE: {
        "description": "Loom at rest",
        "tension_delta": 0,
        "weft_delta": 0,
        "symbol": "·"
    }
}


# ============================================================================
# Weave Event
# ============================================================================

@dataclass
class WeaveEvent:
    """
    A decoded weave motion with full context.
    """
    feature_id: str
    motion: WeaveMotion
    timestamp: datetime

    # Tension change
    old_tension: float
    new_tension: float
    tension_delta: float

    # Position change
    old_weft: float
    new_weft: float

    # Context
    symbolic_effect: str
    symbol: str
    notes: list[str]

    # Flags
    is_regression: bool = False
    is_completion: bool = False
    is_critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "motion": self.motion.value,
            "timestamp": self.timestamp.isoformat(),
            "old_tension": self.old_tension,
            "new_tension": self.new_tension,
            "tension_delta": self.tension_delta,
            "old_weft": self.old_weft,
            "new_weft": self.new_weft,
            "symbolic_effect": self.symbolic_effect,
            "symbol": self.symbol,
            "notes": self.notes,
            "is_regression": self.is_regression,
            "is_completion": self.is_completion,
            "is_critical": self.is_critical
        }

    def to_log_line(self) -> str:
        """Format for weave history log."""
        flags = []
        if self.is_regression:
            flags.append("REGRESSION")
        if self.is_completion:
            flags.append("COMPLETE")
        if self.is_critical:
            flags.append("CRITICAL")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{self.symbol} {self.motion.value.upper()} {self.feature_id}: "
            f"tension {self.old_tension:.1f}→{self.new_tension:.1f} "
            f"({self.tension_delta:+.1f})"
            f"{flag_str}"
        )


# ============================================================================
# Weave Decoder
# ============================================================================

class WeaveDecoder:
    """
    Decodes worker actions into symbolic weave motions.
    
    Usage:
        decoder = WeaveDecoder()
        event = decoder.decode(
            feature=feature,
            action_result=result,
            strand_state=strand
        )
    """

    def decode(
        self,
        feature: Feature,
        action_result: ActionResult,
        strand_state: StrandState,
        previous_status: FeatureStatus | None = None
    ) -> WeaveEvent:
        """
        Decode a worker action into a weave event.
        
        Args:
            feature: The feature that was worked on
            action_result: Result from worker
            strand_state: Current strand state
            previous_status: Status before this action
            
        Returns:
            WeaveEvent describing the symbolic motion
        """
        notes = []

        # Determine motion type
        motion = self._determine_motion(
            feature=feature,
            action_result=action_result,
            strand_state=strand_state,
            previous_status=previous_status,
            notes=notes
        )

        # Get motion effects
        effects = MOTION_EFFECTS[motion]

        # Calculate tension change
        old_tension = strand_state.tension
        tension_delta = effects["tension_delta"]

        # Adjust for special cases
        if strand_state.is_regression:
            tension_delta += 1.5
            notes.append("Regression amplifies tension")

        if strand_state.consecutive_failures > 2:
            tension_delta += 0.5 * (strand_state.consecutive_failures - 2)
            notes.append(f"Consecutive failures: {strand_state.consecutive_failures}")

        new_tension = max(0, min(10, old_tension + tension_delta))

        # Calculate weft change
        old_weft = strand_state.weft_position
        weft_delta = effects["weft_delta"]
        new_weft = max(0, min(1, old_weft + weft_delta))

        # Detect special conditions
        is_regression = (
            previous_status == FeatureStatus.PASSING and
            feature.status == FeatureStatus.FAILING
        )
        is_completion = (
            feature.status == FeatureStatus.PASSING and
            previous_status != FeatureStatus.PASSING
        )
        is_critical = new_tension > 9.0

        if is_regression:
            notes.append("Was passing, now failing")
        if is_completion:
            notes.append("Feature complete!")
        if is_critical:
            notes.append("CRITICAL: Tension near breaking point")

        return WeaveEvent(
            feature_id=feature.id,
            motion=motion,
            timestamp=datetime.now(),
            old_tension=old_tension,
            new_tension=new_tension,
            tension_delta=tension_delta,
            old_weft=old_weft,
            new_weft=new_weft,
            symbolic_effect=effects["description"],
            symbol=effects["symbol"],
            notes=notes,
            is_regression=is_regression,
            is_completion=is_completion,
            is_critical=is_critical
        )

    def _determine_motion(
        self,
        feature: Feature,
        action_result: ActionResult,
        strand_state: StrandState,
        previous_status: FeatureStatus | None,
        notes: list[str]
    ) -> WeaveMotion:
        """Determine which motion type applies."""

        # No action taken
        if action_result.feature_id == "none":
            return WeaveMotion.IDLE

        # Success case
        if action_result.success:
            # First time passing
            if previous_status != FeatureStatus.PASSING:
                notes.append("First successful pass")
                return WeaveMotion.WARP_TUG

            # Already passing, staying passing
            return WeaveMotion.WEFT_ADVANCE

        # Failure case
        else:
            # Regression (was passing)
            if previous_status == FeatureStatus.PASSING:
                notes.append("Regression from passing state")
                return WeaveMotion.TENSION_SPIKE

            # Blocked by dependencies
            if feature.status == FeatureStatus.BLOCKED:
                return WeaveMotion.SNAG

            # Regular failure
            return WeaveMotion.WEFT_WANDER

    def decode_from_progress(
        self,
        feature: Feature,
        progress_entry: ProgressEntry,
        strand_state: StrandState
    ) -> WeaveEvent:
        """
        Decode from a progress log entry.
        
        Useful for replaying history.
        """
        # Simulate action result
        success = progress_entry.action == ProgressAction.COMPLETED

        action_result = ActionResult(
            feature_id=feature.id,
            success=success,
            summary=progress_entry.summary
        )

        return self.decode(
            feature=feature,
            action_result=action_result,
            strand_state=strand_state
        )


# ============================================================================
# Weave History
# ============================================================================

class WeaveHistory:
    """
    Records weave events over time.
    
    Provides:
    - Event timeline
    - Pattern detection
    - Narrative generation
    """

    def __init__(self, max_events: int = 1000):
        self.events: list[WeaveEvent] = []
        self.max_events = max_events

    def record(self, event: WeaveEvent):
        """Record a weave event."""
        self.events.append(event)

        # Trim if over limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_recent(self, count: int = 20) -> list[WeaveEvent]:
        """Get recent events."""
        return self.events[-count:]

    def get_feature_events(self, feature_id: str) -> list[WeaveEvent]:
        """Get events for a specific feature."""
        return [e for e in self.events if e.feature_id == feature_id]

    def get_regressions(self) -> list[WeaveEvent]:
        """Get all regression events."""
        return [e for e in self.events if e.is_regression]

    def get_completions(self) -> list[WeaveEvent]:
        """Get all completion events."""
        return [e for e in self.events if e.is_completion]

    def to_log(self) -> str:
        """Generate log format."""
        lines = ["# Weave History", ""]
        for event in self.events:
            lines.append(event.to_log_line())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "events": [e.to_dict() for e in self.events]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WeaveHistory':
        """Deserialize from storage."""
        history = cls()
        for e_data in data.get("events", []):
            event = WeaveEvent(
                feature_id=e_data["feature_id"],
                motion=WeaveMotion(e_data["motion"]),
                timestamp=datetime.fromisoformat(e_data["timestamp"]),
                old_tension=e_data["old_tension"],
                new_tension=e_data["new_tension"],
                tension_delta=e_data["tension_delta"],
                old_weft=e_data["old_weft"],
                new_weft=e_data["new_weft"],
                symbolic_effect=e_data["symbolic_effect"],
                symbol=e_data["symbol"],
                notes=e_data.get("notes", []),
                is_regression=e_data.get("is_regression", False),
                is_completion=e_data.get("is_completion", False),
                is_critical=e_data.get("is_critical", False)
            )
            history.events.append(event)
        return history


# ============================================================================
# Narrative Generator
# ============================================================================

class WeaveNarrative:
    """
    Generates human-readable narratives from weave history.
    
    "The loom tells a story."
    """

    def summarize_session(self, history: WeaveHistory) -> str:
        """Generate session summary."""
        events = history.events
        if not events:
            return "The loom stands silent. No threads have moved."

        completions = len([e for e in events if e.is_completion])
        regressions = len([e for e in events if e.is_regression])
        criticals = len([e for e in events if e.is_critical])

        # Calculate tension trajectory
        if len(events) >= 2:
            start_tension = events[0].old_tension
            end_tension = events[-1].new_tension
            tension_change = end_tension - start_tension
        else:
            tension_change = 0

        lines = []
        lines.append(f"Session: {len(events)} motions recorded")
        lines.append(f"Completions: {completions}")

        if regressions > 0:
            lines.append(f"⚠ Regressions: {regressions}")

        if criticals > 0:
            lines.append(f"⚡ Critical events: {criticals}")

        if tension_change < -1:
            lines.append("Overall: Tension easing, weave stabilizing")
        elif tension_change > 1:
            lines.append("Overall: Tension building, attention needed")
        else:
            lines.append("Overall: Steady state")

        return "\n".join(lines)

    def describe_event(self, event: WeaveEvent) -> str:
        """Generate narrative description of single event."""
        motion_narratives = {
            WeaveMotion.WARP_TUG: "The structural thread tightens, finding alignment.",
            WeaveMotion.WEFT_ADVANCE: "The shuttle passes smoothly, weaving forward.",
            WeaveMotion.RELAXATION: "The thread settles, tension released.",
            WeaveMotion.WEFT_WANDER: "The weft drifts, seeking its path.",
            WeaveMotion.TENSION_SPIKE: "A sudden pull! The weave strains.",
            WeaveMotion.SNAG: "The thread catches—blocked.",
            WeaveMotion.SHUTTLE_PASS: "The shuttle passes without trace.",
            WeaveMotion.IDLE: "The loom waits."
        }

        base = motion_narratives.get(event.motion, "A motion in the weave.")

        if event.is_regression:
            base += " What was woven has unraveled."
        if event.is_completion:
            base += " A thread finds its place."
        if event.is_critical:
            base += " The thread strains near breaking."

        return base
