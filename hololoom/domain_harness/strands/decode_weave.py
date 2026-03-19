"""
Weave Motion Decoder - Worker Actions → Symbolic Movements

Interprets worker outputs (code changes, test results, notes)
as weave motions in the symbolic framework.

This is where the harness gains a *story* of its own.
Each worker run isn't just "pass/fail" - it's a movement
in the ongoing weave of the project.

Weave Motions:
- warp_tug: Tightening - feature improving, tests passing
- weft_shift: Lateral movement - refactoring without status change
- weft_wander: Introducing slack - failure
- tension_release: Feature stabilizing
- regression_spike: Previously passing, now failing
- knot_tie: Dependency resolved
- thread_snap: Critical failure

Integration:
- Reads worker ActionResult
- Computes weave motion
- Updates strand state
- Returns symbolic interpretation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hololoom.domain_harness.schema import ActionResult, Feature, FeatureStatus
from hololoom.domain_harness.strands.tension_model import (
    TENSION_STABLE,
    TENSION_STRESSED,
    LoomState,
    TensionModel,
    WeaveMotion,
)

# ============================================================================
# Decoded Weave
# ============================================================================

@dataclass
class DecodedWeave:
    """
    Result of decoding a worker action into symbolic movement.
    """
    feature_id: str
    motion: WeaveMotion
    symbolic_effect: str
    tension_delta: float
    new_tension: float

    # Additional context
    was_regression: bool = False
    dependencies_affected: list[str] = None
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "motion": self.motion.value,
            "symbolic_effect": self.symbolic_effect,
            "tension_delta": self.tension_delta,
            "new_tension": self.new_tension,
            "was_regression": self.was_regression,
            "dependencies_affected": self.dependencies_affected or [],
            "narrative": self.narrative
        }


# ============================================================================
# Symbolic Effects (Human-Readable)
# ============================================================================

SYMBOLIC_EFFECTS = {
    WeaveMotion.WARP_TUG: "tightening alignment - pattern emerging",
    WeaveMotion.WEFT_SHIFT: "lateral adjustment - structure reorganizing",
    WeaveMotion.WEFT_WANDER: "introducing slack - thread seeking path",
    WeaveMotion.TENSION_RELEASE: "stabilizing - fabric settling",
    WeaveMotion.REGRESSION_SPIKE: "unraveling - previous work undone",
    WeaveMotion.KNOT_TIE: "binding - dependencies interlocking",
    WeaveMotion.THREAD_SNAP: "rupture - critical path severed"
}

MOTION_NARRATIVES = {
    WeaveMotion.WARP_TUG: [
        "The thread pulls taut. Progress.",
        "Another strand finds its place in the pattern.",
        "The weave tightens. Coherence emerges."
    ],
    WeaveMotion.WEFT_SHIFT: [
        "The shuttle moves sideways. Restructuring.",
        "Not forward, not back—reshaping the possible.",
        "The loom adjusts. Same threads, new arrangement."
    ],
    WeaveMotion.WEFT_WANDER: [
        "The thread wanders. Seeking its path.",
        "Slack enters the weave. Room to explore.",
        "Not failure—discovery of what doesn't work."
    ],
    WeaveMotion.TENSION_RELEASE: [
        "The fabric relaxes. Stability achieved.",
        "Tension flows out. The pattern holds.",
        "What was stressed is now at rest."
    ],
    WeaveMotion.REGRESSION_SPIKE: [
        "The weave unravels. What was done is undone.",
        "A thread snaps back. Ground lost.",
        "The loom remembers. It will remember again."
    ],
    WeaveMotion.KNOT_TIE: [
        "Dependencies bind. The structure locks.",
        "One thread joins another. Strength through connection.",
        "The knot holds. We can build upon it."
    ],
    WeaveMotion.THREAD_SNAP: [
        "Critical failure. The thread breaks.",
        "The pattern cannot hold this strain.",
        "Stop. Assess. The loom needs tending."
    ]
}


# ============================================================================
# Weave Decoder
# ============================================================================

class WeaveDecoder:
    """
    Decodes worker actions into symbolic weave movements.
    
    Usage:
        decoder = WeaveDecoder()
        result = decoder.decode(feature, action_result, loom_state)
    """

    def __init__(self):
        self.tension_model = TensionModel()
        self._narrative_index = {}  # Track which narratives we've used

    def decode(
        self,
        feature: Feature,
        result: ActionResult,
        loom_state: LoomState,
        previous_status: FeatureStatus | None = None
    ) -> DecodedWeave:
        """
        Decode a worker action into a weave motion.
        
        Args:
            feature: The feature being worked on
            result: ActionResult from the worker
            loom_state: Current state of all strands
            previous_status: Feature status before this action
            
        Returns:
            DecodedWeave with symbolic interpretation
        """
        # Get or create strand state
        strand = loom_state.get_or_create_strand(feature.id)

        # Determine if this is a regression
        was_passing = previous_status == FeatureStatus.PASSING

        # Check for blocked dependencies
        has_blocked = bool(feature.depends_on)  # Simplified check

        # Extract notes from result
        notes = [result.summary] if result.summary else []
        if result.test_result and result.test_result.output:
            notes.append(result.test_result.output[:200])

        # Calculate tension delta and motion
        delta, motion = self.tension_model.calculate_delta(
            passed=result.success,
            was_passing=was_passing,
            has_blocked_dependency=has_blocked,
            notes=notes
        )

        # Apply to strand
        self.tension_model.apply_delta(strand, delta, motion)

        # Update loom state
        loom_state.total_weaves += 1
        loom_state.last_weave = datetime.now()

        # Build decoded weave
        decoded = DecodedWeave(
            feature_id=feature.id,
            motion=motion,
            symbolic_effect=SYMBOLIC_EFFECTS.get(motion, "unknown effect"),
            tension_delta=delta,
            new_tension=strand.tension,
            was_regression=(motion == WeaveMotion.REGRESSION_SPIKE),
            dependencies_affected=feature.depends_on,
            narrative=self._get_narrative(motion)
        )

        return decoded

    def _get_narrative(self, motion: WeaveMotion) -> str:
        """Get a narrative line for the motion, cycling through options."""
        narratives = MOTION_NARRATIVES.get(motion, ["The loom moves."])

        # Track index per motion type
        idx = self._narrative_index.get(motion, 0)
        narrative = narratives[idx % len(narratives)]
        self._narrative_index[motion] = idx + 1

        return narrative

    def decode_batch(
        self,
        results: list[tuple[Feature, ActionResult, FeatureStatus | None]],
        loom_state: LoomState
    ) -> list[DecodedWeave]:
        """Decode multiple worker actions."""
        return [
            self.decode(feature, result, loom_state, prev_status)
            for feature, result, prev_status in results
        ]

    def generate_weave_summary(self, loom_state: LoomState) -> str:
        """
        Generate a poetic summary of the current loom state.
        """
        avg_tension = loom_state.average_tension()
        stressed = loom_state.stressed_strands()
        critical = loom_state.critical_strands()

        lines = []

        # Overall state
        if avg_tension <= TENSION_STABLE:
            lines.append("The loom is calm. The weave holds steady.")
        elif avg_tension <= TENSION_STRESSED:
            lines.append("The loom hums with activity. Tension building.")
        else:
            lines.append("The loom strains. Attention needed.")

        # Specific concerns
        if critical:
            lines.append(f"Critical threads: {', '.join(critical)}")
        elif stressed:
            lines.append(f"Stressed threads: {', '.join(stressed)}")

        # Recent motion
        recent_motions = []
        for strand in loom_state.strands.values():
            if strand.last_motion:
                recent_motions.append(strand.last_motion)

        if recent_motions:
            # Most common recent motion
            from collections import Counter
            most_common = Counter(recent_motions).most_common(1)[0][0]
            lines.append(f"The dominant motion: {most_common.value}")

        return "\n".join(lines)


# ============================================================================
# Integration with Worker
# ============================================================================

def decode_worker_result(
    feature: Feature,
    result: ActionResult,
    loom_state: LoomState,
    previous_status: FeatureStatus | None = None
) -> tuple[DecodedWeave, str]:
    """
    Convenience function for decoding a single worker result.
    
    Returns (decoded_weave, narrative_line)
    """
    decoder = WeaveDecoder()
    decoded = decoder.decode(feature, result, loom_state, previous_status)

    # Build a narrative line for logging
    emoji = decoder.tension_model.get_status_emoji(decoded.new_tension)
    narrative = (
        f"{emoji} {decoded.motion.value}: {decoded.narrative} "
        f"(tension: {decoded.new_tension:.1f})"
    )

    return decoded, narrative


# ============================================================================
# Ritual Layer (Optional)
# ============================================================================

class WeaveRitual:
    """
    Symbolic rituals that respond to weave events.
    
    This is where the loom becomes responsive—
    taking symbolic action when certain patterns emerge.
    """

    @staticmethod
    def on_regression(feature_id: str, loom_state: LoomState) -> str:
        """Ritual when regression occurs."""
        strand = loom_state.strands.get(feature_id)
        if strand and strand.regression_count >= 3:
            return f"⚠️ RITUAL: {feature_id} has regressed {strand.regression_count} times. Consider refactoring approach."
        return ""

    @staticmethod
    def on_consecutive_fails(feature_id: str, loom_state: LoomState) -> str:
        """Ritual when feature fails repeatedly."""
        strand = loom_state.strands.get(feature_id)
        if strand and strand.consecutive_fails >= 5:
            return f"🔄 RITUAL: {feature_id} failing {strand.consecutive_fails}x. Break into smaller features?"
        return ""

    @staticmethod
    def on_critical_tension(loom_state: LoomState) -> str:
        """Ritual when overall tension is critical."""
        critical = loom_state.critical_strands()
        if len(critical) >= 3:
            return f"🛑 RITUAL: {len(critical)} threads critical. Pause and assess architecture."
        return ""

    @staticmethod
    def on_completion(loom_state: LoomState) -> str:
        """Ritual when all features pass."""
        all_stable = all(
            s.tension <= TENSION_STABLE
            for s in loom_state.strands.values()
        )
        if all_stable:
            return "🎉 RITUAL: The weave is complete. All threads rest in their places."
        return ""
