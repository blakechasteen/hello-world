"""
Strands Update Module
=====================

Updates strand state after worker runs.

Integrates:
- Tension calculation
- Weave motion decoding
- History recording
- Visualization triggers

Usage:
    from hololoom.domain_harness.strands import StrandsUpdater
    
    updater = StrandsUpdater(Path("./domain_memory"))
    event = updater.update_after_worker(feature, action_result)
"""

import json
from datetime import datetime
from pathlib import Path

from hololoom.domain_harness.schema import ActionResult, Feature, FeatureStatus, ProgressEntry
from hololoom.domain_harness.strands.tension_model import LoomState, StrandState, TensionUpdater
from hololoom.domain_harness.strands.weave_decoder import (
    WeaveDecoder,
    WeaveEvent,
    WeaveHistory,
    WeaveNarrative,
)
from hololoom.domain_harness.templates import FlatFileBackend


class StrandsUpdater:
    """
    Manages strand state updates integrated with domain harness.
    
    Handles:
    1. Loading/saving loom state
    2. Updating tensions after worker runs
    3. Decoding weave motions
    4. Recording weave history
    """

    def __init__(self, domain_memory_dir: Path):
        self.dir = Path(domain_memory_dir)
        self.strands_dir = self.dir / "strands"
        self.loom_state_path = self.strands_dir / "loom_state.json"
        self.weave_history_path = self.strands_dir / "weave_history.json"

        self.backend = FlatFileBackend(self.dir)
        self.tension_updater = TensionUpdater()
        self.decoder = WeaveDecoder()
        self.narrative = WeaveNarrative()

        # Ensure strands directory exists
        self.strands_dir.mkdir(exist_ok=True)

    # ========================================================================
    # State Management
    # ========================================================================

    def load_loom_state(self) -> LoomState:
        """Load loom state from disk."""
        if self.loom_state_path.exists():
            with open(self.loom_state_path) as f:
                data = json.load(f)
            return LoomState.from_dict(data)
        return LoomState()

    def save_loom_state(self, state: LoomState):
        """Save loom state to disk."""
        with open(self.loom_state_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)

    def load_weave_history(self) -> WeaveHistory:
        """Load weave history from disk."""
        if self.weave_history_path.exists():
            with open(self.weave_history_path) as f:
                data = json.load(f)
            return WeaveHistory.from_dict(data)
        return WeaveHistory()

    def save_weave_history(self, history: WeaveHistory):
        """Save weave history to disk."""
        with open(self.weave_history_path, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)

    # ========================================================================
    # Update Operations
    # ========================================================================

    def update_all_tensions(self) -> LoomState:
        """
        Update tensions for all features.
        
        Call this periodically or after any state change.
        """
        features = self.backend.load_features()
        loom_state = self.load_loom_state()

        # Would load progress history from backend
        progress_history: list[ProgressEntry] = []

        # Update tensions
        loom_state = self.tension_updater.update_all(
            features=features,
            progress_history=progress_history,
            loom_state=loom_state
        )

        self.save_loom_state(loom_state)
        return loom_state

    def update_after_worker(
        self,
        feature: Feature,
        action_result: ActionResult,
        previous_status: FeatureStatus | None = None
    ) -> WeaveEvent:
        """
        Update strands after a worker run.
        
        Args:
            feature: Feature that was worked on
            action_result: Result from worker
            previous_status: Status before worker ran
            
        Returns:
            WeaveEvent describing what happened
        """
        # Load current state
        loom_state = self.load_loom_state()
        history = self.load_weave_history()

        # Get or create strand
        if feature.id not in loom_state.strands:
            loom_state.strands[feature.id] = StrandState(feature_id=feature.id)

        strand = loom_state.strands[feature.id]

        # Decode weave motion
        event = self.decoder.decode(
            feature=feature,
            action_result=action_result,
            strand_state=strand,
            previous_status=previous_status
        )

        # Update strand with new values
        strand.tension = event.new_tension
        strand.weft_position = event.new_weft
        strand.tension_history.append(event.new_tension)
        strand.last_motion = event.motion.value
        strand.last_updated = datetime.now()

        # Record event
        history.record(event)

        # Update aggregates
        loom_state.update_aggregates()

        # Save
        self.save_loom_state(loom_state)
        self.save_weave_history(history)

        return event

    def get_session_summary(self) -> str:
        """Get narrative summary of current session."""
        history = self.load_weave_history()
        return self.narrative.summarize_session(history)

    def get_feature_narrative(self, feature_id: str) -> str:
        """Get narrative for a specific feature."""
        history = self.load_weave_history()
        events = history.get_feature_events(feature_id)

        if not events:
            return f"No weave history for {feature_id}"

        lines = [f"Weave history for {feature_id}:"]
        for event in events[-5:]:  # Last 5 events
            lines.append(f"  {self.narrative.describe_event(event)}")

        return "\n".join(lines)

    # ========================================================================
    # Strand Queries
    # ========================================================================

    def get_critical_strands(self) -> list[str]:
        """Get features with critical tension (>8)."""
        loom_state = self.load_loom_state()
        return [
            strand.feature_id
            for strand in loom_state.strands.values()
            if strand.tension > 8.0
        ]

    def get_stale_strands(self) -> list[str]:
        """Get features that haven't been updated recently."""
        loom_state = self.load_loom_state()
        return [
            strand.feature_id
            for strand in loom_state.strands.values()
            if strand.is_stale
        ]

    def get_regression_strands(self) -> list[str]:
        """Get features in regression state."""
        loom_state = self.load_loom_state()
        return [
            strand.feature_id
            for strand in loom_state.strands.values()
            if strand.is_regression
        ]

    def is_loom_stable(self) -> bool:
        """Check if loom is in stable state."""
        loom_state = self.load_loom_state()
        return loom_state.is_stable

    def is_complete(self) -> bool:
        """Check if all strands are complete (low tension)."""
        loom_state = self.load_loom_state()
        return loom_state.is_complete


# ============================================================================
# Integration with run_worker.py
# ============================================================================

def integrate_strands_with_worker(
    domain_memory_dir: Path,
    feature: Feature,
    action_result: ActionResult,
    previous_status: FeatureStatus | None = None,
    verbose: bool = True
) -> WeaveEvent:
    """
    Convenience function for integrating strands with worker.
    
    Call this from run_worker.py after each worker run.
    """
    updater = StrandsUpdater(domain_memory_dir)
    event = updater.update_after_worker(feature, action_result, previous_status)

    if verbose:
        print(f"\n🧵 Weave: {event.symbol} {event.motion.value}")
        print(f"   Tension: {event.old_tension:.1f} → {event.new_tension:.1f}")
        print(f"   {event.symbolic_effect}")

        if event.is_regression:
            print("   ⚠ REGRESSION DETECTED")
        if event.is_critical:
            print("   ⚡ CRITICAL TENSION")
        if event.is_completion:
            print("   ✓ Feature complete!")

    return event
