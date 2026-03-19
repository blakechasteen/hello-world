"""
Strands Tension Model
=====================

Computes symbolic tension for features based on:
- Status (passing/failing)
- Dependencies (blocked/unblocked)
- Progress history (velocity, regressions)
- Time since last change

Tension is a scalar [0, 10] where:
- 0 = Fully relaxed (complete, stable)
- 5 = Neutral (in progress)
- 10 = Maximum tension (blocked, regressing, stalled)

Weaving Metaphor:
Tension determines how tightly the thread is pulled.
High tension = needs attention, risk of snapping.
Low tension = stable, can bear load.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from hololoom.domain_harness.schema import Feature, FeatureStatus, ProgressAction, ProgressEntry

# ============================================================================
# Constants
# ============================================================================

# Base tension by status
STATUS_TENSION = {
    FeatureStatus.PASSING: 0.5,      # Stable, slight maintenance tension
    FeatureStatus.FAILING: 6.0,      # Needs work
    FeatureStatus.PENDING: 4.0,      # Waiting to start
    FeatureStatus.IN_PROGRESS: 5.0,  # Active work
    FeatureStatus.BLOCKED: 8.0,      # High tension - stuck
    FeatureStatus.SKIPPED: 2.0,      # Low priority
}

# Tension modifiers
DEPENDENCY_BLOCKED_MODIFIER = +2.0    # Blocked by unmet dependency
REGRESSION_MODIFIER = +3.0            # Was passing, now failing
STALE_MODIFIER = +1.5                 # No progress in 24+ hours
VELOCITY_POSITIVE_MODIFIER = -1.0     # Recent progress
CONSECUTIVE_FAILS_MODIFIER = +0.5     # Per consecutive failure


# ============================================================================
# Strand State
# ============================================================================

@dataclass
class StrandState:
    """
    Symbolic state of a feature's "thread" in the loom.
    
    Tracks tension, weave position, and history.
    """
    feature_id: str
    tension: float = 5.0

    # Position in the weave
    warp_position: float = 0.0   # Vertical (structure)
    weft_position: float = 0.0   # Horizontal (progress)

    # History
    tension_history: list[float] = field(default_factory=list)
    last_motion: str | None = None
    last_updated: datetime = field(default_factory=datetime.now)

    # Flags
    is_regression: bool = False
    is_stale: bool = False
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "feature_id": self.feature_id,
            "tension": self.tension,
            "warp_position": self.warp_position,
            "weft_position": self.weft_position,
            "tension_history": self.tension_history[-20:],  # Keep last 20
            "last_motion": self.last_motion,
            "last_updated": self.last_updated.isoformat(),
            "is_regression": self.is_regression,
            "is_stale": self.is_stale,
            "consecutive_failures": self.consecutive_failures
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StrandState':
        """Deserialize from JSON."""
        return cls(
            feature_id=data["feature_id"],
            tension=data.get("tension", 5.0),
            warp_position=data.get("warp_position", 0.0),
            weft_position=data.get("weft_position", 0.0),
            tension_history=data.get("tension_history", []),
            last_motion=data.get("last_motion"),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            is_regression=data.get("is_regression", False),
            is_stale=data.get("is_stale", False),
            consecutive_failures=data.get("consecutive_failures", 0)
        )


# ============================================================================
# Tension Calculator
# ============================================================================

class TensionModel:
    """
    Computes tension for features based on state and history.
    
    The model considers:
    1. Current status (passing/failing/blocked)
    2. Dependency state (are blockers resolved?)
    3. Progress velocity (recent activity)
    4. Regression detection (was passing, now failing)
    5. Staleness (time since last change)
    """

    def __init__(
        self,
        stale_threshold_hours: float = 24.0,
        max_tension: float = 10.0,
        min_tension: float = 0.0
    ):
        self.stale_threshold = timedelta(hours=stale_threshold_hours)
        self.max_tension = max_tension
        self.min_tension = min_tension

    def compute_tension(
        self,
        feature: Feature,
        strand_state: StrandState,
        progress_history: list[ProgressEntry],
        dependency_states: dict[str, FeatureStatus]
    ) -> float:
        """
        Compute current tension for a feature.
        
        Returns tension in [min_tension, max_tension].
        """
        # Base tension from status
        tension = STATUS_TENSION.get(feature.status, 5.0)

        # Dependency modifier
        if self._has_blocked_dependencies(feature, dependency_states):
            tension += DEPENDENCY_BLOCKED_MODIFIER
            strand_state.is_regression = False  # Can't regress if blocked

        # Regression modifier
        if self._detect_regression(feature, progress_history):
            tension += REGRESSION_MODIFIER
            strand_state.is_regression = True
        else:
            strand_state.is_regression = False

        # Staleness modifier
        if self._is_stale(strand_state):
            tension += STALE_MODIFIER
            strand_state.is_stale = True
        else:
            strand_state.is_stale = False

        # Velocity modifier (recent progress reduces tension)
        velocity = self._compute_velocity(progress_history)
        if velocity > 0:
            tension += VELOCITY_POSITIVE_MODIFIER

        # Consecutive failures
        failures = self._count_consecutive_failures(progress_history)
        strand_state.consecutive_failures = failures
        tension += failures * CONSECUTIVE_FAILS_MODIFIER

        # Clamp to bounds
        tension = max(self.min_tension, min(self.max_tension, tension))

        return round(tension, 2)

    def _has_blocked_dependencies(
        self,
        feature: Feature,
        dependency_states: dict[str, FeatureStatus]
    ) -> bool:
        """Check if any dependency is not passing."""
        for dep_id in feature.depends_on:
            dep_status = dependency_states.get(dep_id)
            if dep_status != FeatureStatus.PASSING:
                return True
        return False

    def _detect_regression(
        self,
        feature: Feature,
        progress_history: list[ProgressEntry]
    ) -> bool:
        """Detect if feature was passing but is now failing."""
        if feature.status != FeatureStatus.FAILING:
            return False

        # Look for recent PASS followed by FAIL
        feature_progress = [
            p for p in progress_history
            if p.feature_id == feature.id
        ]

        if len(feature_progress) < 2:
            return False

        # Check last two entries
        recent = sorted(feature_progress, key=lambda p: p.timestamp, reverse=True)[:2]
        if len(recent) >= 2:
            if recent[0].action == ProgressAction.FAILED and recent[1].action == ProgressAction.COMPLETED:
                return True

        return False

    def _is_stale(self, strand_state: StrandState) -> bool:
        """Check if feature hasn't been updated recently."""
        age = datetime.now() - strand_state.last_updated
        return age > self.stale_threshold

    def _compute_velocity(self, progress_history: list[ProgressEntry]) -> float:
        """
        Compute progress velocity.
        
        Positive = recent progress
        Negative = stalled
        Zero = neutral
        """
        if not progress_history:
            return 0.0

        # Count actions in last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [p for p in progress_history if p.timestamp > cutoff]

        if not recent:
            return -0.5

        # Weight by outcome
        score = 0.0
        for p in recent:
            if p.action == ProgressAction.COMPLETED:
                score += 1.0
            elif p.action == ProgressAction.FAILED:
                score -= 0.5

        return score

    def _count_consecutive_failures(self, progress_history: list[ProgressEntry]) -> int:
        """Count consecutive failures from most recent."""
        sorted_history = sorted(progress_history, key=lambda p: p.timestamp, reverse=True)

        count = 0
        for p in sorted_history:
            if p.action == ProgressAction.FAILED:
                count += 1
            else:
                break

        return count


# ============================================================================
# Loom State
# ============================================================================

@dataclass
class LoomState:
    """
    Overall state of the weaving loom.
    
    Aggregates tension across all strands.
    """
    strands: dict[str, StrandState] = field(default_factory=dict)
    total_tension: float = 0.0
    average_tension: float = 5.0
    max_tension_feature: str | None = None

    # Loom health
    is_stable: bool = True      # All tensions < 7
    is_critical: bool = False   # Any tension > 9
    is_complete: bool = False   # All features passing

    def update_aggregates(self):
        """Recompute aggregate metrics."""
        if not self.strands:
            return

        tensions = [s.tension for s in self.strands.values()]
        self.total_tension = sum(tensions)
        self.average_tension = self.total_tension / len(tensions)

        # Find max
        max_strand = max(self.strands.values(), key=lambda s: s.tension)
        self.max_tension_feature = max_strand.feature_id

        # Health checks
        self.is_stable = all(s.tension < 7.0 for s in self.strands.values())
        self.is_critical = any(s.tension > 9.0 for s in self.strands.values())
        self.is_complete = all(s.tension < 1.0 for s in self.strands.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "strands": {k: v.to_dict() for k, v in self.strands.items()},
            "total_tension": self.total_tension,
            "average_tension": self.average_tension,
            "max_tension_feature": self.max_tension_feature,
            "is_stable": self.is_stable,
            "is_critical": self.is_critical,
            "is_complete": self.is_complete
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LoomState':
        """Deserialize from storage."""
        loom = cls()
        loom.strands = {
            k: StrandState.from_dict(v)
            for k, v in data.get("strands", {}).items()
        }
        loom.total_tension = data.get("total_tension", 0.0)
        loom.average_tension = data.get("average_tension", 5.0)
        loom.max_tension_feature = data.get("max_tension_feature")
        loom.is_stable = data.get("is_stable", True)
        loom.is_critical = data.get("is_critical", False)
        loom.is_complete = data.get("is_complete", False)
        return loom


# ============================================================================
# Tension Updater
# ============================================================================

class TensionUpdater:
    """
    Updates strand tensions based on domain state.
    
    Call after each worker run to recompute tensions.
    """

    def __init__(self):
        self.model = TensionModel()

    def update_all(
        self,
        features: list[Feature],
        progress_history: list[ProgressEntry],
        loom_state: LoomState | None = None
    ) -> LoomState:
        """
        Update tensions for all features.
        
        Returns updated LoomState.
        """
        if loom_state is None:
            loom_state = LoomState()

        # Build dependency state map
        dep_states = {f.id: f.status for f in features}

        for feature in features:
            # Get or create strand state
            if feature.id not in loom_state.strands:
                loom_state.strands[feature.id] = StrandState(feature_id=feature.id)

            strand = loom_state.strands[feature.id]

            # Filter progress for this feature
            feature_progress = [
                p for p in progress_history
                if p.feature_id == feature.id
            ]

            # Compute new tension
            new_tension = self.model.compute_tension(
                feature=feature,
                strand_state=strand,
                progress_history=feature_progress,
                dependency_states=dep_states
            )

            # Update strand
            strand.tension = new_tension
            strand.tension_history.append(new_tension)
            strand.last_updated = datetime.now()

            # Track weft position (progress through the weave)
            if feature.status == FeatureStatus.PASSING:
                strand.weft_position = 1.0
            elif feature.status == FeatureStatus.FAILING:
                strand.weft_position = max(0, strand.weft_position - 0.1)

        # Update aggregates
        loom_state.update_aggregates()

        return loom_state
