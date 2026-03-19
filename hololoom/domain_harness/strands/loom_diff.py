"""
Symbolic Memory Diff Viewer

Compares two domain memory states and generates:
- Human-readable diff (what changed)
- Symbolic diff (how the loom moved)

The diff isn't just "field A changed to B" —
it's "the weave shifted, tension released, a knot tremored."

Usage:
    from hololoom.domain_harness.strands.loom_diff import diff_states, SymbolicDiff
    
    diff = diff_states("snapshot_old.json", "snapshot_new.json")
    for change in diff:
        print(f"{change.feature_id}: {change.symbol}")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Symbolic Shifts
# ============================================================================

class SymbolicShift(str, Enum):
    """Symbolic interpretation of state changes."""
    KNOT_TREMOR = "knot_tremor"         # Major disruption (Δ > 1.0)
    WARP_TUG = "warp_tug"               # Realignment (Δ > 0.3)
    SOFT_PULL = "soft_pull"             # Minor adjustment (Δ > 0)
    STILLNESS = "stillness"             # No change
    SLACK_RELEASE = "slack_release"     # Minor relaxation (Δ < 0)
    FRACTURE = "fracture"               # Major recoil (Δ < -1.0)
    STATUS_FLIP = "status_flip"         # Pass/fail transition
    EMERGENCE = "emergence"             # New feature appeared
    DISSOLUTION = "dissolution"         # Feature removed


SHIFT_SYMBOLS = {
    SymbolicShift.KNOT_TREMOR: "⚡ Knot tremor: weave disrupted.",
    SymbolicShift.WARP_TUG: "⟡ Warp tug: strands realigned.",
    SymbolicShift.SOFT_PULL: "~ Soft pull: tension eased.",
    SymbolicShift.STILLNESS: "• Stillness: the thread rests.",
    SymbolicShift.SLACK_RELEASE: "↓ Slack release: pressure fades.",
    SymbolicShift.FRACTURE: "❄ Fracture: thread recoils.",
    SymbolicShift.STATUS_FLIP: "◐ Status flip: fate reversed.",
    SymbolicShift.EMERGENCE: "✦ Emergence: new thread appears.",
    SymbolicShift.DISSOLUTION: "✧ Dissolution: thread fades from the weave."
}

SHIFT_NARRATIVES = {
    SymbolicShift.KNOT_TREMOR: [
        "The loom shudders. Something fundamental shifted.",
        "A tremor runs through the weave. The pattern reforms.",
        "The knot tightens violently. Attention required."
    ],
    SymbolicShift.WARP_TUG: [
        "The warp threads pull taut. Alignment emerges.",
        "A firm tug realigns the strands. Progress.",
        "The loom responds. The pattern clarifies."
    ],
    SymbolicShift.SOFT_PULL: [
        "A gentle adjustment. The weave settles.",
        "Barely perceptible, the threads shift.",
        "The loom breathes. Small movements."
    ],
    SymbolicShift.STILLNESS: [
        "The thread rests. No motion detected.",
        "Silence in the weave. The pattern holds.",
        "Nothing stirs. The loom waits."
    ],
    SymbolicShift.SLACK_RELEASE: [
        "Tension drains away. The thread relaxes.",
        "Pressure fades. Room to breathe.",
        "The strand loosens. Flexibility returns."
    ],
    SymbolicShift.FRACTURE: [
        "The thread snaps back. Ground lost.",
        "A violent recoil. The pattern tears.",
        "Fracture detected. The weave resists."
    ],
    SymbolicShift.STATUS_FLIP: [
        "Fate reverses. What was, is not. What wasn't, is.",
        "The thread changes color. A new truth emerges.",
        "Pass becomes fail, or fail becomes pass. The loom remembers both."
    ],
    SymbolicShift.EMERGENCE: [
        "A new thread appears in the warp. The pattern expands.",
        "From nothing, a strand. The weave grows.",
        "Emergence. The loom accepts a new possibility."
    ],
    SymbolicShift.DISSOLUTION: [
        "The thread fades. The pattern contracts.",
        "What was woven, unwoven. The strand disappears.",
        "Dissolution. The loom releases what no longer serves."
    ]
}


# ============================================================================
# Diff Result Types
# ============================================================================

@dataclass
class FeatureDiff:
    """Diff for a single feature."""
    feature_id: str
    shift: SymbolicShift
    symbol: str
    narrative: str

    # Numeric changes
    tension_before: float | None = None
    tension_after: float | None = None
    tension_delta: float = 0.0

    # Status changes
    status_before: str | None = None
    status_after: str | None = None
    status_changed: bool = False

    # Metadata
    is_new: bool = False
    is_removed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "shift": self.shift.value,
            "symbol": self.symbol,
            "narrative": self.narrative,
            "tension_before": self.tension_before,
            "tension_after": self.tension_after,
            "tension_delta": self.tension_delta,
            "status_before": self.status_before,
            "status_after": self.status_after,
            "status_changed": self.status_changed,
            "is_new": self.is_new,
            "is_removed": self.is_removed
        }


@dataclass
class SymbolicDiff:
    """Complete symbolic diff between two states."""
    timestamp: datetime = field(default_factory=datetime.now)
    features: list[FeatureDiff] = field(default_factory=list)

    # Aggregate metrics
    total_tension_before: float = 0.0
    total_tension_after: float = 0.0

    # Summary
    summary_narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "features": [f.to_dict() for f in self.features],
            "total_tension_before": self.total_tension_before,
            "total_tension_after": self.total_tension_after,
            "summary_narrative": self.summary_narrative
        }

    def render_ascii(self) -> str:
        """Render as ASCII art."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║              SYMBOLIC MEMORY DIFF                            ║",
            f"║  {self.timestamp.strftime('%Y-%m-%d %H:%M:%S'):^56}  ║",
            "╠══════════════════════════════════════════════════════════════╣"
        ]

        for f in self.features:
            delta_str = f"{f.tension_delta:+.2f}" if f.tension_delta else "  --"
            lines.append(f"║ {f.feature_id:8} Δ{delta_str:6} {f.symbol[:40]:40} ║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        total_delta = self.total_tension_after - self.total_tension_before
        lines.append(f"║  Total Tension: {self.total_tension_before:.1f} → {self.total_tension_after:.1f} (Δ{total_delta:+.1f})              ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append(self.summary_narrative)
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# Diff Calculator
# ============================================================================

class LoomDiffCalculator:
    """
    Calculates symbolic diffs between memory states.
    """

    def __init__(self):
        self._narrative_index = {}

    def calculate_shift(self, delta: float, status_changed: bool) -> SymbolicShift:
        """Determine symbolic shift from numeric delta."""
        if status_changed:
            return SymbolicShift.STATUS_FLIP

        if delta > 1.0:
            return SymbolicShift.KNOT_TREMOR
        elif delta > 0.3:
            return SymbolicShift.WARP_TUG
        elif delta > 0.0:
            return SymbolicShift.SOFT_PULL
        elif delta == 0.0:
            return SymbolicShift.STILLNESS
        elif delta > -1.0:
            return SymbolicShift.SLACK_RELEASE
        else:
            return SymbolicShift.FRACTURE

    def get_narrative(self, shift: SymbolicShift) -> str:
        """Get rotating narrative for shift type."""
        narratives = SHIFT_NARRATIVES.get(shift, ["The loom moves."])
        idx = self._narrative_index.get(shift, 0)
        narrative = narratives[idx % len(narratives)]
        self._narrative_index[shift] = idx + 1
        return narrative

    def diff_features(
        self,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]]
    ) -> list[FeatureDiff]:
        """Calculate diff for feature lists."""
        diffs = []

        # Index by ID
        before_map = {f["id"]: f for f in before}
        after_map = {f["id"]: f for f in after}

        all_ids = set(before_map.keys()) | set(after_map.keys())

        for fid in sorted(all_ids):
            fb = before_map.get(fid)
            fa = after_map.get(fid)

            if fb is None:
                # New feature
                shift = SymbolicShift.EMERGENCE
                diffs.append(FeatureDiff(
                    feature_id=fid,
                    shift=shift,
                    symbol=SHIFT_SYMBOLS[shift],
                    narrative=self.get_narrative(shift),
                    tension_after=fa.get("strand_state", {}).get("tension", 5.0),
                    status_after=fa.get("status"),
                    is_new=True
                ))
            elif fa is None:
                # Removed feature
                shift = SymbolicShift.DISSOLUTION
                diffs.append(FeatureDiff(
                    feature_id=fid,
                    shift=shift,
                    symbol=SHIFT_SYMBOLS[shift],
                    narrative=self.get_narrative(shift),
                    tension_before=fb.get("strand_state", {}).get("tension", 5.0),
                    status_before=fb.get("status"),
                    is_removed=True
                ))
            else:
                # Both exist - calculate diff
                tb = fb.get("strand_state", {}).get("tension", 5.0)
                ta = fa.get("strand_state", {}).get("tension", 5.0)
                delta = ta - tb

                sb = fb.get("status")
                sa = fa.get("status")
                status_changed = sb != sa

                shift = self.calculate_shift(delta, status_changed)

                diffs.append(FeatureDiff(
                    feature_id=fid,
                    shift=shift,
                    symbol=SHIFT_SYMBOLS[shift],
                    narrative=self.get_narrative(shift),
                    tension_before=tb,
                    tension_after=ta,
                    tension_delta=delta,
                    status_before=sb,
                    status_after=sa,
                    status_changed=status_changed
                ))

        return diffs

    def generate_summary(self, diffs: list[FeatureDiff]) -> str:
        """Generate overall narrative summary."""
        shifts = [d.shift for d in diffs]

        # Count shift types
        from collections import Counter
        counts = Counter(shifts)

        if counts.get(SymbolicShift.KNOT_TREMOR, 0) >= 2:
            return "🌊 The loom trembles. Multiple disruptions ripple through the weave."

        if counts.get(SymbolicShift.FRACTURE, 0) >= 1:
            return "❄️ Fractures detected. The pattern strains against itself."

        if counts.get(SymbolicShift.WARP_TUG, 0) >= 3:
            return "⚡ Strong realignment. The weave tightens toward coherence."

        if counts.get(SymbolicShift.STATUS_FLIP, 0) >= 1:
            return "◐ Fate reversed. The loom remembers transformation."

        if all(s == SymbolicShift.STILLNESS for s in shifts):
            return "🌙 Perfect stillness. The weave rests unchanged."

        total_delta = sum(d.tension_delta for d in diffs)
        if total_delta < -2:
            return "↓ The weave relaxes. Accumulated tension drains away."
        elif total_delta > 2:
            return "↑ Tension builds. The pattern tightens."

        return "~ The loom breathes. Small adjustments across the weave."


# ============================================================================
# Public Interface
# ============================================================================

def load_state(path: Path) -> list[dict[str, Any]]:
    """Load features from a snapshot file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("features", [])


def diff_states(
    before_path: Path,
    after_path: Path
) -> SymbolicDiff:
    """
    Calculate symbolic diff between two state files.
    
    Args:
        before_path: Path to earlier snapshot
        after_path: Path to later snapshot
        
    Returns:
        SymbolicDiff with all changes
    """
    before = load_state(Path(before_path))
    after = load_state(Path(after_path))

    calculator = LoomDiffCalculator()
    feature_diffs = calculator.diff_features(before, after)

    # Calculate totals
    total_before = sum(
        f.get("strand_state", {}).get("tension", 5.0)
        for f in before
    )
    total_after = sum(
        f.get("strand_state", {}).get("tension", 5.0)
        for f in after
    )

    summary = calculator.generate_summary(feature_diffs)

    return SymbolicDiff(
        features=feature_diffs,
        total_tension_before=total_before,
        total_tension_after=total_after,
        summary_narrative=summary
    )


def diff_in_memory(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]]
) -> SymbolicDiff:
    """
    Calculate symbolic diff from in-memory feature lists.
    """
    calculator = LoomDiffCalculator()
    feature_diffs = calculator.diff_features(before, after)

    total_before = sum(
        f.get("strand_state", {}).get("tension", 5.0)
        for f in before
    )
    total_after = sum(
        f.get("strand_state", {}).get("tension", 5.0)
        for f in after
    )

    summary = calculator.generate_summary(feature_diffs)

    return SymbolicDiff(
        features=feature_diffs,
        total_tension_before=total_before,
        total_tension_after=total_after,
        summary_narrative=summary
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python loom_diff.py <before.json> <after.json>")
        sys.exit(1)

    diff = diff_states(sys.argv[1], sys.argv[2])
    print(diff.render_ascii())
