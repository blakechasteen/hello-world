"""
MirrorLog / Archive Eternal Integration

Binds symbolic loom actions to the mythic memory canon.
Every tension spike, regression omen, ritual event, or Court verdict
becomes a "page fragment" seeded into the eternal archive.

The loom doesn't just execute—it writes its own story.

This module:
- Converts loom events to narrative fragments
- Maintains the Archive Eternal (JSON-based canon)
- Generates cross-references between events
- Supports querying the archive by symbol/time/feature

Usage:
    from hololoom.domain_harness.strands.mirror_integration import (
        MirrorLog, CanonFragment, archive_event
    )
    
    mirror = MirrorLog()
    mirror.archive_weave(decoded_weave)
    mirror.archive_ritual(ritual)
    
    # Query
    fragments = mirror.query_by_symbol("regression")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from hololoom.domain_harness.strands.decode_weave import DecodedWeave
from hololoom.domain_harness.strands.loom_court import Verdict
from hololoom.domain_harness.strands.ritual_scheduler import Ritual
from hololoom.domain_harness.strands.tension_model import WeaveMotion

# ============================================================================
# Canon Fragment Types
# ============================================================================

class FragmentType(str, Enum):
    """Types of fragments in the Archive."""
    WEAVE = "weave"         # Worker action / weave motion
    RITUAL = "ritual"       # Cosmic/scheduled ritual
    VERDICT = "verdict"     # Court verdict
    OMEN = "omen"          # Warning / prediction
    MILESTONE = "milestone"  # Significant achievement
    CUSTOM = "custom"       # User-defined entry


# ============================================================================
# Narrative Templates
# ============================================================================

WEAVE_NARRATIVES = {
    WeaveMotion.WARP_TUG: """
On {timestamp}, the loom stirred.
Thread {feature_id} pulled taut with purpose.

    "{effect}"

The weave tightened. Progress recorded.
Tension: {tension_before:.1f} → {tension_after:.1f}
""",

    WeaveMotion.WEFT_WANDER: """
On {timestamp}, the shuttle wandered.
Thread {feature_id} sought its path through uncertainty.

    "{effect}"

The weave loosened, making room for what might be.
Tension: {tension_before:.1f} → {tension_after:.1f}
""",

    WeaveMotion.REGRESSION_SPIKE: """
On {timestamp}, the Archive witnessed a turning back.
Thread {feature_id} unraveled what was woven.

    "{effect}"

The loom remembers. This regression is recorded.
Tension: {tension_before:.1f} → {tension_after:.1f} (↑ SPIKE)
""",

    WeaveMotion.TENSION_RELEASE: """
On {timestamp}, release came to the weave.
Thread {feature_id} found its resting place.

    "{effect}"

The pattern holds. Stability achieved.
Tension: {tension_before:.1f} → {tension_after:.1f}
""",

    WeaveMotion.THREAD_SNAP: """
⚠️ CRITICAL ENTRY ⚠️

On {timestamp}, a thread broke.
{feature_id} exceeded the loom's tolerance.

    "{effect}"

This rupture is recorded for all who follow.
The Archive marks this moment with solemnity.
Tension: {tension_before:.1f} → {tension_after:.1f} (CRITICAL)
""",
}

RITUAL_NARRATIVE = """
═══════════════════════════════════════════
On {timestamp}, the cosmic wheel turned.

    {symbol} {ritual_type}

    "{proclamation}"

The loom aligned with greater cycles.
Effects: tension_mod={tension_mod:+.1f}
═══════════════════════════════════════════
"""

VERDICT_NARRATIVE = """
╔════════════════════════════════════════╗
║        COURT OF THREADS - VERDICT      ║
╠════════════════════════════════════════╣

Case: {conflict_id}
Date: {timestamp}

    "{proclamation}"

Ruling: {ruling}

Affected Threads: {affected_features}
Actions Decreed: {actions}

The Archive preserves this judgment.
╚════════════════════════════════════════╝
"""


# ============================================================================
# Canon Fragment
# ============================================================================

@dataclass
class CanonFragment:
    """
    A single entry in the Archive Eternal.
    """
    fragment_id: str
    fragment_type: FragmentType
    timestamp: datetime
    narrative: str

    # Metadata
    feature_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Cross-references
    related_fragments: list[str] = field(default_factory=list)

    # Raw data (for querying)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "fragment_type": self.fragment_type.value,
            "timestamp": self.timestamp.isoformat(),
            "narrative": self.narrative,
            "feature_ids": self.feature_ids,
            "symbols": self.symbols,
            "tags": self.tags,
            "related_fragments": self.related_fragments,
            "raw_data": self.raw_data
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CanonFragment':
        return cls(
            fragment_id=data["fragment_id"],
            fragment_type=FragmentType(data["fragment_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            narrative=data["narrative"],
            feature_ids=data.get("feature_ids", []),
            symbols=data.get("symbols", []),
            tags=data.get("tags", []),
            related_fragments=data.get("related_fragments", []),
            raw_data=data.get("raw_data", {})
        )


# ============================================================================
# Archive Eternal
# ============================================================================

@dataclass
class ArchiveEternal:
    """
    The complete mythic memory canon.
    """
    entries: list[CanonFragment] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime | None = None

    # Index for fast lookup
    _by_id: dict[str, CanonFragment] = field(default_factory=dict, repr=False)
    _by_feature: dict[str, list[str]] = field(default_factory=lambda: {}, repr=False)
    _by_symbol: dict[str, list[str]] = field(default_factory=lambda: {}, repr=False)

    def add(self, fragment: CanonFragment):
        """Add fragment to archive."""
        self.entries.append(fragment)
        self.last_updated = datetime.now()

        # Update indices
        self._by_id[fragment.fragment_id] = fragment

        for fid in fragment.feature_ids:
            if fid not in self._by_feature:
                self._by_feature[fid] = []
            self._by_feature[fid].append(fragment.fragment_id)

        for sym in fragment.symbols:
            if sym not in self._by_symbol:
                self._by_symbol[sym] = []
            self._by_symbol[sym].append(fragment.fragment_id)

    def get(self, fragment_id: str) -> CanonFragment | None:
        """Get fragment by ID."""
        return self._by_id.get(fragment_id)

    def query_by_feature(self, feature_id: str) -> list[CanonFragment]:
        """Get all fragments mentioning a feature."""
        fragment_ids = self._by_feature.get(feature_id, [])
        return [self._by_id[fid] for fid in fragment_ids if fid in self._by_id]

    def query_by_symbol(self, symbol: str) -> list[CanonFragment]:
        """Get all fragments with a symbol."""
        # Partial match on symbols
        matches = []
        for sym, fids in self._by_symbol.items():
            if symbol.lower() in sym.lower():
                for fid in fids:
                    if fid in self._by_id:
                        matches.append(self._by_id[fid])
        return matches

    def query_by_time(
        self,
        start: datetime,
        end: datetime | None = None
    ) -> list[CanonFragment]:
        """Get fragments in time range."""
        end = end or datetime.now()
        return [
            f for f in self.entries
            if start <= f.timestamp <= end
        ]

    def recent(self, limit: int = 10) -> list[CanonFragment]:
        """Get most recent fragments."""
        return sorted(
            self.entries,
            key=lambda f: f.timestamp,
            reverse=True
        )[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    def save(self, path: Path):
        """Save archive to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ArchiveEternal':
        """Load archive from file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        archive = cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None
        )

        for entry_data in data.get("entries", []):
            fragment = CanonFragment.from_dict(entry_data)
            archive.add(fragment)

        return archive


# ============================================================================
# MirrorLog (Main Interface)
# ============================================================================

class MirrorLog:
    """
    Primary interface for the Archive Eternal.
    
    Converts loom events into narrative canon fragments.
    """

    def __init__(self, archive_path: Path | None = None):
        self.archive_path = archive_path or Path("mirror_canon.json")
        self.archive = ArchiveEternal.load(self.archive_path) if self.archive_path.exists() else ArchiveEternal()
        self._fragment_counter = len(self.archive.entries)

    def _generate_id(self, prefix: str = "FRAG") -> str:
        """Generate unique fragment ID."""
        self._fragment_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}-{timestamp}-{self._fragment_counter:04d}"

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract symbolic keywords from text."""
        symbols = []
        keywords = [
            "regression", "progress", "tension", "release",
            "knot", "weave", "thread", "snap", "tug",
            "ritual", "moon", "solstice", "equinox",
            "court", "verdict", "conflict"
        ]
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                symbols.append(kw)
        return symbols

    # ========================================================================
    # Archive Methods
    # ========================================================================

    def archive_weave(self, decoded: DecodedWeave) -> CanonFragment:
        """
        Archive a decoded weave motion.
        """
        template = WEAVE_NARRATIVES.get(decoded.motion, """
On {timestamp}, the loom moved.
Thread {feature_id}: {effect}
Tension: {tension_before:.1f} → {tension_after:.1f}
""")

        narrative = template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            feature_id=decoded.feature_id,
            effect=decoded.symbolic_effect,
            tension_before=decoded.new_tension - decoded.tension_delta,
            tension_after=decoded.new_tension
        )

        fragment = CanonFragment(
            fragment_id=self._generate_id("WEAVE"),
            fragment_type=FragmentType.WEAVE,
            timestamp=datetime.now(),
            narrative=narrative,
            feature_ids=[decoded.feature_id],
            symbols=self._extract_symbols(narrative) + [decoded.motion.value],
            tags=["weave", decoded.motion.value],
            raw_data=decoded.to_dict()
        )

        self.archive.add(fragment)
        self._save()

        return fragment

    def archive_ritual(self, ritual: Ritual) -> CanonFragment:
        """
        Archive a ritual event.
        """
        narrative = RITUAL_NARRATIVE.format(
            timestamp=ritual.triggered_at.strftime("%Y-%m-%d %H:%M"),
            symbol=ritual.symbol,
            ritual_type=ritual.ritual_type.value.upper().replace("_", " "),
            proclamation=ritual.proclamation,
            tension_mod=ritual.effects.get("tension_mod", 0)
        )

        fragment = CanonFragment(
            fragment_id=self._generate_id("RITUAL"),
            fragment_type=FragmentType.RITUAL,
            timestamp=ritual.triggered_at,
            narrative=narrative,
            symbols=self._extract_symbols(narrative) + [ritual.ritual_type.value],
            tags=["ritual", ritual.ritual_type.value],
            raw_data=ritual.to_dict()
        )

        self.archive.add(fragment)
        self._save()

        return fragment

    def archive_verdict(self, verdict: Verdict) -> CanonFragment:
        """
        Archive a Court verdict.
        """
        narrative = VERDICT_NARRATIVE.format(
            conflict_id=verdict.conflict_id,
            timestamp=verdict.issued_at.strftime("%Y-%m-%d %H:%M"),
            proclamation=verdict.proclamation,
            ruling=verdict.ruling,
            affected_features=", ".join(verdict.affected_features),
            actions="; ".join(verdict.recommended_actions[:3])
        )

        fragment = CanonFragment(
            fragment_id=self._generate_id("VERDICT"),
            fragment_type=FragmentType.VERDICT,
            timestamp=verdict.issued_at,
            narrative=narrative,
            feature_ids=verdict.affected_features,
            symbols=self._extract_symbols(narrative) + ["court", verdict.verdict_type.value],
            tags=["verdict", verdict.verdict_type.value],
            raw_data=verdict.to_dict()
        )

        self.archive.add(fragment)
        self._save()

        return fragment

    def archive_custom(
        self,
        title: str,
        narrative: str,
        feature_ids: list[str] | None = None,
        tags: list[str] | None = None
    ) -> CanonFragment:
        """
        Archive a custom event.
        """
        full_narrative = f"""
═══════════════════════════════════════════
{title}
═══════════════════════════════════════════

{narrative}

Recorded: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""

        fragment = CanonFragment(
            fragment_id=self._generate_id("CUSTOM"),
            fragment_type=FragmentType.CUSTOM,
            timestamp=datetime.now(),
            narrative=full_narrative,
            feature_ids=feature_ids or [],
            symbols=self._extract_symbols(narrative),
            tags=tags or ["custom"]
        )

        self.archive.add(fragment)
        self._save()

        return fragment

    def archive_milestone(
        self,
        title: str,
        description: str,
        feature_ids: list[str] | None = None
    ) -> CanonFragment:
        """
        Archive a milestone achievement.
        """
        narrative = f"""
⭐ MILESTONE ACHIEVED ⭐

{title}

{description}

The Archive marks this moment of significance.
Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""

        fragment = CanonFragment(
            fragment_id=self._generate_id("MILE"),
            fragment_type=FragmentType.MILESTONE,
            timestamp=datetime.now(),
            narrative=narrative,
            feature_ids=feature_ids or [],
            symbols=["milestone", "achievement"],
            tags=["milestone"]
        )

        self.archive.add(fragment)
        self._save()

        return fragment

    # ========================================================================
    # Query Methods
    # ========================================================================

    def query_by_feature(self, feature_id: str) -> list[CanonFragment]:
        """Get all entries for a feature."""
        return self.archive.query_by_feature(feature_id)

    def query_by_symbol(self, symbol: str) -> list[CanonFragment]:
        """Get all entries containing a symbol."""
        return self.archive.query_by_symbol(symbol)

    def recent(self, limit: int = 10) -> list[CanonFragment]:
        """Get recent entries."""
        return self.archive.recent(limit)

    def render_chronicle(self, limit: int = 20) -> str:
        """Render recent archive as chronicle."""
        recent = self.recent(limit)

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║              THE ARCHIVE ETERNAL - CHRONICLE                 ║",
            "╚══════════════════════════════════════════════════════════════╝",
            ""
        ]

        for fragment in recent:
            lines.append(fragment.narrative)
            lines.append("")
            lines.append("─" * 60)
            lines.append("")

        return "\n".join(lines)

    def _save(self):
        """Save archive to file."""
        if self.archive_path:
            self.archive.save(self.archive_path)


# ============================================================================
# Convenience Functions
# ============================================================================

def archive_event(
    event_type: str,
    data: Any,
    archive_path: Path | None = None
) -> CanonFragment:
    """
    Quick archive of any event.
    
    Automatically routes to appropriate handler.
    """
    mirror = MirrorLog(archive_path)

    if isinstance(data, DecodedWeave):
        return mirror.archive_weave(data)
    elif isinstance(data, Ritual):
        return mirror.archive_ritual(data)
    elif isinstance(data, Verdict):
        return mirror.archive_verdict(data)
    else:
        return mirror.archive_custom(
            title=event_type,
            narrative=str(data)
        )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    mirror = MirrorLog()

    # Demo: add some entries
    print("Creating demo archive entries...")

    # Custom entry
    mirror.archive_custom(
        title="System Initialization",
        narrative="The Archive Eternal awakens. The loom begins its chronicle.",
        tags=["init", "system"]
    )

    # Milestone
    mirror.archive_milestone(
        title="Archive System Online",
        description="The MirrorLog integration is complete. All loom events will now be preserved in the eternal chronicle.",
        feature_ids=["SYSTEM"]
    )

    # Print chronicle
    print(mirror.render_chronicle())
