"""
Ritual Scheduler - Cosmic Cycles for the Loom

The loom responds to celestial time:
- Moon phases (new moon, full moon)
- Solar events (solstices, equinoxes)
- Weekly rhythms
- Custom symbolic events

Rituals add symbolic state layers and can trigger:
- Automatic tension adjustments
- Progress log entries
- Loom state snapshots
- Narrative generation

The loom is not just a machine—it breathes with the cosmos.

Usage:
    from hololoom.domain_harness.strands.ritual_scheduler import (
        RitualScheduler, RitualType, check_rituals
    )
    
    scheduler = RitualScheduler()
    rituals = scheduler.check_and_execute()
    for ritual in rituals:
        print(ritual.proclamation)
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Ritual Types
# ============================================================================

class RitualType(str, Enum):
    """Types of rituals the loom can perform."""
    # Lunar
    NEW_MOON = "new_moon"
    FULL_MOON = "full_moon"
    WAXING_QUARTER = "waxing_quarter"
    WANING_QUARTER = "waning_quarter"

    # Solar
    VERNAL_EQUINOX = "vernal_equinox"
    SUMMER_SOLSTICE = "summer_solstice"
    AUTUMNAL_EQUINOX = "autumnal_equinox"
    WINTER_SOLSTICE = "winter_solstice"

    # Weekly
    MONDAY_OPENING = "monday_opening"
    FRIDAY_CLOSING = "friday_closing"

    # Custom
    CUSTOM = "custom"
    ANNIVERSARY = "anniversary"
    MILESTONE = "milestone"


# ============================================================================
# Ritual Proclamations
# ============================================================================

RITUAL_PROCLAMATIONS = {
    RitualType.NEW_MOON: [
        "🌑 New Moon: The loom begins anew. Seeds planted in darkness.",
        "🌑 Dark of the Moon: A time for quiet planning. The weave rests.",
        "🌑 Luna Absentia: What is unseen prepares to emerge."
    ],
    RitualType.FULL_MOON: [
        "🌕 Full Moon: The loom breathes at full tide. Illumination.",
        "🌕 Luna Plena: The weave stands revealed in silver light.",
        "🌕 Moon Ritual: What was hidden is now seen. The pattern clarifies."
    ],
    RitualType.WAXING_QUARTER: [
        "🌓 First Quarter: Building momentum. The weave grows.",
        "🌓 Waxing Moon: Energy increases. Progress accelerates.",
    ],
    RitualType.WANING_QUARTER: [
        "🌗 Last Quarter: Releasing what no longer serves.",
        "🌗 Waning Moon: Energy contracts. Time to consolidate.",
    ],
    RitualType.VERNAL_EQUINOX: [
        "☀️ Vernal Equinox: Light and dark in balance. The loom awakens.",
        "🌱 Spring Turning: New threads emerge. The weave begins its growth cycle.",
    ],
    RitualType.SUMMER_SOLSTICE: [
        "☀️ Summer Solstice: Peak of light. The loom works at full power.",
        "🌞 Midsummer: Maximum illumination. The pattern burns bright.",
    ],
    RitualType.AUTUMNAL_EQUINOX: [
        "🍂 Autumnal Equinox: Balance returns. Harvest what was woven.",
        "☀️ Fall Turning: Light fades. Time to gather the threads.",
    ],
    RitualType.WINTER_SOLSTICE: [
        "❄️ Winter Solstice: Longest night. The loom dreams.",
        "🌙 Midwinter: In deepest dark, the light returns. Rebirth begins.",
    ],
    RitualType.MONDAY_OPENING: [
        "📅 Monday Opening: The week's weave begins. Threads awakening.",
        "🌅 Week Start Ritual: Fresh slate. New patterns possible.",
    ],
    RitualType.FRIDAY_CLOSING: [
        "📅 Friday Closing: The week's weave rests. Threads settle.",
        "🌆 Week End Ritual: Honor what was woven. Release what wasn't.",
    ],
    RitualType.MILESTONE: [
        "⭐ Milestone Reached: A significant knot in the weave.",
        "🎯 Achievement Unlocked: The loom celebrates progress.",
    ],
    RitualType.ANNIVERSARY: [
        "🎂 Anniversary: Another cycle complete. The loom remembers.",
        "📆 Yearly Return: What was started, continues. The pattern endures.",
    ],
}

RITUAL_EFFECTS = {
    RitualType.NEW_MOON: {"tension_mod": -0.5, "symbol": "🌑"},
    RitualType.FULL_MOON: {"tension_mod": 0.0, "symbol": "🌕"},
    RitualType.WAXING_QUARTER: {"tension_mod": 0.3, "symbol": "🌓"},
    RitualType.WANING_QUARTER: {"tension_mod": -0.3, "symbol": "🌗"},
    RitualType.VERNAL_EQUINOX: {"tension_mod": 0.0, "symbol": "🌱"},
    RitualType.SUMMER_SOLSTICE: {"tension_mod": 0.5, "symbol": "🌞"},
    RitualType.AUTUMNAL_EQUINOX: {"tension_mod": 0.0, "symbol": "🍂"},
    RitualType.WINTER_SOLSTICE: {"tension_mod": -0.5, "symbol": "❄️"},
    RitualType.MONDAY_OPENING: {"tension_mod": 0.2, "symbol": "🌅"},
    RitualType.FRIDAY_CLOSING: {"tension_mod": -0.2, "symbol": "🌆"},
    RitualType.MILESTONE: {"tension_mod": -1.0, "symbol": "⭐"},
    RitualType.ANNIVERSARY: {"tension_mod": 0.0, "symbol": "🎂"},
}


# ============================================================================
# Astronomical Calculations
# ============================================================================

def moon_phase(dt: datetime) -> float:
    """
    Calculate moon phase (0-1).
    
    0.0 = New Moon
    0.25 = First Quarter
    0.5 = Full Moon
    0.75 = Last Quarter
    
    Based on simplified lunation calculation.
    """
    # Reference new moon: January 6, 2000, 18:14 UTC
    ref = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Synodic month = 29.530588853 days
    synodic_month = 29.530588853

    diff = (dt - ref).total_seconds() / 86400  # days
    lunations = diff / synodic_month

    return lunations % 1


def moon_phase_name(phase: float, tolerance: float = 0.03) -> RitualType | None:
    """
    Get ritual type if near a significant moon phase.
    """
    if phase < tolerance or phase > (1 - tolerance):
        return RitualType.NEW_MOON
    if abs(phase - 0.5) < tolerance:
        return RitualType.FULL_MOON
    if abs(phase - 0.25) < tolerance:
        return RitualType.WAXING_QUARTER
    if abs(phase - 0.75) < tolerance:
        return RitualType.WANING_QUARTER
    return None


def solar_event(dt: datetime) -> RitualType | None:
    """
    Check if date is a solar event (equinox/solstice).
    
    Uses approximate dates (actual times vary by year).
    """
    month, day = dt.month, dt.day

    # Approximate dates (within 1 day tolerance)
    events = {
        (3, 20): RitualType.VERNAL_EQUINOX,
        (3, 21): RitualType.VERNAL_EQUINOX,
        (6, 20): RitualType.SUMMER_SOLSTICE,
        (6, 21): RitualType.SUMMER_SOLSTICE,
        (9, 22): RitualType.AUTUMNAL_EQUINOX,
        (9, 23): RitualType.AUTUMNAL_EQUINOX,
        (12, 21): RitualType.WINTER_SOLSTICE,
        (12, 22): RitualType.WINTER_SOLSTICE,
    }

    return events.get((month, day))


def weekly_ritual(dt: datetime) -> RitualType | None:
    """Check for weekly rituals."""
    weekday = dt.weekday()
    hour = dt.hour

    # Monday morning (6-9 AM)
    if weekday == 0 and 6 <= hour <= 9:
        return RitualType.MONDAY_OPENING

    # Friday evening (16-19)
    if weekday == 4 and 16 <= hour <= 19:
        return RitualType.FRIDAY_CLOSING

    return None


# ============================================================================
# Ritual Data Structures
# ============================================================================

@dataclass
class Ritual:
    """A scheduled or triggered ritual."""
    ritual_type: RitualType
    triggered_at: datetime
    proclamation: str
    effects: dict[str, Any]
    symbol: str
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ritual_type": self.ritual_type.value,
            "triggered_at": self.triggered_at.isoformat(),
            "proclamation": self.proclamation,
            "effects": self.effects,
            "symbol": self.symbol,
            "executed": self.executed
        }


@dataclass
class RitualLog:
    """Log of executed rituals."""
    entries: list[Ritual] = field(default_factory=list)
    last_check: datetime | None = None

    def add(self, ritual: Ritual):
        self.entries.append(ritual)
        self.last_check = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "last_check": self.last_check.isoformat() if self.last_check else None
        }

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RitualLog':
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        log = cls()
        log.last_check = datetime.fromisoformat(data["last_check"]) if data.get("last_check") else None
        for e in data.get("entries", []):
            log.entries.append(Ritual(
                ritual_type=RitualType(e["ritual_type"]),
                triggered_at=datetime.fromisoformat(e["triggered_at"]),
                proclamation=e["proclamation"],
                effects=e["effects"],
                symbol=e["symbol"],
                executed=e.get("executed", True)
            ))
        return log


# ============================================================================
# Ritual Scheduler
# ============================================================================

class RitualScheduler:
    """
    Schedules and executes loom rituals based on cosmic and custom cycles.
    """

    def __init__(
        self,
        log_path: Path | None = None,
        cooldown_hours: int = 20
    ):
        self.log_path = log_path or Path("ritual_log.json")
        self.cooldown_hours = cooldown_hours
        self.log = RitualLog.load(self.log_path) if self.log_path.exists() else RitualLog()
        self._proclamation_index = {}
        self.custom_rituals: list[Callable[[datetime], RitualType | None]] = []

    def _get_proclamation(self, ritual_type: RitualType) -> str:
        """Get rotating proclamation for ritual type."""
        proclamations = RITUAL_PROCLAMATIONS.get(ritual_type, ["A ritual occurs."])
        idx = self._proclamation_index.get(ritual_type, 0)
        self._proclamation_index[ritual_type] = idx + 1
        return proclamations[idx % len(proclamations)]

    def _in_cooldown(self, ritual_type: RitualType) -> bool:
        """Check if ritual is in cooldown period."""
        cutoff = datetime.now() - timedelta(hours=self.cooldown_hours)

        for entry in reversed(self.log.entries):
            if entry.ritual_type == ritual_type and entry.triggered_at > cutoff:
                return True
        return False

    def add_custom_ritual(
        self,
        checker: Callable[[datetime], RitualType | None]
    ):
        """Add custom ritual checker."""
        self.custom_rituals.append(checker)

    def check_rituals(self, dt: datetime | None = None) -> list[Ritual]:
        """
        Check for active rituals at the given time.
        
        Returns list of rituals that should be triggered.
        """
        dt = dt or datetime.now()
        triggered = []

        # Check lunar
        phase = moon_phase(dt)
        lunar = moon_phase_name(phase)
        if lunar and not self._in_cooldown(lunar):
            effects = RITUAL_EFFECTS.get(lunar, {})
            triggered.append(Ritual(
                ritual_type=lunar,
                triggered_at=dt,
                proclamation=self._get_proclamation(lunar),
                effects=effects,
                symbol=effects.get("symbol", "🌙")
            ))

        # Check solar
        solar = solar_event(dt)
        if solar and not self._in_cooldown(solar):
            effects = RITUAL_EFFECTS.get(solar, {})
            triggered.append(Ritual(
                ritual_type=solar,
                triggered_at=dt,
                proclamation=self._get_proclamation(solar),
                effects=effects,
                symbol=effects.get("symbol", "☀️")
            ))

        # Check weekly
        weekly = weekly_ritual(dt)
        if weekly and not self._in_cooldown(weekly):
            effects = RITUAL_EFFECTS.get(weekly, {})
            triggered.append(Ritual(
                ritual_type=weekly,
                triggered_at=dt,
                proclamation=self._get_proclamation(weekly),
                effects=effects,
                symbol=effects.get("symbol", "📅")
            ))

        # Check custom rituals
        for checker in self.custom_rituals:
            ritual_type = checker(dt)
            if ritual_type and not self._in_cooldown(ritual_type):
                effects = RITUAL_EFFECTS.get(ritual_type, {})
                triggered.append(Ritual(
                    ritual_type=ritual_type,
                    triggered_at=dt,
                    proclamation=self._get_proclamation(ritual_type),
                    effects=effects,
                    symbol=effects.get("symbol", "⭐")
                ))

        return triggered

    def execute(self, ritual: Ritual) -> Ritual:
        """
        Execute a ritual (mark as executed and log).
        """
        ritual.executed = True
        self.log.add(ritual)

        if self.log_path:
            self.log.save(self.log_path)

        return ritual

    def check_and_execute(
        self,
        dt: datetime | None = None
    ) -> list[Ritual]:
        """
        Check for rituals and execute any that trigger.
        
        Returns list of executed rituals.
        """
        triggered = self.check_rituals(dt)
        return [self.execute(r) for r in triggered]

    def render_ritual(self, ritual: Ritual) -> str:
        """Render ritual as formatted string."""
        lines = [
            "",
            f"{'═' * 60}",
            f"  {ritual.symbol}  {ritual.ritual_type.value.upper().replace('_', ' ')}",
            f"{'═' * 60}",
            "",
            f"  {ritual.proclamation}",
            "",
            f"  Triggered: {ritual.triggered_at.strftime('%Y-%m-%d %H:%M')}",
            f"  Effects: tension_mod={ritual.effects.get('tension_mod', 0):+.1f}",
            "",
            f"{'═' * 60}",
            ""
        ]
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def check_rituals(dt: datetime | None = None) -> list[Ritual]:
    """Quick check for active rituals."""
    scheduler = RitualScheduler()
    return scheduler.check_rituals(dt)


def get_moon_info(dt: datetime | None = None) -> dict[str, Any]:
    """Get current moon information."""
    dt = dt or datetime.now()
    phase = moon_phase(dt)

    # Phase names
    if phase < 0.125:
        name = "New Moon"
    elif phase < 0.25:
        name = "Waxing Crescent"
    elif phase < 0.375:
        name = "First Quarter"
    elif phase < 0.5:
        name = "Waxing Gibbous"
    elif phase < 0.625:
        name = "Full Moon"
    elif phase < 0.75:
        name = "Waning Gibbous"
    elif phase < 0.875:
        name = "Last Quarter"
    else:
        name = "Waning Crescent"

    return {
        "phase": phase,
        "name": name,
        "illumination": f"{abs(0.5 - phase) * 200:.0f}%" if phase <= 0.5 else f"{(1 - phase) * 200:.0f}%",
        "date": dt.isoformat()
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("🌙 Ritual Scheduler")
    print("=" * 40)

    # Moon info
    moon = get_moon_info()
    print(f"\nMoon Phase: {moon['name']} ({moon['phase']:.2f})")
    print(f"Illumination: {moon['illumination']}")

    # Check for rituals
    scheduler = RitualScheduler()
    rituals = scheduler.check_and_execute()

    if rituals:
        print("\n🔔 Active Rituals:")
        for r in rituals:
            print(scheduler.render_ritual(r))
    else:
        print("\n✨ No rituals active at this time.")

    # Test specific dates
    print("\n--- Testing Specific Dates ---")

    test_dates = [
        datetime(2025, 6, 21, 12, 0),  # Summer solstice
        datetime(2025, 12, 21, 12, 0),  # Winter solstice
        datetime(2025, 3, 20, 12, 0),   # Vernal equinox
    ]

    for dt in test_dates:
        solar = solar_event(dt)
        if solar:
            print(f"  {dt.strftime('%Y-%m-%d')}: {solar.value}")
