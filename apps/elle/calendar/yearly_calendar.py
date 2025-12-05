"""
Yearly Calendar System for Farm & Kitchen Cooperative

Manages annual farm calendar with:
- Seasonal production windows
- Planting and harvest schedules
- Market schedules
- Financial deadlines (taxes, insurance)
- Regulatory events (inspections, permits)
- Equipment maintenance schedules

Created: 2025-11-16
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EventType(Enum):
    """Types of calendar events"""
    PLANTING = "planting"
    HARVEST = "harvest"
    PRODUCTION = "production"
    MARKET = "market"
    MAINTENANCE = "maintenance"
    REGULATORY = "regulatory"
    FINANCIAL = "financial"
    GENERAL = "general"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Season(Enum):
    """Farm seasons"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


@dataclass
class CalendarEvent:
    """Single calendar event"""
    id: str
    name: str
    type: EventType
    month: int
    day: int
    recurrence: str = "yearly"
    priority: EventPriority = EventPriority.MEDIUM
    notes: str = ""
    end_month: Optional[int] = None
    end_day: Optional[int] = None
    reminder_days: int = 7
    related_sops: List[str] = field(default_factory=list)
    related_products: List[str] = field(default_factory=list)

    @property
    def start_date(self) -> datetime:
        """Get start date for current year"""
        year = datetime.now().year
        return datetime(year, self.month, self.day)

    @property
    def end_date(self) -> Optional[datetime]:
        """Get end date if event spans multiple days"""
        if self.end_month and self.end_day:
            year = datetime.now().year
            return datetime(year, self.end_month, self.end_day)
        return None

    @property
    def is_active(self) -> bool:
        """Check if event is currently active"""
        today = datetime.now()
        if self.end_date:
            return self.start_date <= today <= self.end_date
        else:
            # Single-day event - active on that day only
            return today.date() == self.start_date.date()

    @property
    def is_upcoming(self) -> bool:
        """Check if event is upcoming (within next 30 days)"""
        today = datetime.now()
        days_until = (self.start_date - today).days
        return 0 < days_until <= 30

    @property
    def days_until(self) -> int:
        """Days until event starts"""
        today = datetime.now()
        return (self.start_date - today).days

    @property
    def days_remaining(self) -> Optional[int]:
        """Days remaining in multi-day event"""
        if self.end_date and self.is_active:
            today = datetime.now()
            return (self.end_date - today).days
        return None

    @property
    def needs_reminder(self) -> bool:
        """Check if reminder should be shown"""
        return 0 < self.days_until <= self.reminder_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "month": self.month,
            "day": self.day,
            "priority": self.priority.value,
            "notes": self.notes,
            "end_month": self.end_month,
            "end_day": self.end_day,
            "recurrence": self.recurrence,
            "reminder_days": self.reminder_days,
            "related_sops": self.related_sops,
            "related_products": self.related_products,
            "is_active": self.is_active,
            "is_upcoming": self.is_upcoming,
            "days_until": self.days_until,
            "days_remaining": self.days_remaining,
        }


@dataclass
class SeasonalWindow:
    """Seasonal production window"""
    season: Season
    months: List[int]
    focus: str
    products: List[str]

    @property
    def is_active(self) -> bool:
        """Check if this season is currently active"""
        current_month = datetime.now().month
        return current_month in self.months

    @property
    def months_remaining(self) -> int:
        """Months remaining in this season"""
        current_month = datetime.now().month
        if current_month not in self.months:
            return 0

        # Count months from current to end of season
        remaining = [m for m in self.months if m >= current_month]
        return len(remaining)


class YearlyCalendar:
    """Manages farm yearly calendar"""

    def __init__(self, calendar_path: str = "coz/yearly_calendar.json"):
        self.calendar_path = Path(calendar_path)
        self.events: List[CalendarEvent] = []
        self.seasonal_windows: Dict[Season, SeasonalWindow] = {}
        self.calendar_year: int = datetime.now().year
        self.farm_name: str = ""

        # Load calendar
        self._load_calendar()

    def _load_calendar(self):
        """Load calendar from JSON file"""
        if not self.calendar_path.exists():
            raise FileNotFoundError(f"Calendar file not found: {self.calendar_path}")

        with open(self.calendar_path) as f:
            data = json.load(f)

        self.calendar_year = data.get("calendar_year", datetime.now().year)
        self.farm_name = data.get("farm_name", "Farm & Kitchen Cooperative")

        # Load events
        for event_data in data.get("events", []):
            event = CalendarEvent(
                id=event_data["id"],
                name=event_data["name"],
                type=EventType(event_data["type"].lower()),
                month=event_data["month"],
                day=event_data["day"],
                recurrence=event_data.get("recurrence", "yearly"),
                priority=EventPriority(event_data.get("priority", "medium").lower()),
                notes=event_data.get("notes", ""),
                end_month=event_data.get("end_month"),
                end_day=event_data.get("end_day"),
                reminder_days=event_data.get("reminder_days", 7),
                related_sops=event_data.get("related_sops", []),
                related_products=event_data.get("related_products", []),
            )
            self.events.append(event)

        # Load seasonal windows
        for season_name, season_data in data.get("seasonal_production_windows", {}).items():
            season = Season(season_name.lower())
            window = SeasonalWindow(
                season=season,
                months=season_data["months"],
                focus=season_data["focus"],
                products=season_data["products"],
            )
            self.seasonal_windows[season] = window

    def get_current_season(self) -> Optional[SeasonalWindow]:
        """Get currently active season"""
        for window in self.seasonal_windows.values():
            if window.is_active:
                return window
        return None

    def get_active_events(self) -> List[CalendarEvent]:
        """Get all currently active events"""
        return [e for e in self.events if e.is_active]

    def get_upcoming_events(self, days: int = 30) -> List[CalendarEvent]:
        """Get events upcoming in next N days"""
        today = datetime.now()
        upcoming = []
        for event in self.events:
            days_until = event.days_until
            if 0 < days_until <= days:
                upcoming.append(event)

        # Sort by days until
        upcoming.sort(key=lambda e: e.days_until)
        return upcoming

    def get_events_by_type(self, event_type: EventType) -> List[CalendarEvent]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.type == event_type]

    def get_events_by_priority(self, priority: EventPriority) -> List[CalendarEvent]:
        """Get all events with specific priority"""
        return [e for e in self.events if e.priority == priority]

    def get_critical_deadlines(self) -> List[CalendarEvent]:
        """Get all critical upcoming deadlines"""
        critical = self.get_events_by_priority(EventPriority.CRITICAL)
        return [e for e in critical if e.is_upcoming or e.is_active]

    def get_reminders(self) -> List[CalendarEvent]:
        """Get events needing reminders"""
        return [e for e in self.events if e.needs_reminder]

    def get_production_schedule(self) -> List[CalendarEvent]:
        """Get production-related events (planting, harvest, production)"""
        production_types = [EventType.PLANTING, EventType.HARVEST, EventType.PRODUCTION]
        return [e for e in self.events if e.type in production_types]

    def get_events_by_month(self, month: int) -> List[CalendarEvent]:
        """Get all events in a specific month"""
        events = []
        for event in self.events:
            # Check if event starts in this month
            if event.month == month:
                events.append(event)
            # Check if multi-day event spans this month
            elif event.end_month and event.month < month <= event.end_month:
                events.append(event)

        return events

    def get_calendar_summary(self) -> Dict[str, Any]:
        """Get comprehensive calendar summary"""
        current_season = self.get_current_season()
        active_events = self.get_active_events()
        upcoming_events = self.get_upcoming_events(30)
        reminders = self.get_reminders()
        critical = self.get_critical_deadlines()

        return {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_season": {
                "season": current_season.season.value if current_season else None,
                "focus": current_season.focus if current_season else None,
                "months_remaining": current_season.months_remaining if current_season else 0,
                "products": current_season.products if current_season else [],
            } if current_season else None,
            "active_events": [e.to_dict() for e in active_events],
            "upcoming_events": [e.to_dict() for e in upcoming_events[:10]],  # Next 10
            "reminders": [e.to_dict() for e in reminders],
            "critical_deadlines": [e.to_dict() for e in critical],
            "total_events": len(self.events),
        }

    def format_summary_text(self) -> str:
        """Format calendar summary as text"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"YEARLY CALENDAR - {self.farm_name}")
        lines.append(f"{datetime.now().strftime('%B %d, %Y')}")
        lines.append("=" * 70)
        lines.append("")

        # Current season
        current_season = self.get_current_season()
        if current_season:
            lines.append(f"🍂 CURRENT SEASON: {current_season.season.value.upper()}")
            lines.append(f"   Focus: {current_season.focus}")
            lines.append(f"   Products: {', '.join(current_season.products)}")
            lines.append(f"   {current_season.months_remaining} month(s) remaining")
            lines.append("")

        # Critical deadlines
        critical = self.get_critical_deadlines()
        if critical:
            lines.append("🚨 CRITICAL DEADLINES:")
            for event in critical:
                lines.append(f"   • {event.name} - {event.start_date.strftime('%b %d')} ({event.days_until} days)")
                if event.notes:
                    lines.append(f"     {event.notes}")
            lines.append("")

        # Reminders
        reminders = self.get_reminders()
        if reminders:
            lines.append("⏰ UPCOMING REMINDERS:")
            for event in reminders:
                lines.append(f"   • {event.name} - {event.start_date.strftime('%b %d')} ({event.days_until} days)")
            lines.append("")

        # Active events
        active = self.get_active_events()
        if active:
            lines.append("✅ ACTIVE NOW:")
            for event in active:
                duration_str = f"{event.days_remaining} days remaining" if event.days_remaining else "Today"
                lines.append(f"   • {event.name} ({duration_str})")
                if event.notes:
                    lines.append(f"     {event.notes}")
            lines.append("")

        # Upcoming events
        upcoming = self.get_upcoming_events(30)
        if upcoming and not reminders:  # Don't duplicate reminders
            lines.append("📅 NEXT 30 DAYS:")
            for event in upcoming[:5]:  # Top 5
                lines.append(f"   • {event.name} - {event.start_date.strftime('%b %d')} ({event.days_until} days)")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """Demo: Display calendar summary"""
    calendar = YearlyCalendar()

    print(calendar.format_summary_text())

    # Show production schedule
    print("\n📦 PRODUCTION SCHEDULE:")
    print("=" * 70)

    production_events = calendar.get_production_schedule()
    for event in sorted(production_events, key=lambda e: (e.month, e.day)):
        status = "🟢 ACTIVE" if event.is_active else "⏳ UPCOMING" if event.is_upcoming else ""
        print(f"{event.start_date.strftime('%b %d')}: {event.name} {status}")
        if event.related_sops or event.related_products:
            related = event.related_sops + event.related_products
            print(f"   Related: {', '.join(related)}")
        print()


if __name__ == "__main__":
    main()
