"""
Demo: Yearly Calendar System

Demonstrates farm calendar with:
- Annual production cycles
- Financial deadlines
- Planting/harvest windows
- Market schedules
- Regulatory events

Created: 2025-11-16
"""

import asyncio
import sys
import calendar
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elle.calendar import YearlyCalendar, EventType
from elle.calendar.calendar_visual import render_yearly_calendar_html


def main():
    """Display calendar information"""

    print("=" * 80)
    print("FARM YEARLY CALENDAR SYSTEM")
    print("=" * 80)
    print()

    # Load calendar
    farm_calendar = YearlyCalendar()

    # Display text summary
    print(farm_calendar.format_summary_text())

    # Show production schedule
    print("\n" + "=" * 80)
    print("📦 COMPLETE PRODUCTION SCHEDULE")
    print("=" * 80)
    print()

    production_events = farm_calendar.get_production_schedule()
    for event in sorted(production_events, key=lambda e: (e.month, e.day)):
        status_icon = "🟢" if event.is_active else "⏳" if event.is_upcoming else "⏸️"
        status_text = "ACTIVE" if event.is_active else f"{event.days_until} days" if event.is_upcoming else ""

        print(f"{status_icon} {event.start_date.strftime('%b %d')}: {event.name}")
        print(f"   Type: {event.type.value.title()} | Priority: {event.priority.value.upper()}")

        if event.end_date:
            duration = (event.end_date - event.start_date).days
            print(f"   Duration: {duration} days ({event.start_date.strftime('%b %d')} - {event.end_date.strftime('%b %d')})")
        else:
            print(f"   Single day event")

        if status_text:
            print(f"   Status: {status_text}")

        if event.related_sops or event.related_products:
            related = event.related_sops + event.related_products
            print(f"   Related: {', '.join(related)}")

        if event.notes:
            print(f"   Note: {event.notes}")

        print()

    # Show events by type
    print("\n" + "=" * 80)
    print("📊 EVENTS BY TYPE")
    print("=" * 80)
    print()

    for event_type in EventType:
        type_events = farm_calendar.get_events_by_type(event_type)
        if type_events:
            icon = {
                EventType.PLANTING: "🌱",
                EventType.HARVEST: "🌾",
                EventType.PRODUCTION: "⚗️",
                EventType.MARKET: "🛒",
                EventType.MAINTENANCE: "🔧",
                EventType.REGULATORY: "📋",
                EventType.FINANCIAL: "💰",
                EventType.GENERAL: "📌",
            }.get(event_type, "")

            print(f"{icon} {event_type.value.upper()} ({len(type_events)} events)")
            for event in sorted(type_events, key=lambda e: (e.month, e.day)):
                print(f"   • {event.start_date.strftime('%b %d')}: {event.name}")
            print()

    # Show seasonal breakdown
    print("\n" + "=" * 80)
    print("🍂 SEASONAL BREAKDOWN")
    print("=" * 80)
    print()

    for season, window in farm_calendar.seasonal_windows.items():
        is_current = "← CURRENT" if window.is_active else ""
        print(f"{season.value.upper()} {is_current}")
        print(f"   Months: {', '.join(calendar.month_name[m] for m in window.months)}")
        print(f"   Focus: {window.focus}")
        print(f"   Products: {', '.join(window.products)}")
        print()

    # Generate HTML calendar
    print("\n" + "=" * 80)
    print("🌐 GENERATING HTML CALENDAR")
    print("=" * 80)
    print()

    html = render_yearly_calendar_html(farm_calendar)
    output_path = Path("demos/output/yearly_calendar.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"✓ HTML calendar saved to: {output_path}")
    print(f"✓ Open file://{output_path.absolute()} in your browser")
    print()

    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 CALENDAR STATISTICS")
    print("=" * 80)
    print()

    total_events = len(farm_calendar.events)
    active_events = len(farm_calendar.get_active_events())
    upcoming_events = len(farm_calendar.get_upcoming_events(30))
    critical_events = len(farm_calendar.get_critical_deadlines())

    print(f"Total events in calendar: {total_events}")
    print(f"Currently active: {active_events}")
    print(f"Upcoming (next 30 days): {upcoming_events}")
    print(f"Critical deadlines: {critical_events}")
    print()

    # Event type distribution
    print("Event type distribution:")
    for event_type in EventType:
        count = len(farm_calendar.get_events_by_type(event_type))
        if count > 0:
            percentage = (count / total_events) * 100
            bar = "█" * int(percentage / 5)
            print(f"   {event_type.value.ljust(15)}: {count:2d} {bar} {percentage:.0f}%")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
